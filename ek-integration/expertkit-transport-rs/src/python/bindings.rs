use std::fs;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use tracing_appender::non_blocking::WorkerGuard;

use crate::client::ExpertKitClient as RustExpertKitClient;
use crate::utils::{TensorMetadata, pytorch_to_tch_tensor, tch_to_pytorch_tensor};
use tracing::Level;
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt};

const DEFAULT_THREAD_NUM: usize = 16;

fn init_tracing_subscriber_with_json_writer(channel: &str) -> Option<WorkerGuard> {
    log::info!(
        "Initializing tracing subscriber in expertkit-transport-rs/src/client.rs... (Channel: {channel}..."
    );

    // NOTE: Hardcode output directory for tracer JSON
    let output_dir = format!("benchmark_traces/{channel}");
    if let Err(e) = fs::create_dir_all(&output_dir) {
        log::warn!(
            "Unable to create {output_dir} and initialize tracing subscriber, abort ({e:?})"
        );
        return None;
    }
    // Create the output json file
    let filename = format!("{}/{}.json", output_dir, "client");
    log::info!("Creating JSON tracing log file {filename}...");
    let file = match fs::File::create(filename) {
        Ok(file) => file,
        Err(e) => {
            log::warn!("Unable to create file, abort ({e:?}).");
            return None;
        }
    };

    // Start a non-block writer storing JSON in background thread
    let (non_blocking_writer, non_blocking_guard) = tracing_appender::non_blocking(file);

    // Ref: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/fmt/struct.Layer.html#method.with_span_events
    let result = tracing_subscriber::registry()
        .with(tracing_subscriber::filter::LevelFilter::from_level(
            Level::INFO,
        ))
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_span_list(false) // disable the "spans" field in json
                .with_current_span(true) // enable the "span" field in json
                .with_span_events(FmtSpan::CLOSE) // record the duration
                .with_writer(non_blocking_writer),
        )
        .try_init();
    match result {
        Ok(()) => Some(non_blocking_guard),
        Err(e) => {
            log::warn!("Unable to initialize tracing subscriber ({e:?}).");
            return None;
        }
    }
}

/// High-level ExpertKit client with routing and batching
#[pyclass]
pub struct PyExpertKitClient {
    _guard: Option<WorkerGuard>,
    client: Option<RustExpertKitClient>,
    runtime: Option<tokio::runtime::Runtime>, // Shared runtime for all requests
}

#[pymethods]
impl PyExpertKitClient {
    #[new]
    fn new(
        controller_addr: String,
        timeout_sec: Option<f64>,
        channel: Option<&str>,
    ) -> PyResult<Self> {
        if env_logger::try_init().is_ok() {
            log::info!("Logger initialized");
        }

        let timeout = timeout_sec.unwrap_or(2.0);

        // Create ONE shared Tokio runtime for all requests
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(DEFAULT_THREAD_NUM)
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create runtime: {}",
                    e
                ))
            })?;

        Ok(Self {
            _guard: init_tracing_subscriber_with_json_writer(channel.unwrap_or("grpc")),
            client: Some(RustExpertKitClient::new(controller_addr, timeout)),
            runtime: Some(runtime),
        })
    }

    /// Connect to controller and fetch routing table
    fn connect(&mut self, py: Python) -> PyResult<()> {
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        py.allow_threads(|| {
            runtime
                .block_on(async { client.connect().await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Forward expert computation
    fn forward_expert<'py>(
        &self,
        py: Python<'py>,
        expert_ids: Vec<Vec<String>>,
        hidden_state: &PyAny,
    ) -> PyResult<PyObject> {
        let t = std::time::Instant::now();
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        // Extract tensor metadata from Python
        let metadata = TensorMetadata::from_pytorch(hidden_state)?;

        log::debug!(
            "[PyBinding-Time] 🚗 Runtime get and extracted tensor metadata in {:?} μs",
            t.elapsed().as_micros()
        );

        // Convert PyTorch tensor to tch::Tensor
        let t = std::time::Instant::now();
        let input_tensor = pytorch_to_tch_tensor(hidden_state, &metadata)?;
        log::debug!(
            "[PyBinding-Time] 🚗 Converted PyTorch tensor to tch::Tensor in {:?} μs",
            t.elapsed().as_micros()
        );

        // Release GIL and process
        let t = std::time::Instant::now();
        let output_tensor = py.allow_threads(|| {
            runtime
                .block_on(async { client.forward_expert_tensor(expert_ids, input_tensor).await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        log::debug!(
            "[PyBinding-Time] 🚗 Processed tensor in {:?} μs",
            t.elapsed().as_micros()
        );

        // Convert tch::Tensor back to PyTorch tensor
        let t = std::time::Instant::now();
        let final_tensor = tch_to_pytorch_tensor(py, &output_tensor, &metadata.device_str)?;
        log::debug!(
            "[PyBinding-Time] 🚗 Converted tch::Tensor back to PyTorch tensor in {:?} μs",
            t.elapsed().as_micros()
        );

        Ok(final_tensor.into())
    }

    /// Refresh routing table
    fn refresh_routing(&self, py: Python) -> PyResult<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        py.allow_threads(|| {
            runtime
                .block_on(async { client.refresh_routing().await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}
