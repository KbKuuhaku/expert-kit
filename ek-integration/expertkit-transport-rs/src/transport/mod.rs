use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use async_trait::async_trait;

pub mod auto;
pub mod grpc;
pub mod shm;

#[cfg(feature = "rdma")]
pub mod rdma;

// Re-export WorkerEndpoint from grpc proto
pub use grpc::proto::ek::control::v1::WorkerEndpoint;

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_request_id() -> u64 {
    REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub enum TransportType {
    Grpc,
    SharedMemory,
    Nvshmem,
}

/// Request for a single expert computation
#[derive(Debug, Clone)]
pub struct ExpertRequest {
    pub batch_id: u64,   // unique id for the batch
    pub request_id: u64, // unique id for the request
    pub expert_id: String,
    pub tensor_data: Vec<u8>, // Safetensors blob containing batched sequences
    pub num_sequences: usize, // Number of sequences in this batch
}

impl ExpertRequest {
    /// Create a new expert request with a batched tensor
    pub fn new(
        batch_id: u64,
        expert_id: String,
        tensor_data: Vec<u8>,
        num_sequences: usize,
    ) -> Self {
        Self {
            batch_id,
            request_id: next_request_id(),
            expert_id,
            tensor_data,
            num_sequences,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpertResponse {
    #[allow(unused)]
    pub expert_id: String,
    pub tensor_data: Vec<u8>, // Safetensors blob with output sequences
}

/// Transport abstraction for expert communication
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send batch of expert requests to a worker
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>>;

    /// Get transport type
    #[allow(unused)]
    fn transport_type(&self) -> TransportType;

    /// Check if transport is available for endpoint
    #[allow(unused)]
    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool;
}
