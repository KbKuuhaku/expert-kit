import json
from sympy.physics.units import action
import time
from functools import partial
from transformers import (
    Qwen3MoeForCausalLM,
    Qwen3MoeConfig,
    AutoModelForCausalLM,
    AutoConfig,
)
import os
import argparse
import logging
import torch
from expertkit_torch.models.qwen3_moe.monkey_patch import intercept_moe
from expertkit_torch.models.qwen3_moe.load import partial_load_safetensors

from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)

LOG_LEVEL_MAPPING: dict[str, int] = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

DTYPE_MAPPING: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# The default device should be set according to the environment.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

SEED: int = 42


# class PerfRecord(BaseModel):
#     batch_size: int
#     channel: str = "grpc"
#     iter_idx: int = Field(serialization_alias="iter")
#     latency: float = Field(exclude=True)
#
#     @computed_field
#     @property
#     def latency_us(self) -> float:
#         return self.latency * 1000 * 1000


class Qwen3MoEBenchmark:
    _MODEL_SINGLETON: Qwen3MoeForCausalLM | None = None
    VOCAB_SIZE: int

    def __init__(
        self,
        batch_size: int,
        input_len: int,
        output_len: int,
        model_path: str,
        enable_ek: bool,
        ek_model_name: str,
        ek_addr: str,
        channel: str,
        dtype: str,
    ) -> None:
        torch.manual_seed(SEED)
        self.model_path = model_path
        self.intercept_moe = partial(
            intercept_moe,
            enable_ek=enable_ek,
            ek_addr=ek_addr,
            ek_model_name=ek_model_name,
        )
        self.batch_size = batch_size
        self.dtype = DTYPE_MAPPING[dtype]

        self.model: Qwen3MoeForCausalLM = self.load_model()
        print(self.model)

        self.input_len = input_len
        self.output_len = output_len
        self.channel = channel

    def load_model(self) -> Qwen3MoeForCausalLM:
        # Load model from a singleton
        if Qwen3MoEBenchmark._MODEL_SINGLETON is not None:
            return Qwen3MoEBenchmark._MODEL_SINGLETON

        logger.info(f"Loading model from {self.model_path}")
        self.intercept_moe()

        config = AutoConfig.from_pretrained(
            self.model_path,
        )
        Qwen3MoEBenchmark.VOCAB_SIZE = (
            int(config.vocab_size) if config.vocab_size else 0
        )
        if config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

        with torch.device("meta"):
            model: Qwen3MoeForCausalLM = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=self.dtype,
            )
        model.to_empty(device="cpu")

        missing, unexpected = model.load_state_dict(
            partial_load_safetensors(
                self.model_path,
                dtype=self.dtype,
                keys_to_ignore=["mlp.expert"],
            ),
            strict=False,
            assign=True,  # create new mem instead of in-place
        )
        logger.info(f"Missing keys: {len(missing)}")
        logger.info(f"Unexpected keys: {len(unexpected)}")

        Qwen3MoEBenchmark._MODEL_SINGLETON = model.to(device, dtype=self.dtype)
        Qwen3MoEBenchmark._MODEL_SINGLETON.eval()

        return Qwen3MoEBenchmark._MODEL_SINGLETON

    def warmup(self, warmup_iter) -> None:
        for i in range(warmup_iter):
            logger.info(f"Warmup - Batch size: {self.batch_size}; iteration: {i}")
            self.run()

    def bench(self, num_iter) -> list[float]:
        records: list[float] = []
        # Benchmark
        for i in range(num_iter):
            logger.info(f"Benchmark - Batch size: {self.batch_size}; iteration: {i}")
            latency = self.run(self.output_len)

            records.append(latency)

        return records

    @torch.inference_mode()
    def run(self, output_len: int = 1) -> float:
        input_ids = torch.randint(
            low=0,
            high=Qwen3MoEBenchmark.VOCAB_SIZE,
            size=(self.batch_size, self.input_len),
            device=self.model.device,
        )
        attention_mask = torch.ones_like(
            input_ids,
            device=self.model.device,
        )  # not masking anything

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # 1. Set max/min new tokens to the same value to make sure model always output the same length
        # even though it faces EOS
        # 2. Disable sampling and make beam width to 1 (since we don't care accuracy)
        # 3. Enable caching
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=output_len,
            min_new_tokens=output_len,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=self.model.config.pad_token_id,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start_time

        # Sanity check on the correctness
        # assert outputs.shape[1] == self.input_len + output_len
        # print(
        #     f"output shape ({outputs.shape[1]}) <- input ({self.input_len}) + output ({output_len})"
        # )

        return latency


def main(args: argparse.Namespace) -> None:
    logger.info(f"Benchmarking on batch size = {args.batch_size}...")
    benchmark = Qwen3MoEBenchmark(
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
        model_path=args.model_path,
        enable_ek=args.enable_ek,
        ek_model_name=args.ek_model_name,
        ek_addr=args.ek_addr,
        channel=args.channel,
        dtype=args.dtype,
    )

    benchmark.warmup(args.warmup)

    latencies = benchmark.bench(args.num_iter)

    if args.output_json is None:
        return

    from pathlib import Path

    file = Path(args.output_json)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"latencies": latencies}))

    logger.info(f"Written benchmark logs to {args.output_json}!")


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL_MAPPING[os.getenv("LOG_LEVEL", "info")])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size of the payload.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup steps before benchmark",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=10,
        help="Number of iterations for this benchmark",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="JSON file path for logging the latency records",
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=1,
        help="Length of the input prompt",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        default=1,
        help="Length of the generated output tokens",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--enable_ek",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ExpertKit.",
    )
    parser.add_argument(
        "--ek_model_name",
        type=str,
        default="Qwen3-30B-A3B",
        help="The name of the model used in ExpertKit.",
    )
    parser.add_argument(
        "--ek_addr",
        type=str,
        default="localhost:5002",
        help="The address of the ExpertKit controller.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="grpc",
        help="The number of prompts to use for evaluation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="Data type of model/tensor",
    )
    args = parser.parse_args()
    main(args)
