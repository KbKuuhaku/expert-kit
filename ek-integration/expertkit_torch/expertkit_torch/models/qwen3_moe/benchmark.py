from functools import partial
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig
import os
import argparse
import logging
import torch
from expertkit_torch.models.qwen3_moe.monkey_patch import intercept_moe

logger = logging.getLogger(__name__)

LOG_LEVEL_MAPPING: dict[str, int] = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

# The default device should be set according to the environment.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

BATCH_SIZES: list[int] = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
]
SEED: int = 42


class Qwen3MoEBenchmark:
    _MODEL_SINGLETON: Qwen3MoeForCausalLM | None = None
    VOCAB_SIZE: int

    def __init__(
        self,
        batch_size: int,
        model_path: str,
        enable_ek: bool,
        ek_model_name: str,
        ek_addr: str,
        enable_direct_path: bool,
        channel: str,
    ) -> None:
        torch.manual_seed(SEED)
        self.model_path = model_path
        self.intercept_moe = partial(
            intercept_moe,
            enable_ek=enable_ek,
            ek_addr=ek_addr,
            ek_model_name=ek_model_name,
            enable_direct_path=enable_direct_path,
            channel=channel,
        )
        self.batch_size = batch_size
        self.model: Qwen3MoeForCausalLM = self.load_model()

    def load_model(self) -> Qwen3MoeForCausalLM:
        # Load model from a singleton
        if Qwen3MoEBenchmark._MODEL_SINGLETON is not None:
            return Qwen3MoEBenchmark._MODEL_SINGLETON

        logger.info(f"Loading model from {self.model_path}")
        self.intercept_moe()

        config = Qwen3MoeConfig.from_pretrained(self.model_path)
        config.num_hidden_layers = 1  # load only one layer
        Qwen3MoEBenchmark.VOCAB_SIZE = (
            int(config.vocab_size) if config.vocab_size else 0
        )

        model = Qwen3MoeForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            dtype=config.dtype,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        Qwen3MoEBenchmark._MODEL_SINGLETON = model
        print(model)

        return model

    @torch.inference_mode()
    def run(self, seq_len: int = 1) -> None:
        input_ids = torch.randint(
            Qwen3MoEBenchmark.VOCAB_SIZE,
            (self.batch_size, seq_len),
            device=self.model.device,
        )
        attention_mask = torch.ones_like(
            input_ids,
            device=self.model.device,
        )  # not masking anything

        self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
        )


def get_iterations(batch_size: int) -> int:
    if batch_size < 128:
        return 50
    if batch_size < 1024:
        return 20
    return 5


def main(args: argparse.Namespace) -> None:
    for batch_size in BATCH_SIZES:
        logger.info(f"Benchmarking on batch size = {batch_size}...")
        benchmark = Qwen3MoEBenchmark(
            batch_size=batch_size,
            model_path=args.model_path,
            enable_ek=args.enable_ek,
            ek_model_name=args.ek_model_name,
            ek_addr=args.ek_addr,
            enable_direct_path=args.ek_direct_path,
            channel=args.channel,
        )
        for i in range(get_iterations(batch_size)):
            logger.debug(f"Batch size: {batch_size}; iteration: {i}")
            benchmark.run()


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL_MAPPING[os.getenv("LOG_LEVEL", "info")])

    parser = argparse.ArgumentParser()
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
        "--ek_direct_path",
        action=argparse.BooleanOptionalAction,
        default=True,  # Enabled by default - implements controller's decomposition logic
        help="Enable direct worker communication (bypasses controller forwarding). "
        "Implements request decomposition to match worker's expected format.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="grpc",
        help="The number of prompts to use for evaluation.",
    )
    args = parser.parse_args()
    main(args)
