import torch
from pathlib import Path
from safetensors.torch import load_file
import logging


StateDict = dict[str, torch.Tensor]

logger = logging.getLogger(__name__)


def partial_load_safetensors(
    model_path: str,
    dtype: torch.dtype,
    keys_to_ignore: list[str] | None = None,
) -> StateDict:
    state_dict: StateDict = {}
    mp = Path(model_path)

    logger.info(f"Partial load safetensors with filter: {keys_to_ignore}")

    for file in mp.iterdir():
        if file.suffix != ".safetensors":
            continue

        # Load shard to cpu first
        logger.info(f"Loading shard from {file}...")
        shard = load_file(file, device="cpu")

        # Filter out ignored layers with keys
        filtered_shard = filter_layers(shard, dtype, keys_to_ignore)
        state_dict.update(filtered_shard)

        del shard, filtered_shard

    return state_dict


def filter_layers(
    shard: StateDict,
    dtype: torch.dtype,
    keys_to_ignore: list[str] | None = None,
) -> StateDict:
    if keys_to_ignore is None:
        return shard

    return {
        layer_name: weight.to(dtype)
        for layer_name, weight in shard.items()
        if not any(k in layer_name for k in keys_to_ignore)
    }
