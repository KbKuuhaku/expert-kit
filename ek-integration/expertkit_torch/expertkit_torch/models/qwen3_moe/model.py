# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Modifications Copyright (c) 2025 expertkit-torch.
#
# This file is based on code from the Qwen3 project (originally licensed under Apache 2.0)
# and has been modified.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import torch

from typing import Optional, Dict, Any
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.utils.logging import set_verbosity_error

from expertkit_torch.utils.profiler_manager import ProfilerManager
from expertkit_torch.models.qwen3_moe.monkey_patch import intercept_moe

set_verbosity_error()

# The default device should be set according to the environment.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForCausalLM] = None


def evaluate_batch(
    *,
    model_path="./",
    prompts="What is MoE Model?",
    output_max_length=64,
    enable_ek=True,
    ek_addr="localhost:5002",
    ek_model_name="qwen3",
    enable_direct_path=True,
    channel: str = "grpc",
) -> Dict[str, Any]:
    """
    Batch inference with performance profiling.

    Args:
        model_path: Path to the pretrained model
        prompts: List of prompt strings for batch processing
        enable_ek: Whether to enable expert knowledge

    Returns:
        Dictionary containing results and performance metrics
    """
    if prompts is None:
        prompts = ["What is MoE Model?"]

    # Convert str to list
    if isinstance(prompts, str):
        prompts = [prompts]

    # First intercept the MoE module - completely independent of profiling
    intercept_moe(
        enable_ek=enable_ek,
        ek_addr=ek_addr,
        ek_model_name=ek_model_name,
        enable_direct_path=enable_direct_path,
        channel=channel,
    )

    # Load the tokenizer and the model only once
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
        )
    if model is None:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        print(model)

    # Initialize profiler manager with context manager
    with ProfilerManager(batch_size=len(prompts)) as profiler:
        # Wrap model with profiler - completely non-invasive
        profiler.wrap_model(model)

        # Prepare batch messages
        batch_messages = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            batch_messages.append(text)

        # Tokenize batch inputs with padding
        model_inputs = tokenizer(
            batch_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Generate responses - profiling happens automatically via hooks
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=output_max_length,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Process generated sequences
        results = []
        for i in range(len(prompts)):
            # Extract output tokens
            input_length = len(model_inputs.input_ids[i])
            output_ids = generated_ids[i][input_length:].tolist()

            # Remove padding tokens
            if tokenizer.pad_token_id is not None:
                output_ids = [
                    token_id
                    for token_id in output_ids
                    if token_id != tokenizer.pad_token_id
                ]

            # Extract thinking content
            thinking_finish = False
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_finish = True
            except ValueError:
                # Thinking not finished
                index = len(output_ids) - 1

            thinking_content = tokenizer.decode(
                output_ids[:index], skip_special_tokens=True
            ).strip("\n")

            content = tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")

            results.append(
                {
                    "prompt": prompts[i],
                    "thinking_content": thinking_content,
                    "content": content,
                    "input_tokens": len(model_inputs.input_ids[i]),
                    "output_tokens": len(output_ids),
                }
            )

        # Context manager exit will automatically unwrap the model and print the report
        return {"results": results, "performance": profiler.report()}


def sharegpt(path, max_prompt_len=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for item in data:
        for conversation in item["conversations"]:
            if conversation["from"] == "human":
                if max_prompt_len is not None:
                    prompts.append(conversation["value"][:max_prompt_len])
                else:
                    prompts.append(conversation["value"])
    return prompts


def main():
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
        default="qwen3",
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
        "--detail_profile",
        action="store_true",
        help="Enable detailed profiling of model components (attention vs expert).",
    )
    parser.add_argument(
        "--output_max",
        type=int,
        default=64,
        help="The maximum output length for the model.",
    )
    parser.add_argument(
        "--dataset",
        choices=["none", "sharegpt"],
        default="none",
        help="The dataset to use for evaluation.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--print_response",
        action="store_true",
        help="Print the response content.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=None,
        help="Maximum length of each prompt (applicable for ShareGPT dataset).",
    )
    parser.add_argument(
        "--prompt_num",
        type=int,
        default=512,
        help="The number of prompts to use for evaluation.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="grpc",
        help="The number of prompts to use for evaluation.",
    )
    args = parser.parse_args()

    if args.dataset == "none":
        # Use default prompts if no dataset is specified
        test_prompts = [
            "What is MoE Model?",
            "Explain the benefits of mixture of experts.",
            "How does MoE improve model efficiency?",
            "Compare MoE with dense models.",
        ] * args.prompt_num
        test_prompts = test_prompts[: args.prompt_num]
    elif args.dataset == "sharegpt":
        # Validate that dataset_path is provided
        if args.dataset_path is None:
            raise ValueError(
                "You must provide --dataset_path when using the 'sharegpt' dataset."
            )
        # Load prompts from ShareGPT dataset
        test_prompts = sharegpt(args.dataset_path, max_prompt_len=args.max_prompt_len)
        if len(test_prompts) < args.prompt_num:
            test_prompts *= (args.prompt_num // len(test_prompts)) + 1
        test_prompts = test_prompts[: args.prompt_num]
    else:
        raise ValueError("Invalid dataset specified.")

    test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    test_batch_sizes = [1]
    aggregated_results = []
    for batch_size in test_batch_sizes:
        for prompts in range(0, len(test_prompts), batch_size):
            if prompts / batch_size >= 6:
                break
            batch_result = evaluate_batch(
                model_path=args.model_path,
                prompts=test_prompts[prompts : prompts + batch_size],
                enable_ek=args.enable_ek,
                ek_addr=args.ek_addr,
                ek_model_name=args.ek_model_name,
                enable_direct_path=args.ek_direct_path,
                output_max_length=args.output_max,
                channel=args.channel,
            )
            aggregated_results.extend(batch_result["results"])

    if args.print_response:
        for result in aggregated_results[:5]:
            print()
            print(f"Prompt: {result['prompt']}")
            print(f"Thinking Content: {result['thinking_content']}")
            print(f"Response: {result['content']}")
            print(
                f"Input Tokens: {result['input_tokens']}, Output Tokens: {result['output_tokens']}"
            )
            print("-" * 40)


if __name__ == "__main__":
    main()
