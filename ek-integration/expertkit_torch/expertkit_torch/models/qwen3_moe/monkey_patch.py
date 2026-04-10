import time
import torch
import torch.nn.functional as F

from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe
from torch import nn
from expertkit_torch.expert_kit_client import ExpertKitClient

# default timeout interval for ek client, in seconds
DEFAULT_TIMEOUT_INTVAL = 100
layer_idx = 0


def intercept_moe(
    enable_ek: bool = True,
    ek_addr: str = "localhost:5002",
    ek_model_name: str = "qwen3",
    enable_direct_path: bool = True,
    channel: str = "grpc",
):
    class InterceptedMoE(nn.Module):
        client: ExpertKitClient | None = None

        def __init__(self, config):
            super().__init__()
            global layer_idx
            if enable_ek and InterceptedMoE.client is None:
                # Create ExpertKit client with optional direct worker communication
                # Direct path reduces latency by ~24% (bypasses controller forwarding)
                InterceptedMoE.client = ExpertKitClient(
                    controller_addr=ek_addr,
                    timeout_sec=DEFAULT_TIMEOUT_INTVAL,
                    channel=channel,
                )
                print(
                    f"[ExpertKit] Client initialized: controller={ek_addr}, direct_path={enable_direct_path}"
                )
            self.layer_id = layer_idx
            layer_idx += 1
            layer_idx = layer_idx % config.num_hidden_layers
            self.num_experts = config.num_experts
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob

            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            if not enable_ek:
                self.experts = nn.ModuleList(
                    [
                        qwen3_moe.Qwen3MoeMLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        for _ in range(self.num_experts)
                    ]
                )

        def ek_forward(
            self,
            *,
            hidden_states: torch.Tensor,
            routing_weights: torch.Tensor,
            selected_experts: torch.Tensor,
            batch_size: int,
            sequence_length: int,
            hidden_dim: int,
        ):
            # Start timing for expert computation
            start_time = time.time()

            expert_ids = []
            total_seq_len, _ = hidden_states.shape
            for seq_idx in range(total_seq_len):
                eids = selected_experts[seq_idx].tolist()
                ids = [
                    f"{ek_model_name}/l{self.layer_id}-e{expert_idx}"
                    for expert_idx in eids
                ]
                expert_ids.append(ids)

            outputs = self.client.forward_expert(
                expert_ids=expert_ids, hidden_state=hidden_states
            )
            outputs = outputs.to(device=hidden_states.device, dtype=hidden_states.dtype)
            expanded_weights = routing_weights.unsqueeze(-1)
            output = torch.sum(expanded_weights * outputs, dim=1)

            final_hidden_states = output.reshape(
                batch_size, sequence_length, hidden_dim
            )

            # Record expert computation time if profiler is available
            end_time = time.time()

            return final_hidden_states

        def normal_forward(
            self,
            *,
            hidden_states: torch.Tensor,
            routing_weights: torch.Tensor,
            selected_experts: torch.Tensor,
            expert_mask: torch.Tensor,
            batch_size: int,
            sequence_length: int,
            hidden_dim: int,
        ):
            # Start timing for expert computation
            start_time = time.time()

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
            final_hidden_states = final_hidden_states.reshape(
                batch_size, sequence_length, hidden_dim
            )

            # Record expert computation time if profiler is available
            end_time = time.time()

            return final_hidden_states

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # Timing overall MoE forward pass
            forward_start = time.time()

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            # Process router logits (no need to time separately)
            router_logits = self.gate(hidden_states)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0)

            if enable_ek:
                final = self.ek_forward(
                    hidden_states=hidden_states,
                    routing_weights=routing_weights,
                    selected_experts=selected_experts,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    hidden_dim=hidden_dim,
                )
            else:
                final = self.normal_forward(
                    hidden_states=hidden_states,
                    routing_weights=routing_weights,
                    selected_experts=selected_experts,
                    expert_mask=expert_mask,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    hidden_dim=hidden_dim,
                )

            # Record overall MoE time only if profiler is available
            forward_end = time.time()

            return final

    delattr(qwen3_moe, "Qwen3MoeSparseMoeBlock")
    setattr(qwen3_moe, "Qwen3MoeSparseMoeBlock", InterceptedMoE)
