from __future__ import annotations
from typing import Any, Dict, Optional

import torch

from .moe_comm_method_all2all import AlltoAllCommImpl


def forward_impl_all2all(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    moe_config,
    ep_group,
    ep_rank: int,
    ep_size: int,
    expert_kernel,
    expert_kwargs: Optional[Dict[str, Any]] = None,
    apply_router_weight_on_input: bool = False,
    normalize_router_logits: bool = False,
    quant_type: Any = None,
    bias: Optional[torch.Tensor] = None,
):
    comm = AlltoAllCommImpl(moe_config, ep_group=ep_group, ep_rank=ep_rank, ep_size=ep_size)
    hs_for_experts, router_topk, mc2_mask, context = comm.prepare(
        hidden_states,
        router_logits,
        apply_router_weight_on_input=apply_router_weight_on_input,
        normalize_router_logits=normalize_router_logits,
        quant_type=quant_type,
    )

    mlp_out_partitioned = comm.fused_experts(
        hs_for_experts, expert_kernel=expert_kernel, expert_kwargs=expert_kwargs
    )

    final = comm.finalize(
        mlp_out_partitioned, reduce_results=True, context_metadata=context, bias=bias
    )
    return final
