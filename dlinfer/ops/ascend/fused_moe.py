# Copyright (c) 2024, DeepLink. All rights reserved.
"""Ascend fused MoE using All2All EP implementation."""
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

try:
    from ..moe.common_fused_moe import forward_impl_all2all
except ImportError:
    # Fallback if moe module not available
    forward_impl_all2all = None


def fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    moe_config,
    ep_group: Optional[dist.ProcessGroup] = None,
    ep_rank: Optional[int] = None,
    ep_size: Optional[int] = None,
    expert_kernel,
    expert_kwargs: Optional[Dict[str, Any]] = None,
    apply_router_weight_on_input: bool = False,
    normalize_router_logits: bool = False,
    quant_type: Any = None,
    bias: Optional[torch.Tensor] = None,
):
    """
    Ascend fused MoE using All2All EP path.
    
    This implementation replaces the previous Ascend MoE flow with an All2All
    expert-parallel approach adapted from vllm-ascend.
    
    Args:
        hidden_states: Input hidden states [N_tokens, H_total]
        router_logits: Router logits for expert selection
        moe_config: MoE configuration with num_experts, num_local_experts, experts_per_token
        ep_group: Expert parallel process group (defaults to world group if None)
        ep_rank: Current rank in EP group (inferred if None)
        ep_size: Size of EP group (inferred if None)
        expert_kernel: Function to compute expert outputs
        expert_kwargs: Optional kwargs for expert_kernel
        apply_router_weight_on_input: Whether to apply router weights before expert computation
        normalize_router_logits: Whether to normalize router logits with softmax
        quant_type: Optional quantization type
        bias: Optional bias to add after finalize
        
    Returns:
        Output tensor after MoE computation
    """
    if forward_impl_all2all is None:
        raise RuntimeError("All2All MoE implementation not available")
    
    # Default to world group if no EP group specified
    if ep_group is None:
        if dist.is_initialized():
            ep_group = dist.group.WORLD
        else:
            # Single-process fallback
            ep_group = None
    
    # Infer EP rank and size if not provided
    if ep_rank is None or ep_size is None:
        if ep_group is not None and dist.is_initialized():
            ep_rank = dist.get_rank(ep_group)
            ep_size = dist.get_world_size(ep_group)
        else:
            # Single-process fallback
            ep_rank = 0
            ep_size = 1
    
    return forward_impl_all2all(
        hidden_states,
        router_logits,
        moe_config=moe_config,
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
        expert_kernel=expert_kernel,
        expert_kwargs=expert_kwargs,
        apply_router_weight_on_input=apply_router_weight_on_input,
        normalize_router_logits=normalize_router_logits,
        quant_type=quant_type,
        bias=bias,
    )
