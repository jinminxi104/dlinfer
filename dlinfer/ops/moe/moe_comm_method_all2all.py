# Copyright (c) 2024, DeepLink. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from .prepare_finalize_all2all import PrepareAndFinalizeWithAll2All


class AlltoAllCommImpl:
    def __init__(self, moe_config, *, ep_group, ep_rank: int, ep_size: int):
        self.moe_config = moe_config
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self._pf = PrepareAndFinalizeWithAll2All(self.moe_config)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        apply_router_weight_on_input: bool = False,
        normalize_router_logits: bool = False,
        quant_type: Any = None,
    ):
        return self._pf.prepare(
            hidden_states,
            router_logits,
            apply_router_weight_on_input,
            normalize_router_logits,
            quant_type,
            ep_group=self.ep_group,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
        )

    def fused_experts(
        self,
        hidden_states_for_experts: torch.Tensor,
        *,
        expert_kernel,
        expert_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        expert_kwargs = expert_kwargs or {}
        return expert_kernel(hidden_states_for_experts, **expert_kwargs)

    def finalize(
        self,
        mlp_out_partitioned: torch.Tensor,
        *,
        reduce_results: bool = True,
        context_metadata: Dict[str, Any],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._pf.finalize(mlp_out_partitioned, reduce_results, context_metadata, bias=bias)
