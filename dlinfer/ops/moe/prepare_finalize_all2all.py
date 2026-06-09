from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .token_dispatcher_all2all import TokenDispatcherWithAll2AllV


class PrepareAndFinalizeWithAll2All:
    def __init__(self, moe_config) -> None:
        self.moe_config = moe_config

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        apply_router_weight_on_input: bool = False,
        normalize_router_logits: bool = False,
        quant_type: Any = None,
        *,
        ep_group,
        ep_rank: int,
        ep_size: int,
    ):
        probs = F.softmax(router_logits, dim=-1) if normalize_router_logits else router_logits
        top_k = self.moe_config.experts_per_token
        topk_weights, topk_ids = torch.topk(probs, k=top_k, dim=-1)

        dispatcher = TokenDispatcherWithAll2AllV(
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            top_k=top_k,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        td_out = dispatcher.token_dispatch(hidden_states, topk_weights, topk_ids, expert_map=None)

        context = td_out.get("context", {})
        context["dispatcher"] = dispatcher
        context["router_topk_weights"] = topk_weights
        context["router_topk_ids"] = topk_ids

        return td_out["hidden_states"], topk_weights, None, context

    def finalize(
        self,
        hidden_states_after_mlp: torch.Tensor,
        reduce_results: bool,
        context_metadata: Dict[str, Any],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dispatcher: TokenDispatcherWithAll2AllV = context_metadata["dispatcher"]
        return dispatcher.token_combine(hidden_states_after_mlp, context_metadata, bias=bias)
