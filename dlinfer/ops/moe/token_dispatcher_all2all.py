# Copyright (c) 2024, DeepLink. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

try:
    import torch_npu  # Ascend NPU
except Exception:
    torch_npu = None


class TokenDispatcherWithAll2AllV:
    """
    All-to-All based token dispatcher working on sequence level.
    - Dispatch on entire sequence; partition hidden dim.
    - Requires EP (expert-parallel) process group.
    """

    def __init__(
        self,
        *,
        ep_group: dist.ProcessGroup,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        num_experts: int,
        num_local_experts: int,
        apply_router_weight_on_input: bool = False,
    ) -> None:
        assert num_local_experts > 0, "Expected at least one local expert"
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Expert mapping on this EP rank: continuous local expert ids
        offset = self.ep_rank * self.num_local_experts
        self.local_expert_indices = [offset + i for i in range(self.num_local_experts)]
        assert len(self.local_expert_indices) == self.num_local_experts

        self._sorted_topk_ids: Optional[torch.Tensor] = None
        self._original_shape: Optional[Tuple[int, int]] = None

    @property
    def original_shape(self) -> Optional[Tuple[int, int]]:
        return self._original_shape

    @original_shape.setter
    def original_shape(self, shape: Tuple[int, int]) -> None:
        self._original_shape = shape

    def _async_all_to_all(
        self, tensor: torch.Tensor
    ) -> Tuple[Optional[Any], torch.Tensor, Optional[Any]]:
        """
        Perform all-to-all on EP group for 2D tensor.
        Returns (req, out, handle) for API parity.
        """
        rows = tensor.size(0)
        assert rows % self.ep_size == 0, "Rows must be divisible by ep_size"
        in_splits = [rows // self.ep_size] * self.ep_size
        out_splits = in_splits
        out = torch.empty_like(tensor)
        dist.all_to_all_single(
            out, tensor, out_splits=out_splits, in_splits=in_splits, group=self.ep_group
        )
        return None, out, None

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,          # [N_tokens, H_total]
        topk_weights: torch.Tensor,           # [N_tokens, top_k]
        topk_ids: torch.Tensor,               # [N_tokens, top_k] global expert ids
        expert_map: Optional[torch.Tensor] = None,
        *,
        dynamic_eplb: bool = False,
        pertoken_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        n_tokens, h_total = hidden_states.shape
        self._original_shape = (n_tokens, h_total)

        x = hidden_states
        if self.apply_router_weight_on_input:
            scales = topk_weights.reshape(-1, 1).to(x.dtype)
            x = x.repeat_interleave(self.top_k, dim=0) * scales

        expanded_row_idx: Optional[torch.Tensor] = None
        if torch_npu is not None and hasattr(torch_npu, "npu_moe_token_permute"):
            x, expanded_row_idx = torch_npu.npu_moe_token_permute(
                x, topk_ids.reshape(-1).to(torch.int32), self.num_experts
            )
        else:
            flat_ids = topk_ids.reshape(-1)
            sort_idx = torch.argsort(flat_ids)
            x = x.index_select(0, sort_idx)
            expanded_row_idx = sort_idx

        assert h_total % self.ep_size == 0, "hidden size must be divisible by ep_size"
        h_part = h_total // self.ep_size
        h_begin = self.ep_rank * h_part
        h_end = h_begin + h_part
        x_part = x[..., h_begin:h_end].contiguous()

        _, x_part_a2a, _ = self._async_all_to_all(x_part)

        flat_ids = topk_ids.reshape(-1)
        local_min = self.local_expert_indices[0]
        local_max = self.local_expert_indices[-1]
        local_ids = torch.arange(local_min, local_max + 1, device=flat_ids.device)
        num_tokens_per_local = (flat_ids.unsqueeze(-1) == local_ids).to(torch.int32).sum(0)
        group_list = num_tokens_per_local.to(torch.int64).cumsum(dim=0)

        context = {
            "expanded_row_idx": expanded_row_idx,
            "group_list": group_list,
            "topk_weights": topk_weights.reshape(-1),
            "topk_ids": topk_ids.reshape(-1),
            "h_part": h_part,
        }
        return {
            "group_list_type": 0,
            "hidden_states": x_part_a2a,
            "group_list": group_list,
            "topk_scales": topk_weights.reshape(-1, 1),
            "context": context,
        }

    def token_combine(
        self,
        hidden_states: torch.Tensor,   # expert outputs after local MLP, partitioned hidden
        context_metadata: Dict[str, Any],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self._original_shape is not None
        n_tokens, h_total = self._original_shape
        h_part = context_metadata["h_part"]

        rows = hidden_states.size(0)
        assert rows % self.ep_size == 0, "Rows must be divisible by ep_size"
        in_splits = [rows // self.ep_size] * self.ep_size
        out_splits = in_splits
        gathered = torch.empty_like(hidden_states)
        dist.all_to_all_single(
            gathered,
            hidden_states,
            out_splits=out_splits,
            in_splits=in_splits,
            group=self.ep_group,
        )

        if torch_npu is not None and hasattr(torch_npu, "npu_moe_token_unpermute"):
            final = torch_npu.npu_moe_token_unpermute(
                gathered, context_metadata["expanded_row_idx"].to(torch.int32)
            )
        else:
            idx = context_metadata["expanded_row_idx"]
            inv = torch.empty_like(idx)
            inv[idx] = torch.arange(idx.numel(), device=idx.device)
            final = gathered.index_select(0, inv)

        final = final.reshape(n_tokens, self.top_k, -1).sum(dim=1)
        if bias is not None:
            final = final + bias
        return final
