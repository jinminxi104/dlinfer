# All2All Expert-Parallel MoE Implementation

This module provides an All2All based expert-parallel (EP) Mixture of Experts (MoE) implementation adapted from vllm-ascend for use in dlinfer on Ascend NPUs.

## Overview

The All2All EP MoE implementation distributes experts across multiple devices and uses All2All communication to exchange tokens between devices. This approach is particularly suited for large-scale MoE models where the number of experts exceeds what can fit on a single device.

## Architecture

The implementation consists of four main components:

### 1. TokenDispatcherWithAll2AllV (`token_dispatcher_all2all.py`)

Handles token-level dispatch and combine operations:
- **Dispatch**: Sorts tokens by expert ID, partitions hidden dimensions, and performs All2All communication
- **Combine**: Gathers results via All2All, restores original token order, and reduces over top-k experts

Features:
- Sequence-level dispatch (entire sequence processed across EP ranks)
- Hidden dimension partitioning (splits hidden dim across EP ranks)
- NPU-optimized permute/unpermute operations when `torch_npu.npu_moe_token_permute/unpermute` are available
- Fallback to PyTorch `all_to_all_single` for compatibility

### 2. PrepareAndFinalizeWithAll2All (`prepare_finalize_all2all.py`)

High-level prepare and finalize operations:
- **Prepare**: Computes top-k from router logits, creates token dispatcher, and dispatches tokens
- **Finalize**: Combines expert outputs and restores original token order

### 3. AlltoAllCommImpl (`moe_comm_method_all2all.py`)

Communication method wrapper:
- Wraps prepare/finalize operations
- Provides `fused_experts` hook for expert kernel execution
- Manages distributed communication context

### 4. forward_impl_all2all (`common_fused_moe.py`)

End-to-end forward implementation:
1. Prepare: Router computation and token dispatch
2. Expert compute: Apply expert kernels to dispatched tokens
3. Finalize: Combine results and restore order

## Usage

### Basic Usage

```python
from dlinfer.vendor.ascend.torch_npu_ops import fused_moe_all2all

# Define your MoE configuration
class MoEConfig:
    num_experts = 8          # Total experts across all ranks
    num_local_experts = 2    # Experts per rank (num_experts / ep_size)
    experts_per_token = 2    # Top-k routing

# Setup distributed context
import torch.distributed as dist
ep_group = dist.new_group(...)  # Expert-parallel process group
ep_rank = dist.get_rank(ep_group)
ep_size = dist.get_world_size(ep_group)

# Define expert kernel
def expert_kernel(hidden_states, gate_up_weights, down_weights, **kwargs):
    # Your expert computation here
    # hidden_states: [N_tokens_local, H_part]
    return output  # [N_tokens_local, H_part]

# Forward pass
output = fused_moe_all2all(
    hidden_states=hidden_states,    # [N_tokens, H_total]
    router_logits=router_logits,    # [N_tokens, num_experts]
    moe_config=moe_config,
    ep_group=ep_group,
    ep_rank=ep_rank,
    ep_size=ep_size,
    expert_kernel=expert_kernel,
    expert_kwargs={'gate_up_weights': w1, 'down_weights': w2},
    normalize_router_logits=True,
    bias=None,
)
# output: [N_tokens, H_part]  (partitioned by hidden dim)
```

### Expert Kernel Interface

The expert kernel receives partitioned hidden states and should return expert outputs:

```python
def my_expert_kernel(
    hidden_states: torch.Tensor,  # [N_tokens_dispatched, H_part]
    **expert_kwargs
) -> torch.Tensor:
    # Compute expert outputs
    # - hidden_states are already sorted by expert ID
    # - hidden dim is partitioned across EP ranks
    # - Use group_list from context for batched computation if needed
    
    return expert_outputs  # [N_tokens_dispatched, H_part]
```

### Integration with Existing dlinfer Expert Kernels

The All2All implementation can be integrated with existing dlinfer expert kernels:

```python
# Wrap existing expert computation
def wrapped_expert_kernel(hidden_states, gate_up_weights, down_weights, group_list):
    # Use existing npu_grouped_matmul or other expert ops
    up_proj = torch.ops.npu.npu_grouped_matmul(
        [hidden_states],
        [gate_up_weights],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]
    
    # Activation
    gate_cache = silu_and_mul(up_proj, -1)
    
    # Down projection
    down_proj = torch.ops.npu.npu_grouped_matmul(
        [gate_cache],
        [down_weights],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]
    
    return down_proj
```

## Distributed Setup Requirements

### Expert-Parallel (EP) Group

The implementation requires an EP process group where:
- `ep_size` = number of ranks in the EP group
- `num_experts` must be divisible by `ep_size`
- Each rank holds `num_local_experts = num_experts / ep_size` experts

### Hidden Dimension Partitioning

- `H_total` must be divisible by `ep_size`
- Each rank processes `H_part = H_total / ep_size` of the hidden dimension
- All2All exchanges tokens while maintaining this partitioning

### Process Group Setup Example

```python
import torch.distributed as dist

# Initialize distributed
dist.init_process_group(backend='hccl', ...)

# Create EP group (assuming 4 ranks, all participate in EP)
world_size = dist.get_world_size()
ep_group = dist.new_group(ranks=list(range(world_size)))
```

## Performance Considerations

### NPU Optimizations

When `torch_npu` is available, the implementation uses:
- `torch_npu.npu_moe_token_permute`: Optimized token sorting by expert ID
- `torch_npu.npu_moe_token_unpermute`: Optimized token reordering

These operations are hardware-accelerated on Ascend NPUs.

### Fallback Behavior

When NPU operations are unavailable:
- Falls back to PyTorch `torch.argsort` and `index_select` for sorting
- Maintains functional correctness on CPU/GPU
- Performance may be lower without NPU acceleration

## Limitations and Known Issues

1. **Hidden dimension must be divisible by ep_size**: The current implementation partitions the hidden dimension evenly across ranks.

2. **All2All requires balanced token distribution**: Tokens are exchanged via All2All with equal splits. Extreme load imbalance may cause inefficiency.

3. **Router computation is replicated**: Each rank computes the full top-k routing independently. For very large expert counts, this may be a bottleneck.

## Testing

Unit tests are provided in `tests/ops/moe/`:

```bash
# Run token dispatcher tests
python tests/ops/moe/test_token_dispatcher_all2all.py

# Run end-to-end tests
python tests/ops/moe/test_common_fused_moe_all2all.py
```

Note: Tests use mocked distributed operations and do not require actual multi-device setup.

## Migration from Existing fused_moe

The existing `fused_moe` in torch_npu_ops.py uses a different signature:

```python
# Old API (single-device)
output = fused_moe(
    hidden_states,
    gate_up_weights,
    down_weights,
    topk_weights,  # Already computed
    topk_ids,      # Already computed
    topk,
    renormalize,
)

# New API (expert-parallel)
output = fused_moe_all2all(
    hidden_states,
    router_logits,  # Before top-k computation
    moe_config=moe_config,
    ep_group=ep_group,
    ep_rank=ep_rank,
    ep_size=ep_size,
    expert_kernel=expert_kernel,
    expert_kwargs=expert_kwargs,
    normalize_router_logits=True,
)
```

Key differences:
- New API takes `router_logits` instead of pre-computed `topk_weights/topk_ids`
- New API requires distributed context (`ep_group`, `ep_rank`, `ep_size`)
- New API takes an `expert_kernel` function instead of weight tensors directly
- Output is partitioned by hidden dimension in the new API

## References

- Adapted from vllm-ascend TokenDispatcherWithAll2AllV implementation
- Designed for integration with dlinfer's existing expert kernels
- Supports Ascend NPU optimizations via torch_npu operations
