# Integration Guide: All2All EP MoE in dlinfer

This guide provides step-by-step instructions for integrating the All2All Expert-Parallel (EP) MoE implementation into your dlinfer-based models.

## Prerequisites

- dlinfer installed with Ascend NPU support
- Multi-device setup with HCCL for distributed communication
- PyTorch with torch_npu extensions

## Step 1: Configure MoE Parameters

Create a configuration object with your MoE parameters:

```python
class MoEConfig:
    def __init__(self, num_experts=8, experts_per_token=2, ep_world_size=4):
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token  # top-k
        self.num_local_experts = num_experts // ep_world_size
        
        assert num_experts % ep_world_size == 0, \
            f"num_experts ({num_experts}) must be divisible by ep_world_size ({ep_world_size})"

moe_config = MoEConfig(
    num_experts=8,
    experts_per_token=2,
    ep_world_size=4,  # 4 devices in EP group
)
```

## Step 2: Setup Distributed Environment

Initialize the expert-parallel process group:

```python
import torch.distributed as dist

# Initialize distributed backend (HCCL for Ascend)
dist.init_process_group(backend='hccl')

rank = dist.get_rank()
world_size = dist.get_world_size()

# Create EP process group
# Option 1: All ranks participate in EP
ep_group = dist.new_group(ranks=list(range(world_size)))
ep_rank = rank
ep_size = world_size

# Option 2: Subset of ranks for EP (e.g., with tensor parallelism)
# If you have TP=2, EP=4, and 8 total ranks:
# Ranks 0,2,4,6 form one EP group, ranks 1,3,5,7 form another
# tp_rank = rank % 2
# ep_group_ranks = list(range(tp_rank, world_size, 2))
# ep_group = dist.new_group(ranks=ep_group_ranks)
# ep_rank = rank // 2
# ep_size = len(ep_group_ranks)
```

## Step 3: Define Your Expert Kernel

Adapt your existing expert computation into a kernel function:

```python
def expert_kernel_with_npu_ops(
    hidden_states,  # [N_dispatched, H_part]
    gate_up_weights,  # [num_local_experts, 2*intermediate, H_part]
    down_weights,     # [num_local_experts, H_part, intermediate]
    group_list,       # Cumulative token counts per expert
    **kwargs
):
    """
    Expert kernel using NPU grouped matmul operations.
    
    group_list contains cumulative counts of tokens for each local expert.
    This is used by npu_grouped_matmul for efficient batched computation.
    """
    # Up projection with gating
    up_proj = torch.ops.npu.npu_grouped_matmul(
        [hidden_states],
        [gate_up_weights],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,  # Cumulative counts
    )[0]
    
    # Apply SiLU activation and multiply
    gate_cache = torch.ops.npu.npu_swiglu(up_proj, dim=-1)
    
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


# Create a wrapper that extracts group_list from context
def my_expert_kernel(hidden_states, **expert_kwargs):
    # Extract group_list from dispatcher context if available
    context = expert_kwargs.get('context', {})
    group_list = context.get('group_list', None)
    
    return expert_kernel_with_npu_ops(
        hidden_states,
        expert_kwargs['gate_up_weights'],
        expert_kwargs['down_weights'],
        group_list,
    )
```

## Step 4: Integrate into Your Model

Replace the existing MoE layer with the All2All EP version:

```python
class MyMoELayer(torch.nn.Module):
    def __init__(self, config, moe_config, ep_group, ep_rank, ep_size):
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        
        # Router
        self.gate = torch.nn.Linear(
            config.hidden_size,
            moe_config.num_experts,
            bias=False,
        )
        
        # Expert weights (only local experts on this rank)
        self.gate_up_weights = torch.nn.Parameter(
            torch.empty(
                moe_config.num_local_experts,
                config.intermediate_size * 2,
                config.hidden_size // ep_size,  # Partitioned hidden dim
            )
        )
        self.down_weights = torch.nn.Parameter(
            torch.empty(
                moe_config.num_local_experts,
                config.hidden_size // ep_size,  # Partitioned hidden dim
                config.intermediate_size,
            )
        )
        
    def forward(self, hidden_states):
        # Compute router logits
        router_logits = self.gate(hidden_states)
        
        # Call All2All EP MoE
        from dlinfer.vendor.ascend.torch_npu_ops import fused_moe_all2all
        
        output = fused_moe_all2all(
            hidden_states=hidden_states,
            router_logits=router_logits,
            moe_config=self.moe_config,
            ep_group=self.ep_group,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            expert_kernel=my_expert_kernel,
            expert_kwargs={
                'gate_up_weights': self.gate_up_weights,
                'down_weights': self.down_weights,
            },
            normalize_router_logits=True,
            bias=None,
        )
        
        return output
```

## Step 5: Handle Hidden Dimension Partitioning

Since the All2All implementation partitions the hidden dimension, you need to handle this in your model:

```python
class MyTransformerBlock(torch.nn.Module):
    def __init__(self, config, moe_config, ep_group, ep_rank, ep_size):
        super().__init__()
        self.ep_size = ep_size
        
        # Attention layer
        self.attention = MyAttentionLayer(config)
        
        # MoE layer (outputs partitioned hidden)
        self.moe = MyMoELayer(config, moe_config, ep_group, ep_rank, ep_size)
        
        # All-gather to restore full hidden dimension if needed
        self.all_gather = ep_size > 1
        
    def forward(self, hidden_states):
        # Attention (full hidden dimension)
        attn_output = self.attention(hidden_states)
        
        # MoE (returns partitioned hidden dimension)
        moe_output = self.moe(attn_output)
        
        # Optional: All-gather to restore full hidden dimension
        if self.all_gather:
            # Gather along hidden dimension
            gathered_list = [torch.empty_like(moe_output) for _ in range(self.ep_size)]
            dist.all_gather(gathered_list, moe_output, group=self.ep_group)
            moe_output = torch.cat(gathered_list, dim=-1)
        
        # Residual connection
        output = attn_output + moe_output
        
        return output
```

## Step 6: Testing Your Integration

Validate the integration with a simple test:

```python
def test_moe_integration():
    # Initialize distributed
    dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Setup
    config = MyModelConfig(hidden_size=1024, intermediate_size=4096)
    moe_config = MoEConfig(num_experts=8, experts_per_token=2, ep_world_size=world_size)
    ep_group = dist.new_group(ranks=list(range(world_size)))
    
    # Create layer
    layer = MyMoELayer(config, moe_config, ep_group, rank, world_size)
    layer = layer.to(f'npu:{rank}')
    
    # Test forward pass
    batch_size, seq_len = 4, 128
    hidden_states = torch.randn(
        batch_size * seq_len,
        config.hidden_size,
        device=f'npu:{rank}'
    )
    
    output = layer(hidden_states)
    
    # Verify output shape
    assert output.shape == (
        batch_size * seq_len,
        config.hidden_size // world_size
    ), f"Expected shape {(batch_size * seq_len, config.hidden_size // world_size)}, got {output.shape}"
    
    print(f"[Rank {rank}] Test passed!")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    test_moe_integration()
```

## Troubleshooting

### Issue: "Rows must be divisible by ep_size"

**Cause**: The number of dispatched tokens after top-k routing is not divisible by ep_size.

**Solution**: This is expected to happen occasionally. The implementation handles this by padding. Ensure your ep_size is reasonable (typically 2, 4, or 8).

### Issue: "hidden size must be divisible by ep_size"

**Cause**: The hidden dimension is not evenly divisible by the number of EP ranks.

**Solution**: Adjust your model's hidden_size or ep_size. For example, with hidden_size=1024, use ep_size ∈ {1, 2, 4, 8, 16, 32, ...}.

### Issue: "ModuleNotFoundError: No module named 'torch_npu'"

**Cause**: torch_npu is not installed or not in Python path.

**Solution**: 
1. Install torch_npu: `pip install torch_npu`
2. Or use CPU/GPU fallback: The implementation will automatically use PyTorch operations when torch_npu is unavailable.

### Issue: Poor performance compared to single-device MoE

**Possible causes**:
1. **Communication overhead**: All2All can be expensive. Profile to check if communication dominates.
2. **Load imbalance**: If tokens are not evenly distributed across experts, some ranks may idle.
3. **Small batch sizes**: All2All overhead is amortized over larger batches.

**Solutions**:
- Increase batch size
- Use load balancing loss in training
- Consider hybrid expert placement strategies

## Performance Tips

1. **Batch size**: Use larger batches to amortize communication overhead.

2. **Hidden dimension partitioning**: Larger hidden dimensions benefit more from partitioning.

3. **Expert count**: More experts per rank reduces communication frequency.

4. **NPU optimization**: Ensure torch_npu is available for hardware-accelerated token permute/unpermute.

5. **Profiling**: Use NPU profiler to identify bottlenecks:
   ```python
   with torch.npu.profile():
       output = layer(hidden_states)
   ```

## Advanced: Combining with Tensor Parallelism

If your model uses tensor parallelism (TP), you can combine it with EP:

```python
# Setup: TP=2, EP=4, Total ranks=8
# Ranks 0,2,4,6 form EP group 0 (TP rank 0)
# Ranks 1,3,5,7 form EP group 1 (TP rank 1)

rank = dist.get_rank()
world_size = dist.get_world_size()
tp_size = 2
ep_size = world_size // tp_size

tp_rank = rank % tp_size
ep_rank = rank // tp_size

# Create TP group (ranks with same ep_rank)
tp_ranks = [ep_rank * tp_size + i for i in range(tp_size)]
tp_group = dist.new_group(ranks=tp_ranks)

# Create EP group (ranks with same tp_rank)
ep_ranks = [tp_rank + i * tp_size for i in range(ep_size)]
ep_group = dist.new_group(ranks=ep_ranks)

# Now hidden_size is partitioned by BOTH tp_size and ep_size
effective_hidden_size = config.hidden_size // (tp_size * ep_size)
```

## Summary

The All2All EP MoE implementation provides:
✅ Efficient expert-parallel distribution across devices
✅ Reduced memory per device (experts + hidden dim partitioned)
✅ NPU-optimized operations for Ascend
✅ Compatible with existing dlinfer expert kernels
✅ Flexible integration with tensor parallelism

For questions or issues, refer to the main README.md or open an issue in the repository.
