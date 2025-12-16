# Copyright (c) 2024, DeepLink. All rights reserved.
"""
Unit tests for end-to-end All2All MoE flow

These tests verify the complete forward_impl_all2all flow with mocked components.
"""

import torch
import torch.distributed as dist


class MockProcessGroup:
    """Mock process group for testing."""
    pass


class MockMoeConfig:
    """Mock MoE configuration."""
    def __init__(self, num_experts=8, num_local_experts=2, experts_per_token=2):
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token


def test_forward_impl_all2all_basic():
    """Test basic end-to-end flow of forward_impl_all2all."""
    from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
    
    # Setup
    n_tokens = 16
    h_total = 128
    num_experts = 8
    num_local_experts = 2
    experts_per_token = 2
    ep_size = 4
    ep_rank = 0
    
    moe_config = MockMoeConfig(num_experts, num_local_experts, experts_per_token)
    ep_group = MockProcessGroup()
    
    # Create test inputs
    hidden_states = torch.randn(n_tokens, h_total)
    router_logits = torch.randn(n_tokens, num_experts)
    
    # Define a simple expert kernel that just returns the input
    def dummy_expert_kernel(x, **kwargs):
        """Dummy expert that preserves shape."""
        return x
    
    # Mock all_to_all_single to avoid requiring distributed setup
    def mock_all_to_all_single(output, input_tensor, **kwargs):
        output.copy_(input_tensor)
    
    original_fn = dist.all_to_all_single
    dist.all_to_all_single = mock_all_to_all_single
    
    try:
        # Run forward
        output = forward_impl_all2all(
            hidden_states,
            router_logits,
            moe_config=moe_config,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            expert_kernel=dummy_expert_kernel,
            expert_kwargs=None,
            apply_router_weight_on_input=False,
            normalize_router_logits=True,
            quant_type=None,
            bias=None,
        )
        
        # Check output shape
        h_part = h_total // ep_size
        assert output.shape == (n_tokens, h_part), \
            f"Expected output shape ({n_tokens}, {h_part}), got {output.shape}"
        
        print("✓ End-to-end forward_impl_all2all basic test passed")
        
    finally:
        dist.all_to_all_single = original_fn


def test_forward_impl_all2all_with_bias():
    """Test forward_impl_all2all with bias."""
    from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
    
    # Setup
    n_tokens = 8
    h_total = 64
    ep_size = 2
    h_part = h_total // ep_size
    
    moe_config = MockMoeConfig(num_experts=4, num_local_experts=2, experts_per_token=2)
    ep_group = MockProcessGroup()
    
    hidden_states = torch.randn(n_tokens, h_total)
    router_logits = torch.randn(n_tokens, moe_config.num_experts)
    bias = torch.randn(h_part)
    
    def dummy_expert_kernel(x, **kwargs):
        return x
    
    # Mock all_to_all
    original_fn = dist.all_to_all_single
    dist.all_to_all_single = lambda out, inp, **kw: out.copy_(inp)
    
    try:
        # Test without bias
        output_no_bias = forward_impl_all2all(
            hidden_states,
            router_logits,
            moe_config=moe_config,
            ep_group=ep_group,
            ep_rank=0,
            ep_size=ep_size,
            expert_kernel=dummy_expert_kernel,
            bias=None,
        )
        
        # Test with bias
        output_with_bias = forward_impl_all2all(
            hidden_states,
            router_logits,
            moe_config=moe_config,
            ep_group=ep_group,
            ep_rank=0,
            ep_size=ep_size,
            expert_kernel=dummy_expert_kernel,
            bias=bias,
        )
        
        # Verify bias was added
        diff = output_with_bias - output_no_bias
        assert torch.allclose(diff, bias.unsqueeze(0).expand(n_tokens, -1), atol=1e-5), \
            "Bias was not added correctly in end-to-end flow"
        
        print("✓ End-to-end forward_impl_all2all bias test passed")
        
    finally:
        dist.all_to_all_single = original_fn


def test_forward_impl_all2all_with_expert_kwargs():
    """Test that expert_kwargs are properly passed to expert kernel."""
    from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
    
    n_tokens = 8
    h_total = 64
    ep_size = 2
    
    moe_config = MockMoeConfig(num_experts=4, num_local_experts=2, experts_per_token=2)
    
    # Track if expert kernel received kwargs
    received_kwargs = {}
    
    def expert_kernel_with_kwargs(x, scale=1.0, offset=0.0):
        received_kwargs['scale'] = scale
        received_kwargs['offset'] = offset
        return x * scale + offset
    
    hidden_states = torch.randn(n_tokens, h_total)
    router_logits = torch.randn(n_tokens, moe_config.num_experts)
    
    expert_kwargs = {'scale': 2.0, 'offset': 0.5}
    
    # Mock all_to_all
    original_fn = dist.all_to_all_single
    dist.all_to_all_single = lambda out, inp, **kw: out.copy_(inp)
    
    try:
        output = forward_impl_all2all(
            hidden_states,
            router_logits,
            moe_config=moe_config,
            ep_group=MockProcessGroup(),
            ep_rank=0,
            ep_size=ep_size,
            expert_kernel=expert_kernel_with_kwargs,
            expert_kwargs=expert_kwargs,
        )
        
        # Verify kwargs were passed
        assert received_kwargs.get('scale') == 2.0, "scale kwarg not passed correctly"
        assert received_kwargs.get('offset') == 0.5, "offset kwarg not passed correctly"
        
        print("✓ End-to-end expert_kwargs test passed")
        
    finally:
        dist.all_to_all_single = original_fn


if __name__ == "__main__":
    test_forward_impl_all2all_basic()
    test_forward_impl_all2all_with_bias()
    test_forward_impl_all2all_with_expert_kwargs()
    print("\n✓ All end-to-end MoE tests passed!")
