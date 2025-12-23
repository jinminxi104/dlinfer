# Copyright (c) 2024, DeepLink. All rights reserved.
"""End-to-end tests for All2All MoE flow."""
import pytest
import torch
import torch.distributed as dist
from unittest.mock import MagicMock, patch


class MockMoEConfig:
    """Mock MoE configuration for testing."""
    def __init__(self, num_experts=8, num_local_experts=2, experts_per_token=2):
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token


class TestCommonFusedMoEAll2All:
    """Test suite for forward_impl_all2all end-to-end flow."""

    def test_forward_impl_all2all_basic(self):
        """Test basic end-to-end flow with identity expert kernel."""
        from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
        
        n_tokens = 16
        h_total = 128
        num_experts = 8
        num_local_experts = 2
        top_k = 2
        ep_size = 4
        ep_rank = 0
        
        # Create inputs
        hidden_states = torch.randn(n_tokens, h_total)
        router_logits = torch.randn(n_tokens, num_experts)
        
        # Create mock config
        moe_config = MockMoEConfig(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            experts_per_token=top_k,
        )
        
        # Identity expert kernel for testing
        def identity_expert_kernel(x):
            return x
        
        # Mock distributed ops
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.copy_(input)
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            
            # Run forward
            output = forward_impl_all2all(
                hidden_states,
                router_logits,
                moe_config=moe_config,
                ep_group=mock_ep_group,
                ep_rank=ep_rank,
                ep_size=ep_size,
                expert_kernel=identity_expert_kernel,
            )
            
            # Check output shape
            assert output.shape == hidden_states.shape

    def test_forward_impl_all2all_with_linear_experts(self):
        """Test end-to-end flow with linear expert kernel."""
        from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
        
        n_tokens = 8
        h_total = 64
        num_experts = 4
        num_local_experts = 2
        top_k = 2
        ep_size = 2
        ep_rank = 0
        
        hidden_states = torch.randn(n_tokens, h_total)
        router_logits = torch.randn(n_tokens, num_experts)
        
        moe_config = MockMoEConfig(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            experts_per_token=top_k,
        )
        
        # Linear expert kernel
        h_part = h_total // ep_size
        weight = torch.randn(h_part, h_part)
        
        def linear_expert_kernel(x):
            return torch.matmul(x, weight.T)
        
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.copy_(input)
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            
            output = forward_impl_all2all(
                hidden_states,
                router_logits,
                moe_config=moe_config,
                ep_group=mock_ep_group,
                ep_rank=ep_rank,
                ep_size=ep_size,
                expert_kernel=linear_expert_kernel,
            )
            
            assert output.shape == hidden_states.shape
            # Check that output is not same as input (transformation applied)
            assert not torch.allclose(output, hidden_states)

    def test_forward_impl_all2all_with_bias(self):
        """Test end-to-end flow with bias addition."""
        from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
        
        n_tokens = 8
        h_total = 64
        num_experts = 4
        
        hidden_states = torch.randn(n_tokens, h_total)
        router_logits = torch.randn(n_tokens, num_experts)
        bias = torch.ones(1, h_total) * 0.1
        
        moe_config = MockMoEConfig(num_experts=num_experts, num_local_experts=2, experts_per_token=2)
        
        def identity_expert_kernel(x):
            return torch.zeros_like(x)
        
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.zero_()
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            
            output = forward_impl_all2all(
                hidden_states,
                router_logits,
                moe_config=moe_config,
                ep_group=mock_ep_group,
                ep_rank=0,
                ep_size=2,
                expert_kernel=identity_expert_kernel,
                bias=bias,
            )
            
            assert output.shape == hidden_states.shape

    def test_forward_impl_all2all_normalize_router_logits(self):
        """Test that normalize_router_logits flag works."""
        from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
        
        n_tokens = 4
        h_total = 32
        num_experts = 4
        
        hidden_states = torch.randn(n_tokens, h_total)
        router_logits = torch.randn(n_tokens, num_experts)
        
        moe_config = MockMoEConfig(num_experts=num_experts, num_local_experts=2, experts_per_token=2)
        
        def identity_expert_kernel(x):
            return x
        
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.copy_(input)
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            
            # Should not raise errors
            output = forward_impl_all2all(
                hidden_states,
                router_logits,
                moe_config=moe_config,
                ep_group=mock_ep_group,
                ep_rank=0,
                ep_size=2,
                expert_kernel=identity_expert_kernel,
                normalize_router_logits=True,
            )
            
            assert output.shape == hidden_states.shape

    def test_forward_impl_all2all_expert_kwargs(self):
        """Test that expert_kwargs are passed correctly."""
        from dlinfer.ops.moe.common_fused_moe import forward_impl_all2all
        
        n_tokens = 4
        h_total = 32
        num_experts = 4
        
        hidden_states = torch.randn(n_tokens, h_total)
        router_logits = torch.randn(n_tokens, num_experts)
        
        moe_config = MockMoEConfig(num_experts=num_experts, num_local_experts=2, experts_per_token=2)
        
        # Track if kwargs were passed
        kwargs_received = {}
        
        def expert_kernel_with_kwargs(x, scale=1.0):
            kwargs_received['scale'] = scale
            return x * scale
        
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.copy_(input)
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            
            output = forward_impl_all2all(
                hidden_states,
                router_logits,
                moe_config=moe_config,
                ep_group=mock_ep_group,
                ep_rank=0,
                ep_size=2,
                expert_kernel=expert_kernel_with_kwargs,
                expert_kwargs={'scale': 2.0},
            )
            
            assert output.shape == hidden_states.shape
            assert kwargs_received['scale'] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
