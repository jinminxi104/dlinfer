# Copyright (c) 2024, DeepLink. All rights reserved.
"""Tests for TokenDispatcherWithAll2AllV."""
import pytest
import torch
import torch.distributed as dist
from unittest.mock import MagicMock, patch


# Mock moe_config class for testing
class MockMoEConfig:
    def __init__(self, num_experts=8, num_local_experts=2, experts_per_token=2):
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token


class TestTokenDispatcherAll2All:
    """Test suite for TokenDispatcherWithAll2AllV."""

    def test_dispatcher_initialization(self):
        """Test that dispatcher initializes correctly."""
        from dlinfer.ops.moe.token_dispatcher_all2all import TokenDispatcherWithAll2AllV
        
        # Create a mock EP group
        mock_ep_group = MagicMock(spec=dist.ProcessGroup)
        
        dispatcher = TokenDispatcherWithAll2AllV(
            ep_group=mock_ep_group,
            ep_rank=0,
            ep_size=4,
            top_k=2,
            num_experts=8,
            num_local_experts=2,
        )
        
        assert dispatcher.ep_rank == 0
        assert dispatcher.ep_size == 4
        assert dispatcher.top_k == 2
        assert dispatcher.num_experts == 8
        assert dispatcher.num_local_experts == 2
        assert len(dispatcher.local_expert_indices) == 2
        assert dispatcher.local_expert_indices == [0, 1]

    def test_dispatcher_local_expert_mapping(self):
        """Test that local expert indices are correctly computed."""
        from dlinfer.ops.moe.token_dispatcher_all2all import TokenDispatcherWithAll2AllV
        
        mock_ep_group = MagicMock(spec=dist.ProcessGroup)
        
        # Rank 1 should have experts [2, 3]
        dispatcher = TokenDispatcherWithAll2AllV(
            ep_group=mock_ep_group,
            ep_rank=1,
            ep_size=4,
            top_k=2,
            num_experts=8,
            num_local_experts=2,
        )
        
        assert dispatcher.local_expert_indices == [2, 3]
        
        # Rank 3 should have experts [6, 7]
        dispatcher = TokenDispatcherWithAll2AllV(
            ep_group=mock_ep_group,
            ep_rank=3,
            ep_size=4,
            top_k=2,
            num_experts=8,
            num_local_experts=2,
        )
        
        assert dispatcher.local_expert_indices == [6, 7]

    def test_token_dispatch_shapes(self):
        """Test that token_dispatch produces correct output shapes."""
        from dlinfer.ops.moe.token_dispatcher_all2all import TokenDispatcherWithAll2AllV
        
        n_tokens = 16
        h_total = 128
        top_k = 2
        ep_size = 4
        ep_rank = 0
        num_experts = 8
        num_local_experts = 2
        
        # Create test inputs
        hidden_states = torch.randn(n_tokens, h_total)
        topk_weights = torch.rand(n_tokens, top_k)
        topk_ids = torch.randint(0, num_experts, (n_tokens, top_k))
        
        # Mock all_to_all to return same shape tensor
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.copy_(input)
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            dispatcher = TokenDispatcherWithAll2AllV(
                ep_group=mock_ep_group,
                ep_rank=ep_rank,
                ep_size=ep_size,
                top_k=top_k,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
            )
            
            result = dispatcher.token_dispatch(hidden_states, topk_weights, topk_ids)
            
            # Check output structure
            assert "hidden_states" in result
            assert "group_list" in result
            assert "context" in result
            
            # Check shapes - hidden dim should be partitioned
            h_part = h_total // ep_size
            expected_rows = n_tokens * top_k
            assert result["hidden_states"].shape == (expected_rows, h_part)
            
            # Check group_list shape
            assert result["group_list"].shape == (num_local_experts,)

    def test_token_combine_restores_shape(self):
        """Test that token_combine restores original shape and reduces over top_k."""
        from dlinfer.ops.moe.token_dispatcher_all2all import TokenDispatcherWithAll2AllV
        
        n_tokens = 16
        h_total = 128
        top_k = 2
        ep_size = 4
        ep_rank = 0
        num_experts = 8
        num_local_experts = 2
        
        hidden_states = torch.randn(n_tokens, h_total)
        topk_weights = torch.rand(n_tokens, top_k)
        topk_ids = torch.randint(0, num_experts, (n_tokens, top_k))
        
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.copy_(input)
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            dispatcher = TokenDispatcherWithAll2AllV(
                ep_group=mock_ep_group,
                ep_rank=ep_rank,
                ep_size=ep_size,
                top_k=top_k,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
            )
            
            # Dispatch
            dispatch_result = dispatcher.token_dispatch(hidden_states, topk_weights, topk_ids)
            
            # Create mock expert output (same shape as dispatched hidden states)
            h_part = h_total // ep_size
            expert_output = torch.randn_like(dispatch_result["hidden_states"])
            
            # Combine
            combined = dispatcher.token_combine(expert_output, dispatch_result["context"])
            
            # Check that output shape matches original hidden_states
            assert combined.shape == hidden_states.shape

    def test_token_dispatch_with_bias(self):
        """Test that bias is correctly added in token_combine."""
        from dlinfer.ops.moe.token_dispatcher_all2all import TokenDispatcherWithAll2AllV
        
        n_tokens = 8
        h_total = 64
        top_k = 2
        ep_size = 2
        
        hidden_states = torch.randn(n_tokens, h_total)
        topk_weights = torch.rand(n_tokens, top_k)
        topk_ids = torch.randint(0, 4, (n_tokens, top_k))
        bias = torch.ones(1, h_total) * 0.5
        
        with patch('torch.distributed.all_to_all_single') as mock_a2a:
            def all_to_all_side_effect(output, input, **kwargs):
                output.copy_(input)
                return None
            mock_a2a.side_effect = all_to_all_side_effect
            
            mock_ep_group = MagicMock(spec=dist.ProcessGroup)
            dispatcher = TokenDispatcherWithAll2AllV(
                ep_group=mock_ep_group,
                ep_rank=0,
                ep_size=ep_size,
                top_k=top_k,
                num_experts=4,
                num_local_experts=2,
            )
            
            dispatch_result = dispatcher.token_dispatch(hidden_states, topk_weights, topk_ids)
            expert_output = torch.zeros_like(dispatch_result["hidden_states"])
            
            # Combine with bias
            combined = dispatcher.token_combine(expert_output, dispatch_result["context"], bias=bias)
            
            # Check shape
            assert combined.shape == hidden_states.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
