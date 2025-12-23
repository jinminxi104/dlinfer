# Copyright (c) 2024, DeepLink. All rights reserved.
"""
Unit tests for TokenDispatcherWithAll2AllV

These tests verify the token dispatcher's shape handling and token combine
functionality without requiring actual distributed execution.
"""

import torch
import torch.distributed as dist


class MockProcessGroup:
    """Mock process group for testing without actual distributed setup."""
    pass


def test_token_dispatcher_shapes():
    """Test that token_dispatch and token_combine maintain correct shapes."""
    from dlinfer.ops.moe.token_dispatcher_all2all import TokenDispatcherWithAll2AllV
    
    # Setup test parameters
    n_tokens = 16
    h_total = 128
    top_k = 2
    num_experts = 8
    num_local_experts = 2
    ep_size = 4
    ep_rank = 0
    
    # Create mock EP group
    ep_group = MockProcessGroup()
    
    # Create dispatcher
    dispatcher = TokenDispatcherWithAll2AllV(
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
        top_k=top_k,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        apply_router_weight_on_input=False,
    )
    
    # Create test inputs
    hidden_states = torch.randn(n_tokens, h_total)
    topk_weights = torch.softmax(torch.randn(n_tokens, top_k), dim=-1)
    topk_ids = torch.randint(0, num_experts, (n_tokens, top_k))
    
    # Mock the all_to_all to just partition and return
    def mock_all_to_all(self, tensor):
        # In real all_to_all, this would exchange data across ranks
        # For testing, we just return the same tensor
        return None, tensor, None
    
    original_all_to_all = dispatcher._async_all_to_all
    dispatcher._async_all_to_all = lambda t: mock_all_to_all(dispatcher, t)
    
    try:
        # Test token_dispatch
        result = dispatcher.token_dispatch(hidden_states, topk_weights, topk_ids)
        
        # Check output structure
        assert "hidden_states" in result
        assert "group_list" in result
        assert "context" in result
        
        # Check shapes
        h_part = h_total // ep_size
        expected_rows = n_tokens * top_k
        assert result["hidden_states"].shape[1] == h_part, \
            f"Expected hidden dim {h_part}, got {result['hidden_states'].shape[1]}"
        
        # Test token_combine
        context = result["context"]
        mlp_out = torch.randn_like(result["hidden_states"])
        
        # Mock the reverse all_to_all
        combined = dispatcher.token_combine(mlp_out, context, bias=None)
        
        # Check that combine restores original shape and reduces over top_k
        assert combined.shape == (n_tokens, h_part), \
            f"Expected shape ({n_tokens}, {h_part}), got {combined.shape}"
        
        print("✓ Token dispatcher shape tests passed")
        
    finally:
        # Restore original method
        dispatcher._async_all_to_all = original_all_to_all


def test_token_dispatcher_with_bias():
    """Test token_combine with bias addition."""
    from dlinfer.ops.moe.token_dispatcher_all2all import TokenDispatcherWithAll2AllV
    
    n_tokens = 8
    h_total = 64
    top_k = 2
    ep_size = 2
    
    dispatcher = TokenDispatcherWithAll2AllV(
        ep_group=MockProcessGroup(),
        ep_rank=0,
        ep_size=ep_size,
        top_k=top_k,
        num_experts=4,
        num_local_experts=2,
    )
    
    # Set original shape manually for testing
    h_part = h_total // ep_size
    dispatcher._original_shape = (n_tokens, h_total)
    
    # Create test data
    mlp_out = torch.randn(n_tokens * top_k, h_part)
    bias = torch.randn(h_part)
    
    # Create minimal context
    expanded_row_idx = torch.arange(n_tokens * top_k)
    context = {
        "expanded_row_idx": expanded_row_idx,
        "h_part": h_part,
    }
    
    # Mock all_to_all
    def mock_all_to_all_single(*args, **kwargs):
        # Just copy input to output for testing
        output = args[0]
        input_tensor = args[1]
        output.copy_(input_tensor)
    
    original_fn = dist.all_to_all_single
    dist.all_to_all_single = mock_all_to_all_single
    
    try:
        # Test without bias
        result_no_bias = dispatcher.token_combine(mlp_out, context, bias=None)
        
        # Test with bias
        result_with_bias = dispatcher.token_combine(mlp_out, context, bias=bias)
        
        # Check that bias was added
        diff = result_with_bias - result_no_bias
        assert torch.allclose(diff, bias.unsqueeze(0).expand(n_tokens, -1), atol=1e-5), \
            "Bias was not added correctly"
        
        print("✓ Token dispatcher bias tests passed")
        
    finally:
        dist.all_to_all_single = original_fn


if __name__ == "__main__":
    test_token_dispatcher_shapes()
    test_token_dispatcher_with_bias()
    print("\n✓ All token dispatcher tests passed!")
