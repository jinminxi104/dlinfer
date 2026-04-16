# Copyright (c) 2024, DeepLink. All rights reserved.
import torch
import torch_npu
def weak_ref_tensor(tensor):
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    This ignores 0-size tensors as those don't allocate any memory.
    """
    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        return torch_npu._C._weak_ref_tensor(tensor)
    else:
        return tensor


def weak_ref_tensors(
    tensors: torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
):
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)

    raise ValueError("Invalid type for tensors")


