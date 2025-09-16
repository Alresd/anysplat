"""
PyTorch native implementation of scatter operations to replace torch_scatter
"""

import torch


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    """
    PyTorch native implementation of scatter_add operation

    Args:
        src: source tensor to scatter
        index: indices for scattering
        dim: dimension along which to scatter
        out: output tensor (optional)
        dim_size: size of output dimension (optional)
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    if out is None:
        size = list(src.shape)
        size[dim] = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    return out.scatter_add_(dim, index, src)


def scatter_max(src, index, dim=0, out=None, dim_size=None):
    """
    PyTorch native implementation of scatter_max operation

    Args:
        src: source tensor to scatter
        index: indices for scattering
        dim: dimension along which to scatter
        out: output tensor (optional)
        dim_size: size of output dimension (optional)

    Returns:
        tuple of (values, indices) like torch_scatter.scatter_max
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    if out is None:
        size = list(src.shape)
        size[dim] = dim_size
        out = torch.full(size, float('-inf'), dtype=src.dtype, device=src.device)
        arg_out = torch.zeros(size, dtype=torch.long, device=src.device)
    else:
        out, arg_out = out

    # Manual implementation for compatibility
    for i in range(dim_size):
        mask = (index == i)
        if mask.any():
            masked_src = src[mask]
            if masked_src.numel() > 0:
                max_val, max_idx = torch.max(masked_src, dim=0)
                if dim == 0:
                    out[i] = max_val
                    arg_out[i] = max_idx
                else:
                    # For other dimensions, need more complex indexing
                    out.index_fill_(dim, torch.tensor([i], device=out.device), max_val)

    return out, arg_out


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    """
    PyTorch native implementation of scatter_mean operation
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    # Count occurrences for each index
    ones = torch.ones_like(src)
    count = scatter_add(ones, index, dim, dim_size=dim_size)

    # Sum values
    sum_values = scatter_add(src, index, dim, dim_size=dim_size)

    # Avoid division by zero
    count = torch.clamp(count, min=1)

    return sum_values / count