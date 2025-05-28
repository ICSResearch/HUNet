from math import prod
from typing import Tuple

import numpy as np
import torch
from timm.models.layers import to_2tuple


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)

def blc_to_bhwc(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, L, C = x.shape
    return x.view(B, *x_size, C)

def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows

def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

