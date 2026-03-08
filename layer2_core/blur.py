"""
Layer 2 — Blur Functions

Two blur modes for extracting low-frequency (color/tone) information:

    box_blur:      Exact replica of A1111's blur() from hook.py:331-334.
                   Fast, but sharp frequency cutoff can cause patchy color
                   artifacts when used without tile ControlNet.

    gaussian_blur: Smooth Gaussian filter with sigma = k/2.
                   Smoother frequency rolloff produces cleaner color
                   transitions, especially for standalone use (no tile CN).
"""

import torch
import torch.nn.functional as F


def box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Box blur (mean filter) operating on 4D latent tensors.

    Applies a (2k+1) x (2k+1) average filter with replicate padding.
    Exact copy of A1111's blur() function.

    Args:
        x: Input tensor of shape (B, C, H, W).
        k: Blur kernel radius. Kernel size = (2*k+1, 2*k+1).

    Returns:
        Blurred tensor of same shape as input.
    """
    y = F.pad(x, (k, k, k, k), mode='replicate')
    y = F.avg_pool2d(y, (k * 2 + 1, k * 2 + 1), stride=(1, 1))
    return y


def gaussian_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Gaussian blur operating on 4D latent tensors.

    Applies a (2k+1) x (2k+1) Gaussian filter with sigma = k/2.
    Smoother frequency rolloff than box blur — reduces patchy color
    artifacts when colorfix is used without tile ControlNet.

    Args:
        x: Input tensor of shape (B, C, H, W).
        k: Blur kernel radius. Kernel size = (2*k+1, 2*k+1).
           sigma is automatically set to k/2.

    Returns:
        Blurred tensor of same shape as input.
    """
    kernel_size = k * 2 + 1
    sigma = k / 2.0

    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - k
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    # Outer product → 2D kernel, then expand for depthwise conv
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    channels = x.shape[1]
    kernel = gauss_2d.expand(channels, 1, kernel_size, kernel_size)

    # Replicate pad (same as box_blur) then depthwise convolution
    y = F.pad(x, (k, k, k, k), mode='replicate')
    y = F.conv2d(y, kernel, groups=channels)
    return y


# Map dropdown strings to blur functions
BLUR_MODE_MAP = {
    "box (A1111 original)": box_blur,
    "gaussian (smoother)": gaussian_blur,
}
