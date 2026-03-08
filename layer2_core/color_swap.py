"""
Layer 2 — Color Swap Functions

Implements the frequency-domain color transfer that is the heart of
tile_colorfix. The core formula swaps low-frequency (color/tone)
information while preserving high-frequency (detail/structure).

A1111 source (hook.py:823):
    x0 = x0_prd - blur(x0_prd, k) + blur(x0_origin, k)

A1111 +sharp source (hook.py:825-828):
    detail_weight = float(param.preprocessor['threshold_b']) * 0.01
    neg = detail_weight * blur(x0, k) + (1 - detail_weight) * x0
    x0 = cond_mark * x0 + (1 - cond_mark) * neg
"""

import torch
from typing import Callable
from .blur import box_blur


def color_swap(
    x0_prd: torch.Tensor,
    x0_origin: torch.Tensor,
    k: int,
    blur_fn: Callable = box_blur
) -> torch.Tensor:
    """
    Base color fix: swap low-frequency colors from reference into prediction.

    Formula:
        x0_fixed = x0_prd - blur(x0_prd, k) + blur(x0_origin, k)

    Works with any number of latent channels (4 for SD/SDXL, 16 for Flux).

    Args:
        x0_prd:    Predicted clean image in latent space. Shape (B, C, H, W).
        x0_origin: VAE-encoded control/reference image. Shape (B, C, H, W).
        k:         Blur kernel radius (default 8 in A1111).
        blur_fn:   Blur function to use (box_blur or gaussian_blur).

    Returns:
        Color-fixed latent with prediction's details and reference's colors.
    """
    return x0_prd - blur_fn(x0_prd, k) + blur_fn(x0_origin, k)


def color_swap_sharp(
    x0_prd: torch.Tensor,
    x0_origin: torch.Tensor,
    k: int,
    sharpness: float,
    blur_fn: Callable = box_blur
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sharp variant of color fix (CFG-based, for SD/SDXL).

    Returns two versions for differential CFG processing:
    - cond_result:   Full color-fixed image (keeps all detail)
    - uncond_result: Softened version (slightly blurred)

    CFG amplifies the difference between them → detail enhancement.

    Args:
        x0_prd:    Predicted clean image in latent space. Shape (B, C, H, W).
        x0_origin: VAE-encoded control/reference image. Shape (B, C, H, W).
        k:         Blur kernel radius.
        sharpness: Sharpness parameter (0.0-2.0). Maps to detail_weight via x0.01.
        blur_fn:   Blur function to use (box_blur or gaussian_blur).

    Returns:
        Tuple of (cond_result, uncond_result):
        - cond_result: Full color-fixed x0 (sharp, for conditional path)
        - uncond_result: Softened x0 (for unconditional path -> CFG amplifies detail)
    """
    # Step 1: Apply base color swap
    x0 = color_swap(x0_prd, x0_origin, k, blur_fn)

    # Step 2: Create softened version for uncond path
    detail_weight = sharpness * 0.01
    neg = detail_weight * blur_fn(x0, k) + (1.0 - detail_weight) * x0

    # cond path gets full x0, uncond path gets softened neg
    return x0, neg


def color_swap_sharp_direct(
    x0_prd: torch.Tensor,
    x0_origin: torch.Tensor,
    k: int,
    sharpness: float,
    blur_fn: Callable = box_blur
) -> torch.Tensor:
    """
    Sharp variant of color fix (direct sharpening, for Flux).

    Flux models run without traditional CFG (cfg_scale=1.0), so the
    CFG-differential sharpening approach doesn't work. Instead, this
    applies direct high-pass detail enhancement (unsharp mask style).

    Formula:
        fixed     = x0_prd - blur(x0_prd, k) + blur(x0_origin, k)
        high_freq = fixed - blur(fixed, k)
        result    = fixed + sharpness * 0.1 * high_freq

    The 0.1 scaling factor maps the sharpness range (0-2) to produce
    comparable visual effect to the CFG-based variant at typical settings.

    Args:
        x0_prd:    Predicted clean image in latent space. Shape (B, C, H, W).
        x0_origin: VAE-encoded control/reference image. Shape (B, C, H, W).
        k:         Blur kernel radius.
        sharpness: Sharpness parameter (0.0-2.0). Higher = more detail boost.
        blur_fn:   Blur function to use (box_blur or gaussian_blur).

    Returns:
        Color-fixed and sharpened latent tensor.
    """
    # Step 1: Apply base color swap
    fixed = color_swap(x0_prd, x0_origin, k, blur_fn)

    # Step 2: Extract high-frequency detail from the color-fixed result
    high_freq = fixed - blur_fn(fixed, k)

    # Step 3: Enhance detail by adding amplified high frequencies back
    # 0.1 scaling makes sharpness=1.0 give ~10% detail boost,
    # comparable to original CFG-based effect at typical cfg_scale values
    sharpened = fixed + sharpness * 0.1 * high_freq

    return sharpened
