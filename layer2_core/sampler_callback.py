"""
Layer 2 — Sampler Callback Factory

Creates post_cfg_function callbacks registered on models via
model.set_model_sampler_post_cfg_function(). Callbacks run at every
denoising step, AFTER CFG combination, and apply color correction.

Two callback variants:
    make_colorfix_post_cfg_callback  — For SD/SDXL models (supports CFG-based sharpening)
    make_flux_colorfix_post_cfg_callback — For Flux models (direct sharpening, no CFG dependency)

API contract (comfy/samplers.py):
    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond_denoised": cond_pred,
                "uncond_denoised": uncond_pred, "cond_scale": cond_scale,
                "input": x, "sigma": timestep, ...}
        cfg_result = fn(args)
"""

import torch
import torch.nn.functional as F
from typing import Callable
from .blur import box_blur
from .color_swap import color_swap, color_swap_sharp, color_swap_sharp_direct


def match_size(x0_origin: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Match x0_origin spatial dimensions and batch size to the target tensor.

    Args:
        x0_origin: Reference latent. Shape (B1, C, H1, W1).
        target:    Target tensor to match. Shape (B2, C, H2, W2).

    Returns:
        x0_origin resized/repeated to match target's spatial dims and batch size.
    """
    ref = x0_origin

    # Match spatial dimensions (H, W)
    if ref.shape[2:] != target.shape[2:]:
        ref = F.interpolate(
            ref,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=False
        )

    # Match batch size
    if ref.shape[0] < target.shape[0]:
        repeats = target.shape[0] // ref.shape[0]
        ref = ref.repeat(repeats, 1, 1, 1)

    # If batch sizes still don't match, slice to target size
    if ref.shape[0] > target.shape[0]:
        ref = ref[:target.shape[0]]

    return ref


def make_colorfix_post_cfg_callback(
    x0_origin: torch.Tensor,
    k: int,
    weight: float,
    sharpness: float = 0.0,
    blur_fn: Callable = box_blur
):
    """
    Factory for SD/SDXL post-CFG callback (CFG-based sharpening).

    Args:
        x0_origin:  VAE-encoded reference image tensor (SCALED). Shape (B, C, H, W).
        k:          Blur kernel radius (A1111's "Variation"). Default 8.
        weight:     Color fix strength 0.0-1.0. Default 1.0.
        sharpness:  Sharpness for +sharp variant, 0.0-2.0. Default 0.0 (disabled).
        blur_fn:    Blur function to use (box_blur or gaussian_blur).

    Returns:
        A callback function compatible with model.set_model_sampler_post_cfg_function().
    """
    w = max(0.0, min(1.0, weight))
    use_sharp = sharpness is not None and sharpness > 0.0

    def post_cfg_callback(args):
        """Post-CFG callback applying tile_colorfix at every denoising step."""

        if use_sharp:
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]

            ref = match_size(x0_origin.to(cond_denoised.device), cond_denoised)

            cond_fixed, _ = color_swap_sharp(cond_denoised, ref, k, sharpness, blur_fn)
            _, uncond_softened = color_swap_sharp(uncond_denoised, ref, k, sharpness, blur_fn)

            cond_result = cond_fixed * w + cond_denoised * (1.0 - w)
            uncond_result = uncond_softened * w + uncond_denoised * (1.0 - w)

            return uncond_result + cond_scale * (cond_result - uncond_result)

        else:
            denoised = args["denoised"]

            ref = match_size(x0_origin.to(denoised.device), denoised)

            fixed = color_swap(denoised, ref, k, blur_fn)

            return fixed * w + denoised * (1.0 - w)

    return post_cfg_callback


def make_flux_colorfix_post_cfg_callback(
    x0_origin: torch.Tensor,
    k: int,
    weight: float,
    sharpness: float = 0.0,
    blur_fn: Callable = box_blur
):
    """
    Factory for Flux post-CFG callback (direct sharpening, no CFG dependency).

    Flux models typically run with cfg_scale=1.0 (guidance is embedded in the model),
    so the CFG-differential sharpening approach used for SD/SDXL doesn't work.
    Instead, this uses direct high-pass detail enhancement when sharpness > 0.

    Args:
        x0_origin:  VAE-encoded reference image tensor (SCALED). Shape (B, C, H, W).
                    For Flux, latents have 16 channels.
        k:          Blur kernel radius. Default 8.
        weight:     Color fix strength 0.0-1.0. Default 1.0.
        sharpness:  Detail enhancement strength, 0.0-2.0. Default 0.0 (disabled).
                    Uses direct unsharp-mask style sharpening.
        blur_fn:    Blur function to use (box_blur or gaussian_blur).

    Returns:
        A callback function compatible with model.set_model_sampler_post_cfg_function().
    """
    w = max(0.0, min(1.0, weight))
    use_sharp = sharpness is not None and sharpness > 0.0

    def post_cfg_callback(args):
        """Post-CFG callback applying tile_colorfix for Flux at every denoising step."""
        denoised = args["denoised"]

        ref = match_size(x0_origin.to(denoised.device), denoised)

        if use_sharp:
            fixed = color_swap_sharp_direct(denoised, ref, k, sharpness, blur_fn)
        else:
            fixed = color_swap(denoised, ref, k, blur_fn)

        return fixed * w + denoised * (1.0 - w)

    return post_cfg_callback
