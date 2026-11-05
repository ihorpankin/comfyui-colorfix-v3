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
from .color_swap import color_swap, color_swap_sharp, color_swap_sharp_direct, adain_swap

# transfer_mode values for the Flux callback
TRANSFER_TILE = "tile (spatial)"
TRANSFER_ADAIN = "adain (global)"
TRANSFER_ADAIN_MEAN = "adain mean-only (global tone)"
TRANSFER_MODES = [TRANSFER_TILE, TRANSFER_ADAIN, TRANSFER_ADAIN_MEAN]


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


def make_sigma_to_percent(model_sampling, points: int = 201) -> Callable:
    """
    Build an inverse of model_sampling.percent_to_sigma() via a lookup table.

    Returns a function sigma(float) -> sampling progress in [0, 1]
    (0 = start of denoising / max noise, 1 = end / no noise).
    """
    pcts = [i / (points - 1) for i in range(points)]
    sigmas = [float(model_sampling.percent_to_sigma(p)) for p in pcts]

    def sigma_to_percent(sigma: float) -> float:
        if sigma >= sigmas[0]:
            return 0.0
        if sigma <= sigmas[-1]:
            return 1.0
        # sigmas decrease monotonically with progress
        for i in range(1, points):
            if sigma >= sigmas[i]:
                s_hi, s_lo = sigmas[i - 1], sigmas[i]
                frac = (s_hi - sigma) / (s_hi - s_lo) if s_hi != s_lo else 0.0
                return pcts[i - 1] + frac * (pcts[i] - pcts[i - 1])
        return 1.0

    return sigma_to_percent


def make_flux_colorfix_post_cfg_callback(
    x0_origin: torch.Tensor,
    k: int,
    weight: float,
    sharpness: float = 0.0,
    blur_fn: Callable = box_blur,
    start_percent: float = 0.0,
    k_start: int = 0,
    model_sampling=None,
    transfer_mode: str = TRANSFER_TILE,
    tile_mix: float = 0.0,
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
        start_percent: Sampling progress (0-1) before which the fix is skipped.
                    Early steps decide composition; leaving them alone keeps
                    the model's own geometry. Default 0.0 (fix from step 1).
        k_start:    Blur radius at start_percent, interpolated linearly to k
                    by the end of sampling. A large k_start transfers only very
                    coarse color while composition forms. 0 = constant k.
        model_sampling: The model's model_sampling object. Required when
                    start_percent > 0 or k_start > 0.
        transfer_mode: TRANSFER_TILE (default) swaps low frequencies in place,
                    so colors follow the reference's layout. TRANSFER_ADAIN /
                    TRANSFER_ADAIN_MEAN match global per-channel statistics,
                    so no layout transfers; sharpness is unused, and k and
                    k_start are used only when tile_mix > 0.
        tile_mix:   AdaIN modes only. Share of tile transfer (radius k) blended
                    over the AdaIN result, 0-1: adds position-based color and,
                    with it, some of the reference's layout. Default 0.0.

    Returns:
        A callback function compatible with model.set_model_sampler_post_cfg_function().
    """
    w = max(0.0, min(1.0, weight))
    use_sharp = sharpness is not None and sharpness > 0.0
    start_percent = max(0.0, min(1.0, start_percent or 0.0))
    use_adain = transfer_mode in (TRANSFER_ADAIN, TRANSFER_ADAIN_MEAN)
    adain_std = transfer_mode == TRANSFER_ADAIN
    mix = max(0.0, min(1.0, tile_mix or 0.0)) if use_adain else 0.0
    uses_k = (not use_adain) or mix > 0.0
    use_schedule = uses_k and k_start is not None and k_start > 0 and k_start != k

    sigma_to_percent = None
    if (start_percent > 0.0 or use_schedule) and model_sampling is not None:
        sigma_to_percent = make_sigma_to_percent(model_sampling)

    def post_cfg_callback(args):
        """Post-CFG callback applying tile_colorfix for Flux at every denoising step."""
        denoised = args["denoised"]

        step_k = k
        if sigma_to_percent is not None:
            progress = sigma_to_percent(float(args["sigma"].flatten()[0]))
            if progress < start_percent:
                return denoised
            if use_schedule:
                span = 1.0 - start_percent
                t = (progress - start_percent) / span if span > 0 else 1.0
                step_k = max(1, int(round(k_start + (k - k_start) * t)))

        ref = match_size(x0_origin.to(denoised.device), denoised)

        if use_adain:
            fixed = adain_swap(denoised, ref, match_std=adain_std)
            if mix > 0.0:
                fixed = fixed + mix * (color_swap(fixed, ref, step_k, blur_fn) - fixed)
        elif use_sharp:
            fixed = color_swap_sharp_direct(denoised, ref, step_k, sharpness, blur_fn)
        else:
            fixed = color_swap(denoised, ref, step_k, blur_fn)

        return fixed * w + denoised * (1.0 - w)

    return post_cfg_callback
