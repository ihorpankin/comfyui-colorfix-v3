"""
Layer 1 — FluxTileColorFixSharpPatcher Node

Visible ComfyUI node that patches a Flux MODEL with tile_colorfix +
direct sharpening. Unlike the SD/SDXL sharp variant which relies on
CFG-differential processing, this uses direct high-pass detail
enhancement (unsharp mask style) since Flux runs at cfg_scale=1.0.

How direct sharpening works:
    1. Apply base color swap: fixed = x0 - blur(x0) + blur(ref)
    2. Extract detail: high_freq = fixed - blur(fixed)
    3. Enhance: result = fixed + sharpness * 0.1 * high_freq

The sharpness parameter controls detail boost intensity.
The 0.1 scaling maps the 0-2 range to produce comparable
visual effect to the CFG-based variant at typical settings.

Workflow connection:
    Load Image → VAE Encode → [ref_latent] → FluxTileColorFixSharpPatcher
    Load Diffusion Model (Flux) → [model] → FluxTileColorFixSharpPatcher
    FluxTileColorFixSharpPatcher → [model (patched)] → KSampler
"""

from ..layer2_core import make_flux_colorfix_post_cfg_callback
from ..layer2_core.blur import BLUR_MODE_MAP


class FluxTileColorFixSharpPatcher:
    """
    Patches a Flux model with tile_colorfix + direct sharpening.

    Unlike the SD/SDXL sharp variant which uses CFG-differential
    processing, this applies direct high-pass detail enhancement.
    Flux models run at cfg_scale=1.0 with guidance embedded in the
    model, so CFG-based sharpening has no effect.

    The direct approach extracts high-frequency detail from the
    color-fixed result and adds it back at controlled intensity,
    achieving the same goal of detail enhancement without CFG.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT", {
                    "tooltip": "VAE-encoded control/reference image. "
                               "Connect from VAE Encode node. This provides "
                               "the color palette to transfer. "
                               "Use the Flux VAE for encoding."
                }),
                "blur_k": ("INT", {
                    "default": 8,
                    "min": 3,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Blur kernel radius. "
                               "Controls frequency split point. "
                               "Larger = more aggressive color transfer."
                }),
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Color fix strength. "
                               "0.0 = no correction, 1.0 = full color replacement."
                }),
                "sharpness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Detail enhancement strength. "
                               "Uses direct high-pass boost (unsharp mask). "
                               "0.0 = no sharpening, 1.0 = moderate detail boost, "
                               "2.0 = strong sharpening."
                }),
                "blur_mode": (list(BLUR_MODE_MAP.keys()), {
                    "default": "gaussian (smoother)",
                    "tooltip": "Blur algorithm for frequency split. "
                               "Gaussian = smoother falloff, recommended for Flux. "
                               "Box = sharp cutoff, A1111 original."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "colorfix-v3/flux"
    DESCRIPTION = (
        "Patches Flux model with tile_colorfix + direct sharpening. "
        "Same color correction as FluxTileColorFixPatcher but adds "
        "detail enhancement via high-pass boost (unsharp mask). "
        "Designed for Flux models (no CFG dependency)."
    )

    def apply(self, model, ref_latent, blur_k, weight, sharpness, blur_mode):
        # Clone the model to avoid modifying the original
        m = model.clone()

        # Extract the latent tensor from ComfyUI's LATENT dict
        x0_origin = ref_latent["samples"].clone()

        # Scale reference to match Flux's internal latent space.
        x0_origin = model.model.latent_format.process_in(x0_origin)

        # Resolve blur function from dropdown selection
        blur_fn = BLUR_MODE_MAP[blur_mode]

        # Create the Flux-specific post-CFG callback with direct sharpening
        callback = make_flux_colorfix_post_cfg_callback(
            x0_origin=x0_origin,
            k=blur_k,
            weight=weight,
            sharpness=sharpness,
            blur_fn=blur_fn
        )

        # Register the callback on the model
        # No need for disable_cfg1_optimization since Flux sharpening
        # is direct (doesn't need separate cond/uncond paths)
        m.set_model_sampler_post_cfg_function(callback)

        return (m,)
