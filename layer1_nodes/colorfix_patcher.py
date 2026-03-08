"""
Layer 1 — TileColorFixPatcher Node

Visible ComfyUI node that patches a MODEL with the tile_colorfix
callback. The patched model will apply color correction at every
denoising step inside the KSampler.

Workflow connection:
    Load Image → VAE Encode → [ref_latent] → TileColorFixPatcher
    Load Checkpoint → [model] → TileColorFixPatcher
    TileColorFixPatcher → [model (patched)] → KSampler

Equivalent A1111 parameters:
    blur_k  = threshold_a ("Variation" slider, default 8, range 3-32)
    weight  = param.weight (ControlNet weight, default 1.0)
"""

from ..layer2_core import make_colorfix_post_cfg_callback
from ..layer2_core.blur import BLUR_MODE_MAP


class TileColorFixPatcher:
    """
    Patches a model with tile_colorfix color correction.

    Takes a MODEL and a reference LATENT (VAE-encoded control image),
    returns a patched MODEL that applies color correction at every
    denoising step. The correction swaps low-frequency color information
    from the reference into the generation while preserving AI-generated
    detail.

    This produces identical results to A1111's tile_colorfix preprocessor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT", {
                    "tooltip": "VAE-encoded control/reference image. "
                               "Connect from VAE Encode node. This provides "
                               "the color palette to transfer."
                }),
                "blur_k": ("INT", {
                    "default": 8,
                    "min": 3,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Blur kernel radius (A1111 'Variation'). "
                               "Controls frequency split point. "
                               "Larger = more aggressive color transfer. "
                               "k=8 → 17x17 kernel → ~136px blur in image space."
                }),
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Color fix strength. "
                               "0.0 = no correction, 1.0 = full color replacement."
                }),
                "blur_mode": (list(BLUR_MODE_MAP.keys()), {
                    "default": "box (A1111 original)",
                    "tooltip": "Blur algorithm for frequency split. "
                               "Box = A1111 exact match, sharp cutoff. "
                               "Gaussian = smoother falloff, reduces patchy artifacts."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "colorfix-v3"
    DESCRIPTION = (
        "Patches model with tile_colorfix color correction (SD/SDXL). "
        "Swaps low-frequency colors from reference image into generation "
        "while preserving AI-generated detail. Identical to A1111 tile_colorfix."
    )

    def apply(self, model, ref_latent, blur_k, weight, blur_mode):
        # Clone the model to avoid modifying the original
        m = model.clone()

        # Extract the latent tensor from ComfyUI's LATENT dict
        x0_origin = ref_latent["samples"].clone()

        # CRITICAL: Scale reference to match the model's internal latent space.
        # ComfyUI's KSampler calls model.process_latent_in() which multiplies
        # latents by scale_factor (0.18215 for SD1.5) before sampling.
        # Inside the sampling loop, denoised predictions are in this SCALED space.
        # But VAE Encode output is UNSCALED. We must match the scale so that
        # color_swap operates on tensors with the same magnitude.
        x0_origin = model.model.latent_format.process_in(x0_origin)

        # Resolve blur function from dropdown selection
        blur_fn = BLUR_MODE_MAP[blur_mode]

        # Create the post-CFG callback using Layer 2 core
        callback = make_colorfix_post_cfg_callback(
            x0_origin=x0_origin,
            k=blur_k,
            weight=weight,
            sharpness=0.0,  # Base variant — no sharpening
            blur_fn=blur_fn
        )

        # Register the callback on the model
        # This runs at every denoising step, AFTER CFG combination
        m.set_model_sampler_post_cfg_function(callback)

        return (m,)
