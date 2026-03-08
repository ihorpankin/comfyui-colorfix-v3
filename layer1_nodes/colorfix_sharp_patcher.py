"""
Layer 1 — TileColorFixSharpPatcher Node

Visible ComfyUI node that patches a MODEL with the tile_colorfix+sharp
callback. Like the base variant, but adds a sharpness parameter that
enhances detail through CFG manipulation.

How +sharp works:
    - Conditional path gets full color-fixed x0 (sharp details)
    - Unconditional path gets a slightly blurred version
    - CFG amplifies the DIFFERENCE between them → detail boost
    - Higher sharpness = more aggressive detail enhancement

Equivalent A1111 parameters:
    blur_k    = threshold_a ("Variation" slider, default 8, range 3-32)
    weight    = param.weight (ControlNet weight, default 1.0)
    sharpness = threshold_b ("Sharpness" slider, default 1.0, range 0-2)
"""

from ..layer2_core import make_colorfix_post_cfg_callback
from ..layer2_core.blur import BLUR_MODE_MAP


class TileColorFixSharpPatcher:
    """
    Patches a model with tile_colorfix+sharp color correction (SD/SDXL).

    Identical to TileColorFixPatcher but adds a sharpness parameter
    that enhances detail through differential CFG processing.
    The conditional path keeps full detail while the unconditional
    path is softened — CFG amplifies the difference.

    This produces identical results to A1111's tile_colorfix+sharp preprocessor.
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
                    "tooltip": "Detail enhancement strength (A1111 'Sharpness'). "
                               "Internally mapped to detail_weight = sharpness * 0.01. "
                               "0.0 = no sharpening, 1.0 = subtle detail boost, "
                               "2.0 = more aggressive sharpening."
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
        "Patches model with tile_colorfix+sharp color correction (SD/SDXL). "
        "Same as TileColorFixPatcher but adds detail enhancement via "
        "differential CFG processing. Identical to A1111 tile_colorfix+sharp."
    )

    def apply(self, model, ref_latent, blur_k, weight, sharpness, blur_mode):
        # Clone the model to avoid modifying the original
        m = model.clone()

        # Extract the latent tensor from ComfyUI's LATENT dict
        x0_origin = ref_latent["samples"].clone()

        # CRITICAL: Scale reference to match the model's internal latent space.
        # ComfyUI's KSampler calls model.process_latent_in() which multiplies
        # latents by scale_factor (0.18215 for SD1.5) before sampling.
        # Inside the sampling loop, denoised predictions are in this SCALED space.
        # But VAE Encode output is UNSCALED. We must match the scale.
        x0_origin = model.model.latent_format.process_in(x0_origin)

        # Resolve blur function from dropdown selection
        blur_fn = BLUR_MODE_MAP[blur_mode]

        # Create the post-CFG callback with sharpness enabled
        callback = make_colorfix_post_cfg_callback(
            x0_origin=x0_origin,
            k=blur_k,
            weight=weight,
            sharpness=sharpness,
            blur_fn=blur_fn
        )

        # Register the callback on the model
        # disable_cfg1_optimization=True ensures uncond is always computed,
        # which is needed for the +sharp variant's differential processing
        m.set_model_sampler_post_cfg_function(callback, disable_cfg1_optimization=True)

        return (m,)
