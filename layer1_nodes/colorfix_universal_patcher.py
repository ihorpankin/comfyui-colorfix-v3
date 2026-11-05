"""
Layer 1 — ColorFixPatcher Node

One color-fix node for any image model that runs through KSampler. It patches
the MODEL with a post-CFG callback, and nothing in it depends on the model
family: latent scaling comes from the model's latent format, and the blur
radius is set in image pixels and converted using the model's latent downscale.

Transfer modes:
    tile  — Swap blurred colors in place, so colors follow the reference's layout.
    adain — Match each latent channel's mean and spread: palette only, no layout.
    tile_mix blends tile transfer over the adain result.

Workflow connection:
    Load Image → VAE Encode → [ref_latent] → ColorFixPatcher
    Load model → [model] → ColorFixPatcher
    ColorFixPatcher → [model (patched)] → KSampler
"""

from ..layer2_core import make_flux_colorfix_post_cfg_callback
from ..layer2_core.blur import BLUR_MODE_MAP, radius_px_to_k
from ..layer2_core.sampler_callback import TRANSFER_MODES, TRANSFER_ADAIN


class ColorFixPatcher:
    """
    Patches a model to transfer color from a reference at every denoising step.

    Defaults are the settings that worked best across the models tested with an
    unrelated reference. For img2img or upscaling, where the reference shares
    the output's composition, use tile mode with start_percent 0.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT", {
                    "tooltip": "VAE-encoded reference image. Encode it with the VAE "
                               "that matches the model."
                }),
                "transfer_mode": (TRANSFER_MODES, {
                    "default": TRANSFER_ADAIN,
                    "tooltip": "tile = swap blurred colors in place: strongest object "
                               "color, but colors follow the reference's layout. Best when "
                               "reference and output share a composition. "
                               "adain = match each latent channel's mean and spread: "
                               "palette and tone only, no reference shapes. Best for an "
                               "unrelated reference. mean-only = shift tone, keep contrast."
                }),
                "blur_radius_px": ("INT", {
                    "default": 256,
                    "min": 8,
                    "max": 1024,
                    "step": 8,
                    "tooltip": "Blur radius in image pixels, converted per model. "
                               "Smaller = stronger color but more of the reference's "
                               "structure. Used by tile mode and by tile_mix. "
                               "Try 192 for tile mode and 256 for the hybrid."
                }),
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Color fix strength. 0.0 = no correction, 1.0 = full."
                }),
                "tile_mix": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "adain modes only. Share of tile transfer blended over "
                               "the adain result. Higher = color closer to the reference's "
                               "positions, but more of its layout. Lower it if the "
                               "reference's backdrop bleeds onto your subject."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Skip the fix before this point of sampling. Early steps "
                               "decide composition, so a later start keeps more of the "
                               "model's own shapes. 0.0 = fix from the first step."
                }),
                "blur_mode": (list(BLUR_MODE_MAP.keys()), {
                    "default": "gaussian (smoother)",
                    "tooltip": "Blur algorithm. Gaussian = smoother falloff. "
                               "Box = sharp cutoff, A1111 original."
                }),
            },
            "optional": {
                "blur_radius_start_px": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "step": 8,
                    "tooltip": "Start the blur radius here and move linearly to "
                               "blur_radius_px by the end. 0 = constant radius."
                }),
                "sharpness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Tile mode only. Direct high-pass detail boost. "
                               "0.0 = off, 1.0 = moderate."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "colorfix-v3"
    DESCRIPTION = (
        "Transfers color from a reference image at every denoising step, on any "
        "image model that runs through KSampler. adain keeps the model's composition, "
        "tile matches colors by position, and tile_mix blends the two."
    )

    def apply(self, model, ref_latent, transfer_mode, blur_radius_px, weight, tile_mix,
              start_percent, blur_mode, blur_radius_start_px=0, sharpness=0.0):
        m = model.clone()
        latent_format = model.model.latent_format
        downscale = latent_format.spacial_downscale_ratio

        x0_origin = ref_latent["samples"].clone()
        if getattr(latent_format, "latent_dimensions", 2) == 3 and x0_origin.ndim == 4:
            # Models with a time axis in their latent format scale 5D tensors
            x0_origin = x0_origin.unsqueeze(2)
        x0_origin = latent_format.process_in(x0_origin)

        k_start = radius_px_to_k(blur_radius_start_px, downscale) if blur_radius_start_px > 0 else 0

        callback = make_flux_colorfix_post_cfg_callback(
            x0_origin=x0_origin,
            k=radius_px_to_k(blur_radius_px, downscale),
            weight=weight,
            sharpness=sharpness,
            blur_fn=BLUR_MODE_MAP[blur_mode],
            start_percent=start_percent,
            k_start=k_start,
            model_sampling=m.get_model_object("model_sampling"),
            transfer_mode=transfer_mode,
            tile_mix=tile_mix,
        )
        m.set_model_sampler_post_cfg_function(callback)

        return (m,)
