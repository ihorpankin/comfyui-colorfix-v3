"""
Layer 1 — FluxTileColorFixPatcher Node

Visible ComfyUI node that patches a Flux MODEL with the tile_colorfix
callback. Designed for Flux models which use 16-channel latents and
typically run without traditional CFG (cfg_scale=1.0).

The core color-swap algorithm is identical to the SD/SDXL variant —
swap low-frequency colors from reference while preserving AI detail.
The difference is that the callback always operates on args["denoised"]
(no CFG splitting needed), and latent scaling is handled automatically
via model.model.latent_format.process_in().

Workflow connection:
    Load Image → VAE Encode → [ref_latent] → FluxTileColorFixPatcher
    Load Diffusion Model (Flux) → [model] → FluxTileColorFixPatcher
    FluxTileColorFixPatcher → [model (patched)] → KSampler
"""

from ..layer2_core import make_flux_colorfix_post_cfg_callback
from ..layer2_core.blur import BLUR_MODE_MAP


class FluxTileColorFixPatcher:
    """
    Patches a Flux model with tile_colorfix color correction.

    Takes a MODEL and a reference LATENT (VAE-encoded control image),
    returns a patched MODEL that applies color correction at every
    denoising step. The correction swaps low-frequency color information
    from the reference into the generation while preserving AI-generated
    detail.

    Designed for Flux models (16-channel latents, no traditional CFG).
    The core algorithm is identical to the SD/SDXL variant.
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
                               "Larger = more aggressive color transfer. "
                               "k=8 → 17x17 kernel."
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
        "Patches Flux model with tile_colorfix color correction. "
        "Swaps low-frequency colors from reference image into generation "
        "while preserving AI-generated detail. "
        "Designed for Flux models (16-ch latents, no CFG dependency)."
    )

    def apply(self, model, ref_latent, blur_k, weight, blur_mode):
        # Clone the model to avoid modifying the original
        m = model.clone()

        # Extract the latent tensor from ComfyUI's LATENT dict
        x0_origin = ref_latent["samples"].clone()

        # Scale reference to match Flux's internal latent space.
        # model.model.latent_format.process_in() handles the correct
        # scaling factor for Flux automatically.
        x0_origin = model.model.latent_format.process_in(x0_origin)

        # Resolve blur function from dropdown selection
        blur_fn = BLUR_MODE_MAP[blur_mode]

        # Create the Flux-specific post-CFG callback
        # Uses the simple path (no CFG splitting)
        callback = make_flux_colorfix_post_cfg_callback(
            x0_origin=x0_origin,
            k=blur_k,
            weight=weight,
            sharpness=0.0,  # Base variant — no sharpening
            blur_fn=blur_fn
        )

        # Register the callback on the model
        m.set_model_sampler_post_cfg_function(callback)

        return (m,)
