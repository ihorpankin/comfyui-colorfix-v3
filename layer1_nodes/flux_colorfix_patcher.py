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

Optional inputs: transfer_mode selects a global AdaIN transfer that
carries no layout from the reference; tile_mix blends tile transfer over
it; start_percent skips early steps; blur_k_start schedules the tile radius.

Workflow connection:
    Load Image → VAE Encode → [ref_latent] → FluxTileColorFixPatcher
    Load Diffusion Model (Flux) → [model] → FluxTileColorFixPatcher
    FluxTileColorFixPatcher → [model (patched)] → KSampler
"""

from ..layer2_core import make_flux_colorfix_post_cfg_callback
from ..layer2_core.blur import BLUR_MODE_MAP
from ..layer2_core.sampler_callback import TRANSFER_MODES, TRANSFER_TILE


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
                               "Smaller = stronger color but more of the reference's "
                               "structure; larger = less of both. "
                               "k=8 → 17x17 kernel. Tile mode only."
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
            "optional": {
                "transfer_mode": (TRANSFER_MODES, {
                    "default": TRANSFER_TILE,
                    "tooltip": "tile = swap blurred colors in place: colors follow the "
                               "reference's layout (best when reference and output share "
                               "composition, e.g. img2img/upscale). "
                               "adain = match each latent channel's mean and spread: "
                               "palette and tone only, no reference shapes (best for an "
                               "unrelated reference). mean-only = shift tone, keep contrast. "
                               "With adain, blur_k / blur_mode / blur_k_start apply only "
                               "when tile_mix is above 0."
                }),
                "tile_mix": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "adain modes only. Share of tile transfer blended over "
                               "the adain result. Higher = color closer to the reference's "
                               "positions, but more of its layout. 0 = pure adain."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Skip the color fix before this point of sampling. "
                               "Early steps decide composition; 0.1-0.2 keeps the "
                               "model's own pose and shapes instead of the reference's. "
                               "0.0 = fix from the first step."
                }),
                "blur_k_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Blur radius at the first fixed step, moving linearly "
                               "to blur_k by the end. A large value (e.g. 32) passes "
                               "only very coarse color while composition forms. "
                               "0 = use blur_k throughout."
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
        "Works with FLUX.1 and FLUX.3 Klein (no CFG dependency)."
    )

    def apply(self, model, ref_latent, blur_k, weight, blur_mode,
              transfer_mode=TRANSFER_TILE, tile_mix=0.0, start_percent=0.0, blur_k_start=0):
        # Clone the model to avoid modifying the original
        m = model.clone()

        # Extract the latent tensor from ComfyUI's LATENT dict
        x0_origin = ref_latent["samples"].clone()

        # Scale reference to match Flux's internal latent space.
        # model.model.latent_format.process_in() handles the correct
        # scaling factor for Flux automatically.
        latent_format = model.model.latent_format
        if getattr(latent_format, "latent_dimensions", 2) == 3 and x0_origin.ndim == 4:
            # Models with a time axis in their latent format scale 5D tensors
            x0_origin = x0_origin.unsqueeze(2)
        x0_origin = latent_format.process_in(x0_origin)

        # Resolve blur function from dropdown selection
        blur_fn = BLUR_MODE_MAP[blur_mode]

        # Create the Flux-specific post-CFG callback
        # Uses the simple path (no CFG splitting)
        callback = make_flux_colorfix_post_cfg_callback(
            x0_origin=x0_origin,
            k=blur_k,
            weight=weight,
            sharpness=0.0,  # Base variant — no sharpening
            blur_fn=blur_fn,
            start_percent=start_percent,
            k_start=blur_k_start,
            model_sampling=m.get_model_object("model_sampling"),
            transfer_mode=transfer_mode,
            tile_mix=tile_mix,
        )

        # Register the callback on the model
        m.set_model_sampler_post_cfg_function(callback)

        return (m,)
