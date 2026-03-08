"""
Layer 2 — Core Algorithm

Internal algorithm components that run inside the KSampler at every
denoising step. This layer is NOT visible as nodes on the ComfyUI canvas.

Modules:
    blur.py             — Box blur and Gaussian blur functions
    color_swap.py       — Color swap, CFG-sharpening, and direct-sharpening logic
    sampler_callback.py — Post-CFG callback factories (SD/SDXL + Flux)
"""

from .blur import box_blur, gaussian_blur, BLUR_MODE_MAP
from .color_swap import color_swap, color_swap_sharp, color_swap_sharp_direct
from .sampler_callback import (
    make_colorfix_post_cfg_callback,
    make_flux_colorfix_post_cfg_callback,
    match_size,
)

__all__ = [
    "box_blur",
    "gaussian_blur",
    "BLUR_MODE_MAP",
    "color_swap",
    "color_swap_sharp",
    "color_swap_sharp_direct",
    "make_colorfix_post_cfg_callback",
    "make_flux_colorfix_post_cfg_callback",
    "match_size",
]
