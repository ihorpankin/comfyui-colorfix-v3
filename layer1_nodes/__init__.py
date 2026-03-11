"""
Layer 1 — Visible ComfyUI Nodes

These are the nodes that appear on the ComfyUI canvas. They configure
the colorfix pipeline and patch the model with the appropriate callback.

SD/SDXL Nodes:
    TileColorFixPatcher      — Base tile_colorfix (color swap only)
    TileColorFixSharpPatcher — tile_colorfix+sharp (color swap + detail via CFG)
    ControlNetTileApply      — ControlNet apply with A1111 control_mode
    UltimateControlNetStack  — Multi-ControlNet stack with per-slot control mode
    MultiControlNetApply     — Apply multi-ControlNet stack (SD/SDXL)

Flux Nodes:
    FluxTileColorFixPatcher      — Base tile_colorfix for Flux (16-ch latents)
    FluxTileColorFixSharpPatcher — tile_colorfix+sharp for Flux (direct sharpening)
    FluxControlNetTileApply      — ControlNet apply with Flux-adapted control mode
    FluxMultiControlNetApply     — Apply multi-ControlNet stack (Flux)
"""

from .colorfix_patcher import TileColorFixPatcher
from .colorfix_sharp_patcher import TileColorFixSharpPatcher
from .controlnet_apply import ControlNetTileApply
from .controlnet_stack import UltimateControlNetStack
from .controlnet_multi_apply import MultiControlNetApply
from .flux_colorfix_patcher import FluxTileColorFixPatcher
from .flux_colorfix_sharp_patcher import FluxTileColorFixSharpPatcher
from .flux_controlnet_apply import FluxControlNetTileApply
from .flux_controlnet_multi_apply import FluxMultiControlNetApply

__all__ = [
    "TileColorFixPatcher",
    "TileColorFixSharpPatcher",
    "ControlNetTileApply",
    "UltimateControlNetStack",
    "MultiControlNetApply",
    "FluxTileColorFixPatcher",
    "FluxTileColorFixSharpPatcher",
    "FluxControlNetTileApply",
    "FluxMultiControlNetApply",
]
