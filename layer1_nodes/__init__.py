"""
Layer 1 — Visible ComfyUI Nodes

These are the nodes that appear on the ComfyUI canvas. They configure
the colorfix pipeline and patch the model with the appropriate callback.

SD/SDXL Nodes:
    TileColorFixPatcher      — Base tile_colorfix (color swap only)
    TileColorFixSharpPatcher — tile_colorfix+sharp (color swap + detail via CFG)
    ControlNetTileApply      — ControlNet apply with A1111 control_mode

Flux Nodes:
    FluxTileColorFixPatcher      — Base tile_colorfix for Flux (16-ch latents)
    FluxTileColorFixSharpPatcher — tile_colorfix+sharp for Flux (direct sharpening)
"""

from .colorfix_patcher import TileColorFixPatcher
from .colorfix_sharp_patcher import TileColorFixSharpPatcher
from .controlnet_apply import ControlNetTileApply
from .flux_colorfix_patcher import FluxTileColorFixPatcher
from .flux_colorfix_sharp_patcher import FluxTileColorFixSharpPatcher

__all__ = [
    "TileColorFixPatcher",
    "TileColorFixSharpPatcher",
    "ControlNetTileApply",
    "FluxTileColorFixPatcher",
    "FluxTileColorFixSharpPatcher",
]
