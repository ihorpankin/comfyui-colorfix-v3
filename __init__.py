"""
comfyui-colorfix-v3
===================

ComfyUI custom node extension implementing the tile_colorfix algorithm
from A1111's ControlNet extension. Supports both SD/SDXL and Flux models.

Architecture:
    layer1_nodes/  — Visible ComfyUI nodes (model patchers)
    layer2_core/   — Internal algorithm (blur, color swap, sampler callbacks)

SD/SDXL Nodes:
    TileColorFixPatcher      — Base color correction (swaps colors from reference)
    TileColorFixSharpPatcher — Color correction + detail enhancement via CFG
    ControlNetTileApply      — Apply ControlNet with A1111 control modes

Flux Nodes:
    FluxTileColorFixPatcher      — Base color correction for Flux models
    FluxTileColorFixSharpPatcher — Color correction + direct detail enhancement

Usage (SD/SDXL):
    Load Image → VAE Encode → TileColorFixPatcher → KSampler

Usage (Flux):
    Load Image → VAE Encode (Flux VAE) → FluxTileColorFixPatcher → KSampler
"""

from .layer1_nodes import (
    TileColorFixPatcher,
    TileColorFixSharpPatcher,
    ControlNetTileApply,
    FluxTileColorFixPatcher,
    FluxTileColorFixSharpPatcher,
)

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "TileColorFixPatcher": TileColorFixPatcher,
    "TileColorFixSharpPatcher": TileColorFixSharpPatcher,
    "ControlNetTileApply": ControlNetTileApply,
    "FluxTileColorFixPatcher": FluxTileColorFixPatcher,
    "FluxTileColorFixSharpPatcher": FluxTileColorFixSharpPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileColorFixPatcher": "Tile ColorFix Patcher",
    "TileColorFixSharpPatcher": "Tile ColorFix+Sharp Patcher",
    "ControlNetTileApply": "Apply Ultimate ControlNet",
    "FluxTileColorFixPatcher": "Flux Tile ColorFix Patcher",
    "FluxTileColorFixSharpPatcher": "Flux Tile ColorFix+Sharp Patcher",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
