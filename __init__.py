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
    UltimateControlNetStack  — Multi-ControlNet stack with per-slot control mode
    MultiControlNetApply     — Apply multi-ControlNet stack (SD/SDXL)

Flux Nodes:
    FluxTileColorFixPatcher      — Base color correction for Flux models
    FluxTileColorFixSharpPatcher — Color correction + direct detail enhancement
    FluxControlNetTileApply      — Apply ControlNet with Flux-adapted control modes
    FluxMultiControlNetApply     — Apply multi-ControlNet stack (Flux)

Usage (SD/SDXL):
    Load Image → VAE Encode → TileColorFixPatcher → KSampler

Usage (Flux):
    Load Image → VAE Encode (Flux VAE) → FluxTileColorFixPatcher → KSampler
"""

from .layer1_nodes import (
    TileColorFixPatcher,
    TileColorFixSharpPatcher,
    ControlNetTileApply,
    UltimateControlNetStack,
    MultiControlNetApply,
    FluxTileColorFixPatcher,
    FluxTileColorFixSharpPatcher,
    FluxControlNetTileApply,
    FluxMultiControlNetApply,
)

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "TileColorFixPatcher": TileColorFixPatcher,
    "TileColorFixSharpPatcher": TileColorFixSharpPatcher,
    "ControlNetTileApply": ControlNetTileApply,
    "UltimateControlNetStack": UltimateControlNetStack,
    "MultiControlNetApply": MultiControlNetApply,
    "FluxTileColorFixPatcher": FluxTileColorFixPatcher,
    "FluxTileColorFixSharpPatcher": FluxTileColorFixSharpPatcher,
    "FluxControlNetTileApply": FluxControlNetTileApply,
    "FluxMultiControlNetApply": FluxMultiControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileColorFixPatcher": "Tile ColorFix Patcher",
    "TileColorFixSharpPatcher": "Tile ColorFix+Sharp Patcher",
    "ControlNetTileApply": "Apply Ultimate ControlNet",
    "UltimateControlNetStack": "Ultimate Multi-ControlNet Stack",
    "MultiControlNetApply": "Apply Ultimate Multi-ControlNet",
    "FluxTileColorFixPatcher": "Flux Tile ColorFix Patcher",
    "FluxTileColorFixSharpPatcher": "Flux Tile ColorFix+Sharp Patcher",
    "FluxControlNetTileApply": "Apply Ultimate ControlNet (FLUX)",
    "FluxMultiControlNetApply": "Apply Ultimate Multi-ControlNet (FLUX)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
