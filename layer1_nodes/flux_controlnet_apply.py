"""
Layer 1 — Apply Ultimate ControlNet (FLUX) Node

Visible ComfyUI node that applies a ControlNet to positive/negative
conditioning with control_mode selection adapted for Flux models.

Flux uses a DiT (Diffusion Transformer) architecture with 19 double blocks
and 38 single blocks = 57 control injection points. It typically runs at
cfg_scale=1.0 with guidance embedded in the model, so the A1111 cfg_injection
mechanism (zeroing uncond entries) has no effect.

Instead, this node uses a symmetric layer-scaling approach:

    | Mode          | Base    | Shallowest | Deepest | Effect                |
    |---------------|---------|------------|---------|---------------------- |
    | Balanced      | —       | 1.0        | 1.0     | Uniform control       |
    | My prompt     | < 1.0   | ~0.06      | 1.0     | Decay → prompt wins   |
    | ControlNet    | > 1.0   | ~1.5       | 1.0     | Boost → CN wins       |

All modes use the same formula: scale = base^(total_layers - 1 - i)
    - base < 1 → shallowest layer gets smallest weight (detail control reduced)
    - base > 1 → shallowest layer gets largest weight (detail control boosted)

The decay base is dynamically computed to maintain the same ~6% minimum
weight as A1111's SD1.5 implementation, regardless of total layer count.

Workflow connection:
    Load ControlNet Model → [control_net] → Apply Ultimate ControlNet (FLUX)
    CLIP Text Encode      → [positive]    → Apply Ultimate ControlNet (FLUX)
    CLIP Text Encode      → [negative]    → Apply Ultimate ControlNet (FLUX)
    Load Image            → [image]       → Apply Ultimate ControlNet (FLUX)
    Apply Ultimate ControlNet (FLUX) → [positive, negative] → KSampler
"""

# A1111 reference values for adaptive base computation.
# SD1.5 uses 13 layers with base 0.825 → shallowest gets 0.825^12 ≈ 0.063.
# We compute a new base for any layer count that preserves this ratio.
_SD_SOFT_INJECTION_BASE = 0.825
_SD_REFERENCE_LAYERS = 13

# Boost target for "ControlNet is more important" mode.
# Shallowest layer gets this multiplier (1.5 = 50% boost).
_FLUX_BOOST_TARGET = 1.5


class FluxControlModeWrapper:
    """Wraps a ControlNet to apply Flux-adapted control modes.

    Two scaling directions, applied inside control_merge BEFORE strength
    scaling and before merging with any previous controlnet in the chain:

    Decay  — "My prompt": per-layer exponential decay (base < 1.0)
             Deep/structural layers keep full weight, shallow/detail layers
             get reduced weight. Adaptive base preserves ~6% min weight.

    Boost  — "ControlNet": per-layer exponential boost (base > 1.0)
             Deep layers keep full weight, shallow/detail layers get
             amplified above 1.0 for stronger ControlNet detail influence.
    """

    def __init__(self, controlnet, control_mode):
        self.wrapped = controlnet
        self.control_mode = control_mode

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def copy(self):
        return FluxControlModeWrapper(self.wrapped.copy(), self.control_mode)

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        if self.control_mode == "Balanced":
            return self.wrapped.get_control(
                x_noisy, t, cond, batched_number, transformer_options
            )

        # Temporarily patch control_merge so layer scaling happens
        # BEFORE strength scaling and before merging with previous
        # controlnets in the chain.
        original_merge = self.wrapped.control_merge
        self.wrapped.control_merge = (
            lambda control, control_prev, output_dtype:
            self._modified_merge(original_merge, control, control_prev, output_dtype)
        )

        try:
            result = self.wrapped.get_control(
                x_noisy, t, cond, batched_number, transformer_options
            )
        finally:
            self.wrapped.control_merge = original_merge

        return result

    def _modified_merge(self, original_merge, control, control_prev, output_dtype):
        """Apply adaptive layer scaling, then delegate to original merge."""

        # Count total non-None control layers across all groups
        # For Flux: 'input' (double blocks) + 'output' (single blocks)
        total_layers = sum(
            1 for k in control
            for v in control[k] if v is not None
        )

        if total_layers <= 1:
            return original_merge(control, control_prev, output_dtype)

        # Compute adaptive base for the current layer count
        if self.control_mode == "My prompt is more important":
            # Decay: same min-weight ratio as SD1.5 (0.825^12 ≈ 0.063)
            target_min = _SD_SOFT_INJECTION_BASE ** (_SD_REFERENCE_LAYERS - 1)
            base = target_min ** (1.0 / (total_layers - 1))
        else:
            # Boost: shallowest layer gets _FLUX_BOOST_TARGET multiplier
            base = _FLUX_BOOST_TARGET ** (1.0 / (total_layers - 1))

        # Apply per-layer scaling
        # Formula: scale = base^(total-1-i)
        #   Decay (base < 1): deep=1.0, shallow=small → reduces detail control
        #   Boost (base > 1): deep=1.0, shallow=large → amplifies detail control
        # Note: Flux ControlNet returns tuples (immutable), so we must
        # convert to lists before modifying values in-place.
        for key in control:
            if isinstance(control[key], tuple):
                control[key] = list(control[key])

        global_idx = 0
        for key in control:
            for i in range(len(control[key])):
                if control[key][i] is not None:
                    scale = base ** (total_layers - 1 - global_idx)
                    control[key][i] = control[key][i] * scale
                    global_idx += 1

        return original_merge(control, control_prev, output_dtype)


class FluxControlNetTileApply:
    """
    Applies a ControlNet with control mode selection adapted for Flux models.

    Three modes providing meaningful control over ControlNet influence:

    - Balanced: uniform layer weights, standard control behavior.
    - My prompt is more important: adaptive exponential decay on detail
      layers. Structural guidance is preserved while fine detail control
      is reduced, letting the model's text guidance dominate.
    - ControlNet is more important: adaptive exponential boost on detail
      layers. Structural guidance is preserved while fine detail control
      is amplified above 1.0, making ControlNet details more prominent.
    """

    CONTROL_MODES = [
        "Balanced",
        "My prompt is more important",
        "ControlNet is more important",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "ControlNet influence strength. "
                               "0.0 = no effect, 1.0 = full effect."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Denoising step fraction to START applying control. "
                               "0.0 = from the beginning."
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Denoising step fraction to STOP applying control. "
                               "1.0 = until the end."
                }),
                "control_mode": (cls.CONTROL_MODES, {
                    "default": "Balanced",
                    "tooltip": "Flux-adapted control mode. "
                               "Balanced: uniform weights, standard behavior. "
                               "My prompt: decay detail-layer control (prompt dominates details). "
                               "ControlNet: boost detail-layer control (ControlNet dominates details)."
                }),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "colorfix-v3/flux"
    DESCRIPTION = (
        "Applies ControlNet with Flux-adapted control mode. "
        "Choose Balanced, My prompt is more important, or "
        "ControlNet is more important to control how the ControlNet "
        "influences generation detail vs structural guidance."
    )

    def apply_controlnet(self, positive, negative, control_net, image,
                         strength, start_percent, end_percent,
                         control_mode="Balanced", vae=None):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(
                        control_hint, strength,
                        (start_percent, end_percent), vae=vae
                    )
                    if control_mode != "Balanced":
                        c_net = FluxControlModeWrapper(c_net, control_mode)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False

                n = [t[0], d]
                c.append(n)
            out.append(c)

        return (out[0], out[1])
