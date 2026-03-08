"""
Layer 1 — Apply Ultimate ControlNet Node

Visible ComfyUI node that applies a ControlNet to positive/negative
conditioning with A1111-style control_mode selection.

Control modes (matching A1111 sd-webui-controlnet):
    Balanced                     — Uniform layer weights, control on both paths
    My prompt is more important  — Decayed layer weights (soft_injection)
    ControlNet is more important — Decayed layer weights + zero uncond (cfg_injection)

A1111 mechanism (from hook.py / controlnet.py):
    soft_injection  — Exponential decay per ControlNet layer:
                      scale_i = 0.825^(12-i) for 13 layers.
                      Shallow/detail layers get less weight,
                      deep/structural layers keep full weight.
                      Used by BOTH non-Balanced modes.

    cfg_injection   — Multiply control by cond_mark, zeroing the control
                      signal for uncond batch entries. Only used by
                      "ControlNet is more important".

    | Mode          | soft_injection | cfg_injection |
    |---------------|----------------|---------------|
    | Balanced      | no             | no            |
    | My prompt     | yes            | no            |
    | ControlNet    | yes            | yes           |

Implementation:
    The controlnet is always attached to BOTH positive and negative
    conditioning (required because ComfyUI's sampler shares one control
    object per batch). A ControlModeWrapper intercepts control_merge to
    apply soft_injection and cfg_injection before the standard merge.

Workflow connection:
    Load ControlNet Model → [control_net] → ApplyUltimateControlNet
    CLIP Text Encode      → [positive]    → ApplyUltimateControlNet
    CLIP Text Encode      → [negative]    → ApplyUltimateControlNet
    Load Image            → [image]       → ApplyUltimateControlNet
    ApplyUltimateControlNet → [positive, negative] → KSampler
"""

import torch

# A1111 soft_injection base ratio. In A1111, layer scales are 0.825^(12-i)
# for a fixed 13 layers. We use an adaptive version: 0.825^(total-1-i)
# where total = actual number of ControlNet output layers. This ensures
# the deepest layer always gets scale=1.0 regardless of model architecture
# (SD1.5 has 13 layers, SDXL has ~10).
_SOFT_INJECTION_BASE = 0.825


class ControlModeWrapper:
    """Wraps a ControlNet to apply A1111-style control modes.

    Two mechanisms, applied inside control_merge BEFORE strength scaling
    and before merging with any previous controlnet in the chain:

    soft_injection — per-layer exponential decay weights (both non-Balanced modes)
    cfg_injection  — zero control for uncond batch entries ("ControlNet" mode only)
    """

    def __init__(self, controlnet, control_mode):
        self.wrapped = controlnet
        self.control_mode = control_mode
        self._cond_or_uncond = None

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def copy(self):
        return ControlModeWrapper(self.wrapped.copy(), self.control_mode)

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        if self.control_mode == "Balanced":
            return self.wrapped.get_control(
                x_noisy, t, cond, batched_number, transformer_options
            )

        # Store cond_or_uncond for cfg_injection inside control_merge
        self._cond_or_uncond = transformer_options.get("cond_or_uncond", [])

        # Temporarily patch control_merge so soft_injection and cfg_injection
        # happen BEFORE strength scaling and before merging with previous
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
            self._cond_or_uncond = None

        return result

    def _modified_merge(self, original_merge, control, control_prev, output_dtype):
        """Apply soft_injection + cfg_injection, then delegate to original merge."""

        # --- soft_injection: adaptive exponential decay layer weights ---
        # Both "My prompt" and "ControlNet" modes use this.
        # A1111 uses fixed 13-layer formula, but SDXL has ~10 layers.
        # Adaptive: 0.825^(total-1-i) so deepest layer always = 1.0.
        total_layers = sum(
            1 for k in ('input', 'middle', 'output')
            for v in control.get(k, []) if v is not None
        )
        global_idx = 0
        for key in ('input', 'middle', 'output'):
            if key not in control:
                continue
            for i in range(len(control[key])):
                if control[key][i] is not None:
                    scale = _SOFT_INJECTION_BASE ** (total_layers - 1 - global_idx)
                    control[key][i] = control[key][i] * scale
                    global_idx += 1

        # --- cfg_injection: zero uncond entries ("ControlNet" mode only) ---
        cond_or_uncond = self._cond_or_uncond or []
        if self.control_mode == "ControlNet is more important" and cond_or_uncond:
            mask_values = [0.0 if flag == 1 else 1.0 for flag in cond_or_uncond]
            for key in control:
                for i in range(len(control[key])):
                    if control[key][i] is not None:
                        tensor = control[key][i]
                        mask = torch.tensor(
                            mask_values, device=tensor.device, dtype=tensor.dtype
                        )
                        mask = mask.reshape(-1, *([1] * (tensor.ndim - 1)))
                        control[key][i] = tensor * mask

        return original_merge(control, control_prev, output_dtype)


class ControlNetTileApply:
    """
    Applies a ControlNet with A1111-style control mode selection.

    Wraps the standard ControlNetApplyAdvanced logic but adds a control_mode
    dropdown matching A1111's sd-webui-controlnet behavior:

    - Balanced: uniform layer weights, control on both cond and uncond.
    - My prompt is more important: exponential decay layer weights (soft_injection).
      Reduces detail-level control while keeping structural guidance.
    - ControlNet is more important: same decay + zeroed uncond (cfg_injection).
      At CFG>1, this amplifies ControlNet influence through CFG scaling.
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
                    "tooltip": "A1111-style control mode. "
                               "Balanced: uniform weights, standard behavior. "
                               "My prompt: reduced detail control (soft_injection). "
                               "ControlNet: reduced detail + uncond zeroed (cfg_injection)."
                }),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "colorfix-v3"
    DESCRIPTION = (
        "Applies ControlNet with A1111-style control mode. "
        "Choose Balanced, My prompt is more important, or "
        "ControlNet is more important to control how the ControlNet "
        "interacts with CFG guidance."
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
                        c_net = ControlModeWrapper(c_net, control_mode)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False

                n = [t[0], d]
                c.append(n)
            out.append(c)

        return (out[0], out[1])
