"""
Layer 1 — Apply Ultimate Multi-ControlNet Node (SD/SDXL)

Applies all ControlNets from a CONTROLNET_STACK to positive/negative
conditioning, with per-entry A1111-style control_mode.

Each stack entry is a tuple:
    (controlnet, image, strength, start_percent, end_percent, control_mode)

Uses ControlModeWrapper from controlnet_apply.py for soft_injection and
cfg_injection mechanisms matching A1111 sd-webui-controlnet behavior.

Workflow connection:
    UltimateControlNetStack → [controlnet_stack] → ApplyUltimateMultiControlNet
    CLIP Text Encode        → [positive]         → ApplyUltimateMultiControlNet
    CLIP Text Encode        → [negative]         → ApplyUltimateMultiControlNet
    ApplyUltimateMultiControlNet → [positive, negative] → KSampler
"""

from .controlnet_apply import ControlModeWrapper


class MultiControlNetApply:
    """
    Applies a stack of ControlNets with per-entry A1111-style control modes.

    Iterates through the CONTROLNET_STACK and chains each ControlNet onto
    the conditioning, respecting each entry's strength, timing, and control
    mode (Balanced / My prompt / ControlNet is more important).

    A global switch allows disabling the entire stack without disconnecting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "controlnet_stack": ("CONTROLNET_STACK",),
                "switch": (["Off", "On"], {
                    "default": "On",
                    "tooltip": "Global switch. Off = bypass all ControlNets.",
                }),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_stack"
    CATEGORY = "colorfix-v3"
    DESCRIPTION = (
        "Applies all ControlNets from a stack with per-entry A1111-style "
        "control modes. Each entry keeps its own strength, timing, and mode. "
        "Global switch to bypass the entire stack."
    )

    def apply_stack(self, positive, negative, controlnet_stack,
                    switch="On", vae=None):
        if switch == "Off" or not controlnet_stack:
            return (positive, negative)

        for entry in controlnet_stack:
            control_net, image, strength, start_percent, end_percent, control_mode = entry

            if strength == 0:
                continue

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

            positive = out[0]
            negative = out[1]

        return (positive, negative)
