"""
Layer 1 — Apply Ultimate Multi-ControlNet Node (FLUX)

Applies all ControlNets from a CONTROLNET_STACK to positive/negative
conditioning, with per-entry control_mode adapted for Flux models.

Each stack entry is a tuple:
    (controlnet, image, strength, start_percent, end_percent, control_mode)

Uses FluxControlModeWrapper from flux_controlnet_apply.py for symmetric
layer-scaling approach suited to Flux's DiT architecture (no CFG dependency).

Workflow connection:
    UltimateControlNetStack    → [controlnet_stack] → FluxMultiControlNetApply
    CLIP Text Encode           → [positive]         → FluxMultiControlNetApply
    CLIP Text Encode           → [negative]         → FluxMultiControlNetApply
    FluxMultiControlNetApply → [positive, negative] → KSampler
"""

from .flux_controlnet_apply import FluxControlModeWrapper


class FluxMultiControlNetApply:
    """
    Applies a stack of ControlNets with per-entry Flux-adapted control modes.

    Iterates through the CONTROLNET_STACK and chains each ControlNet onto
    the conditioning. Uses symmetric layer-scaling (decay/boost) instead of
    A1111's cfg_injection, since Flux runs at cfg_scale=1.0.

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
    CATEGORY = "colorfix-v3/flux"
    DESCRIPTION = (
        "Applies all ControlNets from a stack with per-entry Flux-adapted "
        "control modes. Uses symmetric layer scaling (decay for prompt priority, "
        "boost for ControlNet priority). Global switch to bypass the entire stack."
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
                            c_net = FluxControlModeWrapper(c_net, control_mode)
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
