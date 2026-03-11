"""
Layer 1 — Ultimate Multi-ControlNet Stack Node

Builds a CONTROLNET_STACK list with up to 3 ControlNet entries, each with
its own switch, model, image, strength, start/end percent, and A1111-style
control_mode. Stacks can be chained by feeding one stack's output into
another's controlnet_stack input.

Each stack entry is a tuple:
    (controlnet, image, strength, start_percent, end_percent, control_mode)

Works with both SD/SDXL and Flux apply nodes — the control_mode string
is interpreted by the corresponding apply node's wrapper.

Workflow connection:
    Load ControlNet Model → [controlnet_1] → UltimateControlNetStack
    Load Image            → [image_1]      → UltimateControlNetStack
    UltimateControlNetStack → [CONTROLNET_STACK] → ApplyUltimateMultiControlNet
"""

CONTROL_MODES = [
    "Balanced",
    "My prompt is more important",
    "ControlNet is more important",
]


class UltimateControlNetStack:
    """
    Builds a multi-ControlNet stack with per-entry control mode selection.

    Up to 3 ControlNet slots, each independently switchable. Chain multiple
    stacks by connecting the controlnet_stack output to another stack's input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_1": (["Off", "On"], {"default": "Off"}),
                "controlnet_1": ("CONTROL_NET",),
                "image_1": ("IMAGE",),
                "strength_1": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "ControlNet influence strength for slot 1.",
                }),
                "start_percent_1": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Denoising fraction to START applying slot 1.",
                }),
                "end_percent_1": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Denoising fraction to STOP applying slot 1.",
                }),
                "control_mode_1": (CONTROL_MODES, {
                    "default": "Balanced",
                    "tooltip": "A1111-style control mode for slot 1.",
                }),
            },
            "optional": {
                "switch_2": (["Off", "On"], {"default": "Off"}),
                "controlnet_2": ("CONTROL_NET",),
                "image_2": ("IMAGE",),
                "strength_2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "ControlNet influence strength for slot 2.",
                }),
                "start_percent_2": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Denoising fraction to START applying slot 2.",
                }),
                "end_percent_2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Denoising fraction to STOP applying slot 2.",
                }),
                "control_mode_2": (CONTROL_MODES, {
                    "default": "Balanced",
                    "tooltip": "A1111-style control mode for slot 2.",
                }),
                "switch_3": (["Off", "On"], {"default": "Off"}),
                "controlnet_3": ("CONTROL_NET",),
                "image_3": ("IMAGE",),
                "strength_3": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "ControlNet influence strength for slot 3.",
                }),
                "start_percent_3": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Denoising fraction to START applying slot 3.",
                }),
                "end_percent_3": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Denoising fraction to STOP applying slot 3.",
                }),
                "control_mode_3": (CONTROL_MODES, {
                    "default": "Balanced",
                    "tooltip": "A1111-style control mode for slot 3.",
                }),
                "controlnet_stack": ("CONTROLNET_STACK",),
            },
        }

    RETURN_TYPES = ("CONTROLNET_STACK",)
    RETURN_NAMES = ("controlnet_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "colorfix-v3"
    DESCRIPTION = (
        "Builds a multi-ControlNet stack with per-slot control mode. "
        "Each slot has its own switch, strength, timing, and A1111-style "
        "control mode (Balanced / My prompt / ControlNet is more important). "
        "Chain multiple stacks via the controlnet_stack input."
    )

    def build_stack(self, switch_1, controlnet_1, image_1,
                    strength_1, start_percent_1, end_percent_1,
                    control_mode_1="Balanced",
                    switch_2="Off", controlnet_2=None, image_2=None,
                    strength_2=1.0, start_percent_2=0.0, end_percent_2=1.0,
                    control_mode_2="Balanced",
                    switch_3="Off", controlnet_3=None, image_3=None,
                    strength_3=1.0, start_percent_3=0.0, end_percent_3=1.0,
                    control_mode_3="Balanced",
                    controlnet_stack=None):

        stack = list(controlnet_stack) if controlnet_stack is not None else []

        slots = [
            (switch_1, controlnet_1, image_1, strength_1,
             start_percent_1, end_percent_1, control_mode_1),
            (switch_2, controlnet_2, image_2, strength_2,
             start_percent_2, end_percent_2, control_mode_2),
            (switch_3, controlnet_3, image_3, strength_3,
             start_percent_3, end_percent_3, control_mode_3),
        ]

        for switch, cnet, img, strength, start, end, mode in slots:
            if switch == "On" and cnet is not None and img is not None:
                stack.append((cnet, img, strength, start, end, mode))

        return (stack,)
