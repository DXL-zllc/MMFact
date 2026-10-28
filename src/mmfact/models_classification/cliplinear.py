"""Frozen paired CLIP features and a supervised linear head (C06)."""

from mmfact.models_classification._neural_baselines import _NeuralBaseline
from mmfact.registry import register_model


@register_model("CLIPLinear")
class CLIPLinear(_NeuralBaseline):
    clip = True
    head_kind = "linear"
