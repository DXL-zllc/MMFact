"""Frozen paired CLIP features and a supervised MLP head (C07)."""

from mmfact.models_classification._neural_baselines import _NeuralBaseline
from mmfact.registry import register_model


@register_model("CLIPMLP")
class CLIPMLP(_NeuralBaseline):
    clip = True
    head_kind = "mlp"
