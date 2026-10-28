"""Text-only language PLM with a supervised classification head (C03)."""

from mmfact.models_classification._neural_baselines import _NeuralBaseline
from mmfact.registry import register_model


@register_model("TextPLM")
class TextPLM(_NeuralBaseline):
    modalities = ("text",)
