"""GAMED-PaperCorrected: full experts, news-label consistency and veto voting."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import RGBEncoder
from mmfact.models_classification._gamed_encoders import (
    _GAMEDImageTokens,
    _GAMEDPattern,
    _GAMEDTextTokens,
    _validate_gamed_analyzers,
)
from mmfact.registry import register_model

_HEADS = ("ip", "is", "t", "mm", "mix")


class _MLP(nn.Sequential):
    def __init__(self, inputs, hidden, outputs):
        super().__init__(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def _validate_tokens(values, mask):
    if values.ndim != 3 or mask.shape != values.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("GAMED tokens require [B,L,D] values and a boolean [B,L] mask.")
    if not mask.any(-1).all():
        raise ValueError("GAMED requires at least one valid token per example.")


def _masked_mean(values, mask):
    _validate_tokens(values, mask)
    return values.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)


class _MaskedBlock(nn.Module):
    """Pre-norm self-attention; padding cannot enter keys, residuals or pooling."""

    def __init__(self, dimension, heads, multiplier, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.projection = nn.Linear(dimension, dimension)
        self.attention_dropout, self.dropout = dropout, nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * multiplier),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * multiplier, dimension),
        )

    def forward(self, values, mask):
        _validate_tokens(values, mask)
        values = values.masked_fill(~mask[..., None], 0)
        batch, length, dimension = values.shape
        query, key, value = (
            self.qkv(self.norm1(values))
            .reshape(batch, length, 3, self.heads, dimension // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = (
            F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask[:, None, None, :],
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(batch, length, dimension)
        )
        values = values + self.dropout(self.projection(attended))
        values = values + self.dropout(self.ffn(self.norm2(values)))
        return values.masked_fill(~mask[..., None], 0)


class _Expert(nn.Module):
    def __init__(self, dimension, heads, multiplier, depth, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_MaskedBlock(dimension, heads, multiplier, dropout) for _ in range(depth)]
        )

    def forward(self, values, mask):
        for block in self.blocks:
            values = block(values, mask)
        return values


class _Mixture(nn.Module):
    """Tasks share experts, with independent signed token attention and raw gates."""

    def __init__(
        self,
        dimension,
        experts,
        heads,
        multiplier,
        hidden,
        tasks,
        depth=1,
        dropout=0.0,
        epsilon=1e-6,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.experts = nn.ModuleList(
            [_Expert(dimension, heads, multiplier, depth, dropout) for _ in range(experts)]
        )
        self.attention = nn.ModuleList([_MLP(dimension, hidden, 1) for _ in range(tasks)])
        self.gates = nn.ModuleList([_MLP(dimension, hidden, experts) for _ in range(tasks)])

    def forward(self, values, mask):
        _validate_tokens(values, mask)
        values = values.masked_fill(~mask[..., None], 0)
        expert_values = torch.stack([expert(values, mask) for expert in self.experts], 1)
        outputs = []
        for attention, gate in zip(self.attention, self.gates):
            scores = attention(values).squeeze(-1).masked_fill(~mask, 0)
            total = scores.sum(-1, keepdim=True)
            regular = total.abs() >= self.epsilon
            weights = scores / torch.where(regular, total, torch.ones_like(total))
            uniform = mask.to(values.dtype) / mask.sum(-1, keepdim=True)
            weights = torch.where(regular, weights, uniform)
            pooled = (weights[..., None] * values).sum(1)
            outputs.append(
                torch.einsum("be,beld->bld", gate(pooled), expert_values).masked_fill(
                    ~mask[..., None], 0
                )
            )
        return tuple(outputs)


class _AdaptiveNorm(nn.Module):
    def __init__(self, hidden, epsilon):
        super().__init__()
        self.mean, self.scale = _MLP(1, hidden, 1), _MLP(1, hidden, 1)
        self.epsilon = epsilon

    def forward(self, values, probability):
        centered = values - values.mean(-1, keepdim=True)
        normalized = centered / (centered.square().mean(-1, keepdim=True) + self.epsilon).sqrt()
        condition = probability[:, None]
        return self.scale(condition) * normalized + self.mean(condition)


def _veto(probabilities, initial, low, high):
    """Paper Rules 2/3, using Fake probabilities and the fixed four-head majority."""
    fake_count = (probabilities > 0.5).sum(-1)
    majority = torch.where(fake_count == 2, initial > 0.5, fake_count > 2)
    maximum = probabilities.max(-1).values
    result = initial
    for probability in probabilities.unbind(-1):
        replace = (probability > high) & (probability > result)
        reconsider = (probability < low) & ((probability > 0.5) == majority)
        result = torch.where(replace, probability, result)
        result = torch.where(reconsider, (result + maximum) / 2, result)
    return result


@register_model("GAMED")
class GAMED(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        positive = (
            "routing_dim",
            "hidden_dim",
            "num_experts",
            "num_heads",
            "depth",
            "ffn_multiplier",
        )
        if any(type(p[key]) is not int or p[key] < 1 for key in positive):
            raise ValueError(
                "GAMED dimensions, expert counts and depths must be positive integers."
            )
        if p["routing_dim"] % p["num_heads"]:
            raise ValueError("GAMED routing_dim must be divisible by num_heads.")
        if (
            not isinstance(p["epsilon"], (int, float))
            or isinstance(p["epsilon"], bool)
            or not math.isfinite(p["epsilon"])
            or not 0 < p["epsilon"] < 0.5
            or not isinstance(p["irrelevant_std"], (int, float))
            or isinstance(p["irrelevant_std"], bool)
            or not math.isfinite(p["irrelevant_std"])
            or p["irrelevant_std"] <= 0
            or not isinstance(p["dropout"], (int, float))
            or isinstance(p["dropout"], bool)
            or not math.isfinite(p["dropout"])
            or not 0 <= p["dropout"] < 1
        ):
            raise ValueError("GAMED requires finite valid epsilon, irrelevant_std and dropout.")
        veto = p["veto"]
        if (
            not isinstance(veto, dict)
            or set(veto) != {"low_threshold", "high_threshold"}
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in veto.values()
            )
            or not 0 < veto["low_threshold"] < veto["high_threshold"] < 1
        ):
            raise ValueError("GAMED veto requires 0 < low_threshold < high_threshold < 1.")
        weights = p["loss_weights"]
        if (
            not isinstance(weights, dict)
            or set(weights) != set(_HEADS)
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
                for value in weights.values()
            )
        ):
            raise ValueError("GAMED requires positive finite loss weights for all five heads.")
        _validate_gamed_analyzers(p["text"], p["image"], p["pattern"])
        self.epsilon = p["epsilon"]
        self.low_threshold, self.high_threshold = veto["low_threshold"], veto["high_threshold"]
        self.loss_weights = dict(weights)
        self.text_encoder = material_encoder(config, "text_encoder", _GAMEDTextTokens, **p["text"])
        self.image_encoder = material_encoder(
            config, "image_encoder", _GAMEDImageTokens, **p["image"]
        )
        self.pattern_pixels = RGBEncoder(
            image_size=p["image"]["image_size"], minimum_size=0, center_crop=False
        )
        self.pattern_pixels.cache_features = False
        self.pattern = _GAMEDPattern(**p["pattern"])
        dimension, hidden = p["routing_dim"], p["hidden_dim"]
        if self.text_encoder.output_dim != dimension or self.image_encoder.output_dim != dimension:
            raise ValueError("GAMED routing_dim must match both text and MAE token widths.")
        self.pattern_projection = _MLP(self.pattern.output_dim, hidden, dimension)
        options = dict(
            dimension=dimension,
            experts=p["num_experts"],
            heads=p["num_heads"],
            multiplier=p["ffn_multiplier"],
            hidden=hidden,
            depth=p["depth"],
            dropout=p["dropout"],
            epsilon=self.epsilon,
        )
        self.image_mixture, self.text_mixture = (
            _Mixture(**options, tasks=2),
            _Mixture(**options, tasks=2),
        )
        self.refinement, self.final_mixture = (
            _Mixture(**options, tasks=2),
            _Mixture(**options, tasks=1),
        )
        self.coarse = nn.ModuleDict({name: _MLP(dimension, hidden, 1) for name in _HEADS[:-1]})
        self.classifier = _MLP(dimension, hidden, 1)
        self.adjustment = nn.ModuleDict(
            {name: _AdaptiveNorm(hidden, self.epsilon) for name in ("ip", "is", "t", "x")}
        )
        self.irrelevant = nn.Parameter(torch.empty(1, dimension))
        nn.init.normal_(self.irrelevant, std=p["irrelevant_std"])

    def forward(self, batch):
        text_tokens = self.text_encoder(batch["text"])
        text_mask = self.text_encoder.token_mask(batch["text"]).to(text_tokens.device)
        pixels = self.image_encoder.pixels(batch["image_path"])
        image_tokens = self.image_encoder(batch["image_path"], pixels=pixels)
        image_mask = torch.ones(
            image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device
        )
        images, texts = (
            self.image_mixture(image_tokens, image_mask),
            self.text_mixture(text_tokens, text_mask),
        )
        combined_mask = torch.cat((image_mask, text_mask), 1)
        fused = self.refinement(torch.cat((images[1], texts[1]), 1), combined_mask)
        views = {
            "ip": self.pattern_projection(self.pattern(self.pattern_pixels(batch["image_path"]))),
            "is": _masked_mean(images[0], image_mask),
            "t": _masked_mean(texts[0], text_mask),
            "mm": _masked_mean(fused[0], combined_mask),
        }
        logits = {name: self.coarse[name](view).squeeze(-1) for name, view in views.items()}
        adjusted = [
            self.adjustment[name](views[name], logits[name].sigmoid()) for name in ("ip", "is", "t")
        ]
        unrelated = self.adjustment["x"](
            self.irrelevant.expand(len(text_tokens), -1), 1 - logits["mm"].sigmoid()
        )
        final_tokens = torch.stack((*adjusted, _masked_mean(fused[1], combined_mask), unrelated), 1)
        final_mask = torch.ones(
            final_tokens.shape[:2], dtype=torch.bool, device=final_tokens.device
        )
        mixed = self.final_mixture(final_tokens, final_mask)[0]
        logits["mix"] = self.classifier(_masked_mean(mixed, final_mask)).squeeze(-1)
        return logits

    def calculate_loss(self, batch):
        logits = self(batch)
        targets = batch["labels"].to(logits["mix"])
        return {
            name: self.loss_weights[name] * F.binary_cross_entropy_with_logits(output, targets)
            for name, output in logits.items()
        }

    @torch.no_grad()
    def predict(self, batch):
        modes = {module: module.training for module in self.modules()}
        try:
            self.eval()
            logits = self(batch)
            scores = torch.stack([logits[name] for name in _HEADS], -1)
            if scores.dtype != torch.float64:
                scores = scores.float()
            probabilities = scores.sigmoid()
            probability = _veto(
                probabilities[:, :4], probabilities[:, 4], self.low_threshold, self.high_threshold
            )
            score = torch.logit(probability.clamp(self.epsilon, 1 - self.epsilon))
            return torch.stack((torch.zeros_like(score), score), -1)
        finally:
            for module, mode in modes.items():
                module.training = mode
