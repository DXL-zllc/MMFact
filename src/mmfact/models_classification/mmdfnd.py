"""MMDFND-DomainAdapted/Corrected: domain PLE, stepwise pivots and AdaIN."""

import math
from copy import deepcopy
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder, material_mask
from mmfact.models_classification._domains import SingleDomainRouter
from mmfact.models_classification._encoders import (
    BertEncoder,
    CLIPImageEncoder,
    CLIPTextEncoder,
    MAETokenEncoder,
)
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _validate(p):
    for key in (
        "num_specific",
        "num_shared",
        "cnn_channels",
        "domain_dim",
        "gate_hidden",
        "multimodal_hidden",
        "multimodal_gate_hidden",
        "classifier_hidden",
        "pivot_tokens",
        "pivot_rounds",
        "pivot_heads",
        "pivot_ffn_multiplier",
        "final_heads",
        "adain_hidden",
    ):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"MMDFND {key} must be a positive integer.")
    for key in ("kernels", "final_gate_hidden"):
        values = p[key]
        if (
            not isinstance(values, (list, tuple))
            or not values
            or any(type(value) is not int or value < 1 for value in values)
            or (key == "kernels" and len(set(values)) != len(values))
        ):
            raise ValueError(f"MMDFND {key} must contain positive integers.")
    dimension = len(p["kernels"]) * p["cnn_channels"]
    if dimension % p["pivot_heads"] or dimension % p["final_heads"]:
        raise ValueError("MMDFND expert dimension must be divisible by both head counts.")
    for key in (
        "gate_dropout",
        "projection_dropout",
        "classifier_dropout",
        "pivot_dropout",
        "pivot_attention_dropout",
        "pivot_readout_dropout",
        "alpha",
        "adain_epsilon",
    ):
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"MMDFND {key} must be finite.")
        if not (0 <= value < 1 if "dropout" in key else value > 0):
            raise ValueError(f"MMDFND {key} is outside its valid range.")
    for key in ("text", "image", "clip_text", "clip_image"):
        if p[key].get("freeze") is not True:
            raise ValueError("MMDFND requires frozen pretrained encoders.")
    text, image = p["text"], p["image"]
    if text.get("output") != "tokens" or text.get("mask_padding") is not False:
        raise ValueError("MMDFND BERT tokens must retain padding hidden states.")
    if type(text.get("max_length")) is not int or text["max_length"] < max(2, max(p["kernels"])):
        raise ValueError("MMDFND BERT length must fit every CNN kernel.")
    if image.get("augment") is not False:
        raise ValueError("MMDFND MAE must use deterministic unaugmented pixels.")
    if image.get("center_crop") is not True or image.get("normalize") is not True:
        raise ValueError("MMDFND MAE requires center crop and ImageNet normalization.")
    if any(
        type(image.get(key)) is not int or image[key] < 1 for key in ("image_size", "resize_size")
    ):
        raise ValueError("MMDFND MAE image/resize size must be positive integers.")
    if (
        image["resize_size"] < image["image_size"]
        or type(image.get("minimum_size")) is not int
        or image["minimum_size"] != 0
    ):
        raise ValueError("MMDFND MAE crop must fit resize and must not discard small images.")
    first, second = p["clip_text"], p["clip_image"]
    if any(first.get(key) != second.get(key) for key in ("model_name", "revision")):
        raise ValueError("MMDFND CLIP towers must use the same joint checkpoint.")
    if type(first.get("max_length")) is not int or first["max_length"] < 2:
        raise ValueError("MMDFND CLIP length must be an integer >= 2.")
    if type(second.get("square_resize")) is not bool:
        raise ValueError("MMDFND CLIP square_resize must be boolean.")
    for values in (first, second):
        if values.get("normalize") is not True or values.get("output", "pooled") != "pooled":
            raise ValueError("MMDFND requires normalized pooled CLIP projections.")


def _mlp(inputs, hidden, outputs, dropout):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        SingletonBatchNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, outputs),
    )


def _gate(inputs, hidden, outputs, dropout):
    layers = []
    for size in hidden:
        layers.extend((nn.Linear(inputs, size), nn.SiLU()))
        inputs = size
    return nn.Sequential(*layers, nn.Linear(inputs, outputs), nn.Dropout(dropout))


def _mapping(inputs, dimension, dropout):
    return nn.Sequential(
        nn.Linear(inputs, dimension),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(dimension, dimension),
        nn.SiLU(),
        nn.Dropout(dropout),
    )


class _CNN(nn.Module):
    def __init__(self, inputs, kernels, channels):
        super().__init__()
        self.convolutions = nn.ModuleList(nn.Conv1d(inputs, channels, kernel) for kernel in kernels)

    def forward(self, values):
        return torch.cat(
            [conv(values.transpose(1, 2)).max(-1).values for conv in self.convolutions], -1
        )


class _TokenAttention(nn.Module):
    def __init__(self, dimension, masked=False):
        super().__init__()
        self.masked = masked
        self.scores = (
            nn.Linear(dimension, 1)
            if masked
            else nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, 1))
        )

    def forward(self, values, mask=None):
        scores = self.scores(values).squeeze(-1)
        if self.masked:
            scores = scores.masked_fill(~mask.bool(), -torch.inf).softmax(-1)
        return (scores.unsqueeze(-1) * values).sum(1)


class _DPLE(nn.Module):
    """One softmax across specific/shared experts; neither part is renormalized."""

    def __init__(
        self, expert, domains, specific, shared, inputs, hidden, dropout, domain_inputs=False
    ):
        super().__init__()
        self.specific = nn.ModuleList(
            nn.ModuleList(expert() for _ in range(specific)) for _ in range(domains)
        )
        self.shared = nn.ModuleList(expert() for _ in range(shared))
        self.gates = nn.ModuleList(
            _gate(inputs, hidden, specific + shared, dropout) for _ in range(domains)
        )
        self.domain_inputs = domain_inputs

    def forward(self, values, condition):
        common = None
        if not self.domain_inputs:
            common = torch.stack([expert(values) for expert in self.shared], 1)
        specific, shared = [], []
        for index, (experts, gate) in enumerate(zip(self.specific, self.gates)):
            inputs = values[:, index] if self.domain_inputs else values
            context = condition[:, index] if self.domain_inputs else condition
            weights = gate(context).softmax(-1).unsqueeze(-1)
            features = torch.stack([expert(inputs) for expert in experts], 1)
            shared_features = (
                common
                if common is not None
                else torch.stack([expert(inputs) for expert in self.shared], 1)
            )
            specific.append((weights[:, : len(experts)] * features).sum(1))
            shared.append((weights[:, len(experts) :] * shared_features).sum(1))
        return torch.stack(specific, 1), torch.stack(shared, 1)


class _Attention(nn.Module):
    def __init__(self, dimension, heads, dropout=0, std=None, qkv_bias=True, output_bias=False):
        super().__init__()
        self.heads, self.head_dim = heads, dimension // heads
        self.query = nn.Linear(dimension, dimension, bias=qkv_bias)
        self.key = nn.Linear(dimension, dimension, bias=qkv_bias)
        self.value = nn.Linear(dimension, dimension, bias=qkv_bias)
        self.output = nn.Linear(dimension, dimension, bias=output_bias)
        self.dropout = nn.Dropout(dropout)
        if std is not None:
            for projection in (self.query, self.key, self.value, self.output):
                nn.init.normal_(projection.weight, std=std)

    def forward(self, values):
        projected = [
            self.dropout(layer(values))
            .reshape(len(values), -1, self.heads, self.head_dim)
            .transpose(1, 2)
            for layer in (self.query, self.key, self.value)
        ]
        query, key, value = projected
        weights = (query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(-1)
        attended = (weights @ value).transpose(1, 2).reshape_as(values)
        return self.dropout(self.output(attended))


class _Transformer(nn.Module):
    def __init__(self, dimension, heads, multiplier, dropout, attention_dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.attention = _Attention(dimension, heads, attention_dropout, std=0.02)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * multiplier),
            nn.ReLU(),
            nn.Linear(dimension * multiplier, dimension),
        )
        for layer in (self.ffn[0], self.ffn[-1]):
            nn.init.normal_(layer.weight, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        values = values + self.dropout(self.attention(self.norm1(values)))
        return values + self.dropout(self.ffn(self.norm2(values)))


class _Pivot(nn.Module):
    def __init__(self, dimension, p):
        super().__init__()
        self.sequences = nn.ModuleList(
            nn.ModuleList(
                _mapping(dimension, dimension, p["pivot_dropout"]) for _ in range(p["pivot_tokens"])
            )
            for _ in range(3)
        )
        self.steps = nn.ModuleList(
            _Transformer(
                dimension,
                p["pivot_heads"],
                p["pivot_ffn_multiplier"],
                p["pivot_dropout"],
                p["pivot_attention_dropout"],
            )
            for _ in range(p["pivot_rounds"] * 3)
        )
        self.readout = nn.Sequential(
            nn.Dropout(p["pivot_readout_dropout"]),
            _mapping(dimension * p["pivot_tokens"], dimension, p["pivot_readout_dropout"]),
        )

    def forward(self, text, image, multimodal):
        sequences = [
            torch.stack([mapping(values) for mapping in mappings], 1)
            for values, mappings in zip((text, image, multimodal), self.sequences)
        ]
        pivot = torch.stack(sequences).mean(0)
        count = pivot.shape[1]
        for index, step in enumerate(self.steps):
            view = (1, 2, 0)[index % 3]
            output = step(torch.cat((pivot, sequences[view]), 1))
            pivot = (pivot + output[:, :count]) / 2
            sequences[view] = output[:, count:]
        return self.readout(pivot.flatten(1))


class _AdaIN(nn.Module):
    def __init__(self, dimension, hidden, epsilon):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.sigma = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.epsilon = epsilon

    def forward(self, values):
        centered = values - values.mean(-1, keepdim=True)
        scale = ((centered.square().sum(-1, keepdim=True) + self.epsilon) / values.shape[-1]).sqrt()
        return self.sigma(values) * centered / scale + self.mu(values)


class _FinalExpert(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.attention = _Attention(dimension, heads, qkv_bias=False, output_bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Linear(dimension * 4, dimension)
        )

    def forward(self, values):
        values = values + self.attention(self.norm1(values))
        return (values + self.ffn(self.norm2(values)))[:, 0]


def _domain_loss(logits, labels, domains):
    losses = []
    for index in range(logits.shape[1]):
        selected = domains == index
        losses.append(
            F.binary_cross_entropy_with_logits(logits[selected, index], labels[selected].float())
            if selected.any()
            else logits[:, index].sum() * 0
        )
    return torch.stack(losses).sum()


@register_model("MMDFND")
class MMDFND(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate(p)
        self._settings = deepcopy(p)
        self.domains = SingleDomainRouter(**p["domains"])
        self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **p["text"])
        self.image_encoder = material_encoder(
            config, "image_encoder", MAETokenEncoder, **p["image"]
        )
        self.clip_text_encoder = material_encoder(
            config, "clip_text_encoder", CLIPTextEncoder, **p["clip_text"]
        )
        self.clip_image_encoder = material_encoder(
            config, "clip_image_encoder", CLIPImageEncoder, **p["clip_image"]
        )
        if self.image_encoder.sequence_length < max(p["kernels"]):
            raise ValueError("MMDFND MAE sequence must fit every CNN kernel.")
        if self.clip_text_encoder.output_dim != self.clip_image_encoder.output_dim:
            raise ValueError("MMDFND joint CLIP projections must have equal dimensions.")
        dimension = len(p["kernels"]) * p["cnn_channels"]
        domains = p["domains"]["num_domains"]
        self.domain_embedding = nn.Embedding(domains, p["domain_dim"])
        self.text_attention = _TokenAttention(self.text_encoder.output_dim, masked=True)
        self.image_attention = _TokenAttention(self.image_encoder.output_dim)
        self.views = nn.ModuleDict()
        for name, encoder in (("text", self.text_encoder), ("image", self.image_encoder)):
            self.views[name] = _DPLE(
                lambda: _CNN(encoder.output_dim, p["kernels"], p["cnn_channels"]),
                domains,
                p["num_specific"],
                p["num_shared"],
                encoder.output_dim + p["domain_dim"],
                [p["gate_hidden"]],
                p["gate_dropout"],
            )
        self.multimodal_projection = _mlp(
            self.clip_text_encoder.output_dim * 2,
            p["multimodal_hidden"],
            dimension,
            p["projection_dropout"],
        )
        self.views["multimodal"] = _DPLE(
            lambda: nn.Sequential(
                nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
            ),
            domains,
            p["num_specific"],
            p["num_shared"],
            dimension + p["domain_dim"],
            [p["multimodal_gate_hidden"]],
            p["gate_dropout"],
        )
        self.view_classifiers = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    _mlp(dimension, p["classifier_hidden"], 1, p["classifier_dropout"])
                    for _ in range(domains)
                )
                for name in self.views
            }
        )
        self.domain_pivots = nn.ModuleList(_Pivot(dimension, p) for _ in range(domains))
        self.shared_pivot = _Pivot(dimension, p)
        self.domain_adain = _AdaIN(dimension, p["adain_hidden"], p["adain_epsilon"])
        self.shared_adain = _AdaIN(dimension, p["adain_hidden"], p["adain_epsilon"])
        self.final_attention = _TokenAttention(dimension)
        self.final = _DPLE(
            lambda: _FinalExpert(dimension, p["final_heads"]),
            domains,
            p["num_specific"],
            p["num_shared"],
            dimension + p["domain_dim"],
            p["final_gate_hidden"],
            p["gate_dropout"],
            domain_inputs=True,
        )
        self.final_classifiers = nn.ModuleList(
            _mlp(dimension, p["classifier_hidden"], 1, p["classifier_dropout"])
            for _ in range(domains)
        )
        self.alpha = p["alpha"]

    def prepare(self, train_dataset):
        self.domains.prepare(train_dataset)

    @staticmethod
    def _classify(features, classifiers):
        return torch.cat([head(features[:, index]) for index, head in enumerate(classifiers)], -1)

    def _outputs(self, batch):
        domains = self.domains.route(batch, device=self.domain_embedding.weight.device)
        embedding = self.domain_embedding(domains)
        text = self.text_encoder(batch["text"])
        image = self.image_encoder(batch["image_path"])
        mask = material_mask(self.text_encoder, batch["text"]).to(text.device)
        clip_text = F.normalize(self.clip_text_encoder(batch["text"]), dim=-1)
        clip_image = F.normalize(self.clip_image_encoder(batch["image_path"]), dim=-1)
        multimodal = (clip_text * clip_image).sum(-1, keepdim=True) * self.multimodal_projection(
            torch.cat((clip_image, clip_text), -1)
        )
        inputs = (text, image, multimodal)
        conditions = (self.text_attention(text, mask), self.image_attention(image), multimodal)
        parts = [
            self.views[name](values, torch.cat((embedding, condition), -1))
            for name, values, condition in zip(self.views, inputs, conditions)
        ]
        view_logits = {
            name: self._classify(specific + shared, self.view_classifiers[name])
            for name, (specific, shared) in zip(self.views, parts)
        }
        rows = torch.arange(len(domains), device=domains.device)
        shared = self.shared_pivot(*(values[rows, domains] for _, values in parts))
        specific = torch.stack(
            [
                pivot(*(values[:, index] for values, _ in parts))
                for index, pivot in enumerate(self.domain_pivots)
            ],
            1,
        )
        knowledge = torch.stack(
            (self.domain_adain(specific), self.shared_adain(shared)[:, None].expand_as(specific)), 2
        )
        pooled = self.final_attention(knowledge.flatten(0, 1)).reshape_as(specific)
        condition = torch.cat((embedding[:, None].expand(-1, specific.shape[1], -1), pooled), -1)
        first, second = self.final(knowledge, condition)
        return self._classify(first + second, self.final_classifiers), view_logits, domains

    def forward(self, batch):
        logits, _, domains = self._outputs(batch)
        selected = logits.gather(1, domains[:, None]).squeeze(-1)
        return torch.stack((torch.zeros_like(selected), selected), -1)

    def calculate_loss(self, batch):
        logits, views, domains = self._outputs(batch)
        return {
            "final": _domain_loss(logits, batch["labels"], domains),
            **{
                name: self.alpha * _domain_loss(values, batch["labels"], domains)
                for name, values in views.items()
            },
        }

    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            return self(batch)
        finally:
            for module, training in modes:
                module.training = training

    def get_extra_state(self):
        return {"version": 1, "parameters": deepcopy(self._settings)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("parameters") != self._settings
        ):
            raise ValueError("MMDFND checkpoint parameters do not match the configured contract.")
