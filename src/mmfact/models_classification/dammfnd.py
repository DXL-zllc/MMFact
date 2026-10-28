"""DAMMFND-DomainAdapted/Corrected: disentangled domain-aware view decisions."""

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder, material_mask
from mmfact.models_classification._domains import DomainTargets
from mmfact.models_classification._encoders import (
    BertEncoder,
    CLIPImageEncoder,
    CLIPTextEncoder,
    MAETokenEncoder,
)
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _mlp(inputs, hidden, outputs, dropout):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        SingletonBatchNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, outputs),
    )


def _validate(parameters):
    for key in (
        "num_channels",
        "cnn_channels",
        "multimodal_hidden",
        "multimodal_gate_hidden",
        "mask_hidden",
        "classifier_hidden",
        "domain_rank",
        "num_heads",
        "ffn_dim",
    ):
        if type(parameters[key]) is not int or parameters[key] < 1:
            raise ValueError(f"DAMMFND {key} must be a positive integer.")
    kernels = parameters["kernels"]
    if (
        not isinstance(kernels, (list, tuple))
        or not kernels
        or any(type(kernel) is not int or kernel < 1 for kernel in kernels)
        or len(set(kernels)) != len(kernels)
    ):
        raise ValueError("DAMMFND kernels must be distinct positive integers.")
    dimension = len(kernels) * parameters["cnn_channels"]
    if dimension % parameters["num_heads"] or parameters["domain_rank"] >= dimension:
        raise ValueError("DAMMFND view dimension must fit heads and a strictly lower rank.")
    for key in (
        "dropout",
        "gate_dropout",
        "attention_dropout",
        "alpha_fnd",
        "alpha_domain",
        "alpha_semantic",
    ):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"DAMMFND {key} must be finite.")
        if not (0 <= value < 1 if "dropout" in key else value >= 0):
            raise ValueError(f"DAMMFND {key} is outside its valid range.")
    for key in ("text", "image", "clip_text", "clip_image"):
        if parameters[key].get("freeze") is not True:
            raise ValueError("DAMMFND uses frozen pretrained encoders.")
    text, image = parameters["text"], parameters["image"]
    if text.get("output") != "tokens" or text.get("mask_padding") is not False:
        raise ValueError("DAMMFND needs complete BERT tokens with retained padding hidden.")
    if type(text.get("max_length")) is not int or text["max_length"] < max(2, max(kernels)):
        raise ValueError("DAMMFND text length must accommodate every CNN kernel.")
    if image.get("augment") is not False:
        raise ValueError("DAMMFND MAE encoding must be deterministic without augmentation.")
    for key in ("image_size", "resize_size"):
        if type(image.get(key)) is not int or image[key] < 1:
            raise ValueError(f"DAMMFND image {key} must be a positive integer.")
    if image.get("center_crop") is not True or image.get("normalize") is not True:
        raise ValueError("DAMMFND MAE needs center crop and ImageNet normalization.")
    if (
        image["resize_size"] < image["image_size"]
        or type(image.get("minimum_size")) is not int
        or image["minimum_size"] != 0
    ):
        raise ValueError("DAMMFND crop must fit resize and must not discard small images.")
    for key in ("clip_text", "clip_image"):
        if parameters[key].get("normalize") is not True:
            raise ValueError("DAMMFND CLIP features must be L2-normalized.")
    if type(parameters["clip_image"].get("square_resize")) is not bool:
        raise ValueError("DAMMFND CLIP square_resize must be a boolean.")
    first, second = parameters["clip_text"], parameters["clip_image"]
    if any(first.get(key) != second.get(key) for key in ("model_name", "revision")):
        raise ValueError("DAMMFND CLIP towers must use the same joint checkpoint.")
    if type(first.get("max_length")) is not int or first["max_length"] < 2:
        raise ValueError("DAMMFND CLIP text length must be an integer >= 2.")


class _CNN(nn.Module):
    def __init__(self, dimension, kernels, channels):
        super().__init__()
        self.convolutions = nn.ModuleList(nn.Conv1d(dimension, channels, k) for k in kernels)

    def forward(self, values):
        return torch.cat([conv(values.transpose(1, 2)).amax(-1) for conv in self.convolutions], -1)


class _TokenAttention(nn.Module):
    def __init__(self, dimension, masked):
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


class _Channels(nn.Module):
    def __init__(self, experts, inputs, hidden, dropout):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(self.experts)),
            nn.Dropout(dropout),
        )

    def forward(self, values, condition):
        weights = self.gate(condition).softmax(-1)
        features = torch.stack([expert(values) for expert in self.experts], 1)
        return (weights.unsqueeze(-1) * features).sum(1)


class _Disentangle(nn.Module):
    def __init__(self, dimension, mask_hidden, classifier_hidden, domains, dropout):
        super().__init__()
        self.domain_mask = _mlp(dimension, mask_hidden, dimension, dropout)
        self.semantic_mask = _mlp(dimension, mask_hidden, dimension, dropout)
        self.domain_classifier = _mlp(dimension, classifier_hidden, domains, dropout)
        self.semantic_classifier = _mlp(dimension, classifier_hidden, domains, dropout)

    def forward(self, values):
        domain = values * self.domain_mask(values).sigmoid()
        semantic = values * self.semantic_mask(values).sigmoid()
        return domain, semantic, self.domain_classifier(domain), self.semantic_classifier(semantic)


class _Decision(nn.Module):
    def __init__(self, dimension, heads, ffn_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, dimension)
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dimension, 3)

    def forward(self, semantic, filtered):
        query = semantic.mean(1, keepdim=True) + filtered.sum(1, keepdim=True)
        attended = self.attention(query, semantic, semantic, need_weights=False)[0]
        values = self.norm1(query + self.dropout(attended))
        values = self.norm2(values + self.dropout(self.ffn(values)))
        return self.output(values[:, 0])


def _mixture_logit(view_logits, decision_logits):
    """Log odds of a probability mixture, without saturating sigmoid/clamping."""
    log_weights = decision_logits.log_softmax(-1)
    positive = torch.logsumexp(log_weights + F.logsigmoid(view_logits), -1)
    negative = torch.logsumexp(log_weights + F.logsigmoid(-view_logits), -1)
    return positive - negative


@register_model("DAMMFND")
class DAMMFND(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate(p)
        self.domains = DomainTargets(**p["domains"])
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
            raise ValueError("DAMMFND MAE token sequence must accommodate all CNN kernels.")
        if self.clip_text_encoder.output_dim != self.clip_image_encoder.output_dim:
            raise ValueError("DAMMFND CLIP towers must have equal projection dimensions.")
        dimension = len(p["kernels"]) * p["cnn_channels"]
        self.text_attention = _TokenAttention(self.text_encoder.output_dim, True)
        self.image_attention = _TokenAttention(self.image_encoder.output_dim, False)
        self.channels = nn.ModuleDict()
        for name, encoder in (("text", self.text_encoder), ("image", self.image_encoder)):
            self.channels[name] = _Channels(
                [
                    _CNN(encoder.output_dim, p["kernels"], p["cnn_channels"])
                    for _ in range(p["num_channels"])
                ],
                encoder.output_dim,
                encoder.output_dim,
                p["gate_dropout"],
            )
        self.multimodal_projection = _mlp(
            self.clip_text_encoder.output_dim * 2,
            p["multimodal_hidden"],
            dimension,
            p["gate_dropout"],
        )
        self.channels["multimodal"] = _Channels(
            [
                nn.Sequential(
                    nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
                )
                for _ in range(p["num_channels"])
            ],
            dimension,
            p["multimodal_gate_hidden"],
            p["gate_dropout"],
        )
        self.disentangle = nn.ModuleDict(
            {
                name: _Disentangle(
                    dimension,
                    p["mask_hidden"],
                    p["classifier_hidden"],
                    p["domains"]["num_domains"],
                    p["dropout"],
                )
                for name in self.channels
            }
        )
        self.filters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dimension, p["domain_rank"], bias=False),
                    nn.Linear(p["domain_rank"], dimension),
                    nn.Sigmoid(),
                )
                for name in self.channels
            }
        )
        self.classifiers = nn.ModuleDict(
            {
                name: _mlp(dimension * 2, p["classifier_hidden"], 1, p["dropout"])
                for name in self.channels
            }
        )
        self.decision = _Decision(dimension, p["num_heads"], p["ffn_dim"], p["attention_dropout"])
        self.alpha_fnd, self.alpha_domain, self.alpha_semantic = (
            p["alpha_fnd"],
            p["alpha_domain"],
            p["alpha_semantic"],
        )

    def prepare(self, train_dataset):
        self.domains.prepare(train_dataset)

    def forward(self, batch):
        text = self.text_encoder(batch["text"])
        image = self.image_encoder(batch["image_path"])
        mask = material_mask(self.text_encoder, batch["text"]).to(text.device)
        clip_text = F.normalize(self.clip_text_encoder(batch["text"]), dim=-1)
        clip_image = F.normalize(self.clip_image_encoder(batch["image_path"]), dim=-1)
        correlation = (clip_text * clip_image).sum(-1, keepdim=True)
        multimodal = correlation * self.multimodal_projection(
            torch.cat((clip_image, clip_text), -1)
        )
        features = {
            "text": self.channels["text"](text, self.text_attention(text, mask)),
            "image": self.channels["image"](image, self.image_attention(image)),
            "multimodal": self.channels["multimodal"](multimodal, multimodal),
        }
        domain, semantic, domain_logits, semantic_logits = zip(
            *(self.disentangle[name](values) for name, values in features.items())
        )
        global_semantic = torch.stack(semantic, 1).sum(1)
        filtered = [
            self.filters[name](global_semantic) * values for name, values in zip(features, domain)
        ]
        view_logits = torch.cat(
            [
                self.classifiers[name](torch.cat((sem, dom), -1))
                for name, sem, dom in zip(features, semantic, filtered)
            ],
            -1,
        )
        decisions = self.decision(torch.stack(semantic, 1), torch.stack(filtered, 1))
        return {
            "final": _mixture_logit(view_logits, decisions),
            "views": view_logits,
            "domain": torch.stack(domain_logits, 1),
            "semantic": torch.stack(semantic_logits, 1),
            "weights": decisions.softmax(-1),
        }

    def calculate_loss(self, batch):
        outputs = self(batch)
        labels = batch["labels"].to(outputs["final"].dtype)
        targets = self.domains.targets(batch["sample_id"], device=outputs["final"].device)
        losses = {"classification": F.binary_cross_entropy_with_logits(outputs["final"], labels)}
        for index, name in enumerate(self.channels):
            losses[f"view_{name}"] = self.alpha_fnd * F.binary_cross_entropy_with_logits(
                outputs["views"][:, index], labels
            )
            losses[f"domain_{name}"] = (
                self.alpha_domain
                * F.binary_cross_entropy_with_logits(
                    outputs["domain"][:, index], targets, reduction="none"
                )
                .sum(-1)
                .mean()
            )
            semantic = outputs["semantic"][:, index]
            losses[f"semantic_{name}"] = self.alpha_semantic * F.kl_div(
                semantic.log_softmax(-1),
                torch.full_like(semantic, 1 / semantic.shape[-1]),
                reduction="batchmean",
            )
        return losses

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            logits = self(batch)["final"]
            return torch.stack((torch.zeros_like(logits), logits), -1)
        finally:
            self.train(was_training)
