"""MMRGV: two-stage alignment and three independently gated rationale views."""

import math
from copy import deepcopy

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import BatchEncoding

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.materials.encoders import material_encoder
from mmfact.materials.resources import material_resources
from mmfact.models_classification._encoders import BertEncoder, MAETokenEncoder, RGBEncoder
from mmfact.models_classification._mmrgv_resources import VIEWS, RationaleStore
from mmfact.registry import register_model


class _FinalTextEncoder(BertEncoder):
    def __init__(self, **options):
        if options.get("freeze") is not False or options.get("output") != "tokens":
            raise ValueError("MMRGV text encoders require trainable token outputs.")
        super().__init__(**options)
        self.network.requires_grad_(False)
        self.network.encoder.layer[-1].requires_grad_(True)
        self.train(self.training)

    def _tokenize(self, texts):
        # Padding to max_length makes every row independent, so per-item rows are cached
        # (identical to batch tokenization; avoids re-tokenizing every step and in token_mask).
        cache = self.__dict__.setdefault("_token_cache", {})
        rows = []
        for item in texts:
            row = cache.get(item)
            if row is None:
                row = dict(self._tokenize_batch([item]))
                cache[item] = row
            rows.append(row)
        return BatchEncoding({key: torch.cat([row[key] for row in rows]) for key in rows[0]})

    def _tokenize_batch(self, texts):
        if texts and isinstance(texts[0], tuple):
            return self.tokenizer(
                [pair[0] for pair in texts],
                text_pair=[pair[1] for pair in texts],
                padding="max_length",
                truncation="longest_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
        return super()._tokenize(texts)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "network"):
            self.network.eval()
            self.network.encoder.layer[-1].train(mode)
        return self


class _CachedRGB(RGBEncoder):
    """RGBEncoder with identical arithmetic that decodes/resizes each image file only once."""

    def _encode(self, paths):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import (
            center_crop,
            normalize,
            pil_to_tensor,
            resize,
        )

        cache = self.__dict__.setdefault("_pixel_cache", {})
        pixels = []
        for path in paths:
            if path not in cache:
                with Image.open(path) as image:
                    if min(image.size) < self.minimum_size:
                        cache[path] = None
                    else:
                        image = image.convert("RGB")
                        if self.center_crop:
                            image = center_crop(
                                resize(image, self.resize_size, InterpolationMode.BILINEAR),
                                self.image_size,
                            )
                        else:
                            image = image.resize(
                                (self.image_size, self.image_size), Image.Resampling.BILINEAR
                            )
                        cache[path] = pil_to_tensor(image)
            raw = cache[path]
            if raw is None:
                value = torch.zeros(3, self.image_size, self.image_size)
            else:
                value = raw.float().div(255)
            if self.normalize:
                value = normalize(value, self.mean, self.std)
            pixels.append(value)
        return torch.stack(pixels).to(self.device)


class _FinalImageEncoder(MAETokenEncoder):
    def __init__(self, **options):
        if options.get("freeze") is not False:
            raise ValueError("MMRGV MAE requires final-block fine-tuning.")
        if options.get("output") != "patches":
            raise ValueError("MMRGV image attention requires ordered image patches without CLS.")
        super().__init__(**options)
        self.pixels.__class__ = _CachedRGB
        self.network.requires_grad_(False)
        self.network.encoder.layer[-1].requires_grad_(True)
        self.network.layernorm.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "network"):
            self.network.eval()
            self.network.encoder.layer[-1].train(mode)
            self.network.layernorm.train(mode)
        return self


def _mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def _mlp(inputs, hidden, outputs, dropout):
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, outputs)
    )


def _focal(logits, targets, alpha, gamma):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_correct = torch.exp(-ce)
    weights = alpha * targets + (1 - alpha) * (1 - targets)
    return (weights * (1 - probability_correct).pow(gamma) * ce).mean()


class _Attention(nn.Module):
    """Eq.4: single-head projected attention, with independent query/key/value weights."""

    def __init__(self, dimension):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(dimension, dimension, bias=False) for _ in range(3))

    def forward(self, query, key, key_mask):
        if (
            query.ndim != 3
            or key.ndim != 3
            or query.shape[0] != key.shape[0]
            or query.shape[-1] != key.shape[-1]
            or key_mask.shape != key.shape[:2]
            or key_mask.dtype != torch.bool
            or not key_mask.any(-1).all()
        ):
            raise ValueError(
                "MMRGV attention requires compatible sequences and nonempty valid keys."
            )
        scores = self.q(query) @ self.k(key).transpose(1, 2) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~key_mask[:, None, :], -torch.inf)
        return scores.softmax(-1) @ self.v(key)


class _RationaleGate(nn.Module):
    def __init__(self, dimension, hidden, dropout):
        super().__init__()
        self.rationale_attention, self.content_attention = (
            _Attention(dimension),
            _Attention(dimension),
        )
        self.correctness = _mlp(dimension, hidden, 1, dropout)
        self.gate = _mlp(dimension, hidden, 1, dropout)

    def forward(self, content, rationale, content_mask, rationale_mask):
        updated_rationale = self.rationale_attention(rationale, content, content_mask)
        updated_content = self.content_attention(content, rationale, rationale_mask)
        pooled_content = _mean(updated_content, content_mask)
        correctness = self.correctness(pooled_content).squeeze(-1)
        gate = self.gate(pooled_content).sigmoid()
        return gate * _mean(updated_rationale, rationale_mask), correctness


class _ViewAggregation(nn.Module):
    def __init__(self, dimension, depth):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(depth, 4, dimension))
        nn.init.normal_(self.weights, std=dimension**-0.5)

    def forward(self, views):
        for weights in self.weights:
            attention = torch.einsum("bjd,id->bij", views, weights).softmax(-1)
            views = views + attention @ views
        return views


@register_model("MMRGV")
class MMRGV(AbstractDetector):
    """Independent reconstruction of Eq.1–16; missing supplement defaults are explicit."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for name in ("hidden_dim", "alignment_dim", "aggregation_layers"):
            if type(p[name]) is not int or p[name] < 1:
                raise ValueError(f"MMRGV {name} must be a positive integer.")
        for name in (
            "temperature",
            "correctness_weight",
            "judgment_weight",
            "dropout",
            "correctness_alpha",
            "classification_alpha",
            "correctness_gamma",
            "classification_gamma",
        ):
            value = p[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"MMRGV {name} must be finite.")
            if name in {"dropout", "correctness_alpha", "classification_alpha"}:
                valid = 0 <= value < 1 if name == "dropout" else 0 < value < 1
            else:
                valid = value >= 0 if name.endswith("gamma") else value > 0
            if not valid:
                raise ValueError(f"MMRGV {name} is outside its permitted range.")
        if p["classification_loss"] not in {"ce", "focal"}:
            raise ValueError("MMRGV classification_loss must be ce or focal.")
        self.parameters_spec = deepcopy(p)
        self.resources = material_resources(config, "mmrgv", RationaleStore, **p["resources"])
        self.content_encoder = material_encoder(
            config, "content_encoder", _FinalTextEncoder, **p["text"]
        )
        self.rationale_encoder = material_encoder(
            config, "rationale_encoder", _FinalTextEncoder, **p["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", _FinalImageEncoder, **p["image"]
        )
        dimension = self.content_encoder.output_dim
        self.image_projection = (
            nn.Identity()
            if self.image_encoder.output_dim == dimension
            else nn.Linear(self.image_encoder.output_dim, dimension)
        )
        hidden, dropout = p["hidden_dim"], p["dropout"]
        self.image_alignment = _mlp(dimension, hidden, p["alignment_dim"], dropout)
        self.rationale_alignment = _mlp(dimension, hidden, p["alignment_dim"], dropout)
        self.judgment = nn.ModuleDict({view: _mlp(dimension, hidden, 3, dropout) for view in VIEWS})
        self.gates = nn.ModuleDict(
            {view: _RationaleGate(dimension, hidden, dropout) for view in VIEWS}
        )
        self.news_attention = nn.Parameter(torch.empty(dimension))
        nn.init.normal_(self.news_attention, std=dimension**-0.5)
        self.aggregation = _ViewAggregation(dimension, p["aggregation_layers"])
        self.classifier = _mlp(4 * dimension, hidden, 1, dropout)
        self.stage = None
        self._trainable = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        self.set_training_stage(None)

    def prepare(self, train_dataset):
        for row in train_dataset:
            self.resources.lookup(row["text"], row["image_path"])

    def training_stages(self):
        return (TrainingStage("alignment"), TrainingStage("detection"))

    def set_training_stage(self, stage):
        if stage not in {None, "alignment", "detection"}:
            raise ValueError("Unknown MMRGV stage.")
        self.stage = stage
        alignment = (
            "rationale_encoder.",
            "image_encoder.",
            "image_projection.",
            "image_alignment.",
            "rationale_alignment.",
        )
        for name, parameter in self.named_parameters():
            enabled = name in self._trainable
            if stage == "alignment":
                enabled = enabled and name.startswith(alignment)
            elif stage in {None, "detection"}:
                enabled = enabled and not name.startswith(
                    ("image_alignment.", "rationale_alignment.")
                )
            parameter.requires_grad_(enabled)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "stage", None) == "alignment":
            self.content_encoder.eval()
        return self

    def _inputs(self, batch):
        texts, paths = batch["text"], batch["image_path"]
        if not texts or len(texts) != len(paths):
            raise ValueError("MMRGV requires one text and image per news item.")
        records = [self.resources.lookup(text, path) for text, path in zip(texts, paths)]
        rationales, masks, judgments = {}, {}, {}
        for view in VIEWS:
            values = [record["rationales"][view]["text"] for record in records]
            rationales[view] = self.rationale_encoder(values)
            masks[view] = self.rationale_encoder.token_mask(values)
            judgments[view] = torch.tensor(
                [record["rationales"][view]["judgment"] for record in records],
                device=rationales[view].device,
            )
        image = self.image_projection(self.image_encoder(paths))
        return texts, records, rationales, masks, judgments, image

    def _alignment_loss(self, batch):
        _, _, rationales, masks, judgments, image = self._inputs(batch)
        image = self.image_alignment(image.mean(1))
        rationale = self.rationale_alignment(_mean(rationales["id"], masks["id"]))
        scores = image @ rationale.T / self.parameters_spec["temperature"]
        targets = torch.arange(scores.shape[0], device=scores.device)
        correct = judgments["id"].eq(batch["labels"]).to(scores.dtype)
        return (correct * F.cross_entropy(scores, targets, reduction="none")).mean()

    def _detection(self, batch):
        texts, records, rationales, masks, judgments, image = self._inputs(batch)
        consistency = [(text, record["caption"]) for text, record in zip(texts, records)]
        content = {
            "td": self.content_encoder(texts),
            "itc": self.content_encoder(consistency),
            "id": image,
        }
        content_masks = {
            "td": self.content_encoder.token_mask(texts),
            "itc": self.content_encoder.token_mask(consistency),
            "id": torch.ones(image.shape[:2], dtype=torch.bool, device=image.device),
        }
        gated, correctness, judgment_logits = {}, {}, {}
        for view in VIEWS:
            gated[view], correctness[view] = self.gates[view](
                content[view], rationales[view], content_masks[view], masks[view]
            )
            judgment_logits[view] = self.judgment[view](_mean(rationales[view], masks[view]))
        weights = (content["td"] @ self.news_attention).masked_fill(
            ~content_masks["td"], -torch.inf
        )
        news = (weights.softmax(-1).unsqueeze(-1) * content["td"]).sum(1)
        views = torch.stack([news, *(gated[view] for view in VIEWS)], dim=1)
        logits = self.classifier(self.aggregation(views).flatten(1)).squeeze(-1)
        return logits, correctness, judgment_logits, judgments

    def calculate_loss(self, batch):
        if self.stage == "alignment":
            return {"alignment": self._alignment_loss(batch)}
        logits, correctness, judgment_logits, judgments = self._detection(batch)
        p, targets = self.parameters_spec, batch["labels"].to(logits.dtype)
        classification = (
            F.binary_cross_entropy_with_logits(logits, targets)
            if p["classification_loss"] == "ce"
            else _focal(logits, targets, p["classification_alpha"], p["classification_gamma"])
        )
        correctness_loss = torch.stack(
            [
                _focal(
                    correctness[view],
                    judgments[view].eq(batch["labels"]).to(logits.dtype),
                    p["correctness_alpha"],
                    p["correctness_gamma"],
                )
                for view in VIEWS
            ]
        ).mean()
        judgment_loss = torch.stack(
            [F.cross_entropy(judgment_logits[view], judgments[view]) for view in VIEWS]
        ).mean()
        return {
            "classification": classification,
            "correctness": p["correctness_weight"] * correctness_loss,
            "judgment": p["judgment_weight"] * judgment_loss,
        }

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            logits = self._detection(batch)[0]
            return torch.stack((torch.zeros_like(logits), logits), dim=-1)
        finally:
            self.train(was_training)

    def get_extra_state(self):
        return {"version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("MMRGV checkpoint parameters differ from this configuration.")
