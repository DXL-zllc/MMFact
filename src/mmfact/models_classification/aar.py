"""AAR: frozen CLIP features, opposing arguments and two supervised judges."""

import json
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.materials.resources import material_resources
from mmfact.models_classification._encoders import (
    CLIPImageEncoder,
    CLIPTextEncoder,
    _PretrainedEncoder,
)
from mmfact.models_classification._generated import GeneratedTextStore
from mmfact.registry import register_model


class _ArgumentEncoder(CLIPTextEncoder):
    """Joint pair encoding; the news encoder is the sole owner of frozen weights."""

    def __init__(self, source, lookup):
        _PretrainedEncoder.__init__(self)
        # A nonregistered reference avoids duplicate backbone state_dict paths.
        object.__setattr__(self, "network", source.network)
        for name in (
            "tokenizer",
            "output_dim",
            "sequence_length",
            "max_length",
            "mask_padding",
            "output",
            "normalize",
            "freeze",
            "_cls_pooling",
        ):
            setattr(self, name, getattr(source, name))
        self.lookup = lookup
        self.encoding_spec = deepcopy(source.encoding_spec)
        self.encoding_spec.update(
            input_format="json-pair:genuine,fake",
            joint_truncation="longest_first",
        )
        if self.max_length < self.tokenizer.num_special_tokens_to_add(pair=True) + 2:
            raise ValueError(
                "AAR argument length must retain both stances and pair special tokens."
            )
        source.register_load_state_dict_post_hook(self._clear_shared_cache)
        self.train(self.training)

    def _clear_shared_cache(self, _owner, incompatible_keys):
        self._clear_cache(self, incompatible_keys)

    def inputs_from_rows(self, rows):
        pairs = []
        for row in rows:
            record = self.lookup(row["text"], row["image_path"])
            pairs.append(
                json.dumps(
                    [record["genuine"], record["fake"]], ensure_ascii=False, separators=(",", ":")
                )
            )
        return pairs

    def _tokenize(self, texts):
        pairs = [json.loads(text) for text in texts]
        if any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(text, str) or not text.strip() for text in pair)
            for pair in pairs
        ):
            raise ValueError("AAR argument inputs must be nonempty [genuine, fake] JSON pairs.")
        return self.tokenizer(
            [pair[0] for pair in pairs],
            text_pair=[pair[1] for pair in pairs],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )


class _Attention(nn.Module):
    """Batch-first multihead attention with a single residual and LayerNorm."""

    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dimension // heads
        self.q = nn.Linear(dimension, dimension, bias=False)
        self.k = nn.Linear(dimension, dimension, bias=False)
        self.v = nn.Linear(dimension, dimension, bias=False)
        self.o = nn.Linear(dimension, dimension, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, query, key, value, key_mask=None, query_mask=None):
        if (
            query.ndim != 3
            or key.ndim != 3
            or value.shape != key.shape
            or query.shape[0] != key.shape[0]
        ):
            raise ValueError("AAR attention requires [batch, length, dimension] inputs.")
        batch, length, dimension = query.shape

        def split(values):
            return values.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(value))
        scores = q @ k.transpose(-2, -1) * self.head_dim**-0.5
        if key_mask is not None:
            if key_mask.shape != key.shape[:2] or not key_mask.any(-1).all():
                raise ValueError("AAR attention needs at least one valid key per news item.")
            scores = scores.masked_fill(~key_mask[:, None, None, :], -torch.inf)
        weights = self.attention_dropout(scores.softmax(-1))
        mixed = (weights @ v).transpose(1, 2).reshape(batch, length, dimension)
        output = self.norm(query + self.output_dropout(self.o(mixed)))
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(output.dtype)
        return output


def _mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def _judge(dimension, hidden, dropout):
    return nn.Sequential(
        nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
    )


@register_model("AAR")
class AAR(AbstractDetector):
    """Paper-led AAR with fixed generated resources and corrected attention axes."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for name in ("attention_heads", "classifier_hidden"):
            if not isinstance(p[name], int) or isinstance(p[name], bool) or p[name] < 1:
                raise ValueError(f"AAR {name} must be a positive integer.")
        for name in ("dropout", "fusion_weight", "reasoning_weight"):
            value = p[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or (not 0 <= value < 1 if name == "dropout" else value <= 0)
            ):
                raise ValueError(f"AAR {name} is outside its finite permitted range.")
        for name in ("text", "image"):
            encoder = p[name]
            for field in ("model_name", "revision"):
                value = encoder.get(field, "main" if field == "revision" else None)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"AAR {name}.{field} must be a nonempty string.")
            if encoder.get("freeze", True) is not True or encoder.get("output") != "tokens":
                raise ValueError("AAR requires frozen CLIP token encoders.")
            if encoder.get("normalize", False) is not False:
                raise ValueError("AAR consumes unnormalized projected CLIP tokens.")
        if p["text"].get("mask_padding", True) is not True:
            raise ValueError("AAR requires explicit text padding masks.")
        length = p["text"].get("max_length", 77)
        if not isinstance(length, int) or isinstance(length, bool) or length < 6:
            raise ValueError("AAR max_length must be an integer >= 6 for paired arguments.")
        if any(
            p["text"].get(name, "main") != p["image"].get(name, "main")
            for name in ("model_name", "revision")
        ):
            raise ValueError("AAR text and image must use the same CLIP checkpoint and revision.")
        if type(p["image"].get("square_resize", False)) is not bool:
            raise ValueError("AAR image square_resize must be boolean.")
        resources = p["resources"]
        if (
            not isinstance(resources.get("language"), str)
            or resources["language"] not in {"en", "zh"}
            or not isinstance(resources.get("manifest"), str)
            or not resources["manifest"].strip()
        ):
            raise ValueError("AAR resources require en/zh and a nonempty manifest path.")

        self.parameters_spec = deepcopy(p)
        self.arguments = material_resources(
            config, "aar", GeneratedTextStore, task="aar", **resources
        )
        self.text_encoder = material_encoder(config, "text_encoder", CLIPTextEncoder, **p["text"])
        self.image_encoder = material_encoder(
            config, "image_encoder", CLIPImageEncoder, **p["image"]
        )
        dimension = self.text_encoder.output_dim
        if dimension != self.image_encoder.output_dim or dimension % p["attention_heads"]:
            raise ValueError("AAR CLIP dimensions must match and divide evenly among heads.")
        self.argument_encoder = material_encoder(
            config, "argument_encoder", _ArgumentEncoder, self.text_encoder, self.arguments.lookup
        )
        heads, dropout = p["attention_heads"], p["dropout"]
        self.text_attention = _Attention(dimension, heads, dropout)
        self.image_attention = _Attention(dimension, heads, dropout)
        self.reasoning_attention = _Attention(dimension, heads, dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.w1 = nn.Linear(dimension, dimension, bias=False)
        self.w2 = nn.Linear(dimension, dimension, bias=False)
        self.fusion_judge = _judge(dimension, p["classifier_hidden"], dropout)
        self.reasoning_judge = _judge(dimension, p["classifier_hidden"], dropout)
        self.fusion_weight, self.reasoning_weight = p["fusion_weight"], p["reasoning_weight"]

    def _logits(self, batch):
        if not batch["text"] or len(batch["text"]) != len(batch["image_path"]):
            raise ValueError("AAR needs one text and image per news item in a nonempty batch.")
        text = self.text_encoder(batch["text"])
        text_mask = self.text_encoder.token_mask(batch["text"])
        image = self.image_encoder(batch["image_path"])
        text = self.text_attention(text, text, text, text_mask, text_mask)
        image = self.image_attention(image, image, image)
        fused = self.alpha * image.mean(1) + self.beta * _mean(text, text_mask)
        fusion_logits = self.fusion_judge(self.w2(fused))
        rows = [
            dict(text=text, image_path=path)
            for text, path in zip(batch["text"], batch["image_path"])
        ]
        pairs = self.argument_encoder.inputs_from_rows(rows)
        arguments = self.argument_encoder(pairs)
        argument_mask = self.argument_encoder.token_mask(pairs)
        news = self.w1(fused).unsqueeze(1)
        reasoning = self.reasoning_attention(arguments, news, news, query_mask=argument_mask)
        reasoning_logits = self.reasoning_judge(_mean(reasoning, argument_mask))
        return fusion_logits, reasoning_logits

    def forward(self, batch):
        return self._logits(batch)[1]

    def calculate_loss(self, batch):
        fusion, reasoning = self._logits(batch)
        return {
            "fusion": self.fusion_weight * F.cross_entropy(fusion, batch["labels"]),
            "reasoning": self.reasoning_weight * F.cross_entropy(reasoning, batch["labels"]),
        }

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)

    def get_extra_state(self):
        return {"format_version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError("AAR checkpoint parameters do not match this model configuration.")
