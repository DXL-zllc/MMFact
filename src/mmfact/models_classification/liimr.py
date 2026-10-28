"""LIIMR: fine-grained unimodal encoders and multiplicative class scores."""

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.data.features import file_digest, fingerprint
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder, _PretrainedEncoder
from mmfact.registry import register_model


def _sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _validate_protocol(protocol):
    if (
        not isinstance(protocol, dict)
        or protocol.get("encoder") != "liimr-vg-faster-rcnn-bottom-up"
        or protocol.get("regions") != 36
        or protocol.get("dimension") != 2048
        or protocol.get("backend") != "MILVLG/bottom-up-attention.pytorch"
        or not isinstance(protocol.get("backend_commit"), str)
        or len(protocol["backend_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in protocol["backend_commit"])
        or any(
            not _sha(protocol.get(key))
            for key in ("source_sha256", "adapter_sha256", "config_sha256", "weights_sha256")
        )
        or protocol.get("mode") != "caffe-roi_feats"
        or protocol.get("input") != "PIL-RGB-first-frame-lossless-PNG"
        or not isinstance(protocol.get("runtime"), dict)
        or not protocol["runtime"]
        or any(not isinstance(v, str) or not v for v in protocol["runtime"].values())
    ):
        raise ValueError("LIIMR requires a versioned VG bottom-up 36×2048 producer protocol.")
    fingerprint(protocol)
    return deepcopy(protocol)


class _Regions(_PretrainedEncoder):
    """Immutable region artifacts; checkpoints carry indices, not huge feature matrices."""

    input_kind = "image"
    packages = ("torch", "numpy")

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.network = nn.Identity()
        self.records, self.protocol = {}, None
        self.encoding_spec = {}
        self.output_dim = 2048
        self._set_frozen(True)

    def _set_protocol(self, protocol):
        protocol = _validate_protocol(protocol)
        if self.protocol is not None and protocol != self.protocol:
            raise ValueError("LIIMR region producer differs from the checkpoint.")
        self.protocol = protocol
        self.encoding_spec = {"encoder": "liimr-fixed-region-artifacts", "producer": protocol}

    def _load(self):
        manifest = self.root / "manifest.json"
        if not manifest.is_file():
            raise ValueError(
                "Missing LIIMR regions; download the required materials with --download-materials."
            )
        state = json.loads(manifest.read_text("utf-8"))
        if not isinstance(state, dict) or state.get("schema") != 1:
            raise ValueError("Invalid LIIMR region manifest.")
        protocol = _validate_protocol(state.get("producer"))
        records = self._validate_records(state.get("records"))
        if fingerprint(records) != state.get("records_sha256"):
            raise ValueError("LIIMR region manifest checksum is invalid.")
        if any(
            key in self.records and self.records[key] != value for key, value in records.items()
        ):
            raise ValueError("LIIMR region records conflict with the checkpoint.")
        self._set_protocol(protocol)
        self.records.update(records)

    @staticmethod
    def _validate_records(records):
        if not isinstance(records, dict) or any(
            not _sha(k) or not _sha(v) for k, v in records.items()
        ):
            raise ValueError("LIIMR region indices require image and feature SHA256.")
        return dict(records)

    def paths(self, images):
        resolved = []
        for image in images:
            digest = file_digest(image)
            if digest not in self.records:
                self._load()
            if digest not in self.records:
                raise ValueError("LIIMR has no region artifact for this image.")
            path = self.root / "features" / f"{digest}.npy"
            if not path.is_file() or file_digest(path) != self.records[digest]:
                raise ValueError("LIIMR region artifact is missing or its content changed.")
            resolved.append(str(path))
        return resolved

    def inputs_from_rows(self, rows):
        return self.paths([row["image_path"] for row in rows])

    def _encode(self, paths):
        values = []
        for path in paths:
            value = np.load(path, allow_pickle=False)
            if (
                value.shape != (36, 2048)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError("LIIMR region features must be finite float32 [36,2048].")
            values.append(torch.from_numpy(value.copy()))
        return torch.stack(values).to(self.device)

    def get_extra_state(self):
        return {
            "schema": 1,
            "protocol": deepcopy(self.protocol),
            "records": dict(self.records),
            "records_sha256": fingerprint(self.records),
        }

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("schema") != 1:
            raise ValueError("Incompatible LIIMR region checkpoint.")
        records = self._validate_records(state.get("records"))
        if fingerprint(records) != state.get("records_sha256") or (
            records and state.get("protocol") is None
        ):
            raise ValueError("LIIMR region checkpoint checksum is invalid.")
        if state.get("protocol") is not None:
            self._set_protocol(state["protocol"])
        self.records = records


class _TextEncoder(BertEncoder):
    def __init__(self, **options):
        super().__init__(**options)
        self.network.pooler = None


class _ImageLayer(nn.Module):
    """One post-LN BERT image layer, without extra FFN-intermediate dropout."""

    def __init__(self, dimension, heads, feedforward, dropout, activation="gelu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout, batch_first=True)
        bound = dimension**-0.5
        nn.init.uniform_(self.attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.attention.in_proj_bias, -bound, bound)
        nn.init.uniform_(self.attention.out_proj.bias, -bound, bound)
        self.attention_norm = nn.LayerNorm(dimension, eps=1e-12)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(feedforward, dimension),
        )
        self.output_norm = nn.LayerNorm(dimension, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, regions):
        attended = self.attention(regions, regions, regions, need_weights=False)[0]
        regions = self.attention_norm(regions + self.dropout(attended))
        return self.output_norm(regions + self.dropout(self.feedforward(regions)))


def _multiplicative_scores(text, image, beta):
    """Paper §4.4, two modalities; class weights retain their formula gradients."""
    text_log, image_log = F.log_softmax(text, -1), F.log_softmax(image, -1)
    # Binary complement is the other class; avoid catastrophic 1 - exp(logp).
    text_weight = (beta * text_log.flip(-1)).exp()
    image_weight = (beta * image_log.flip(-1)).exp()
    return image_weight * text_log + text_weight * image_log


def _pool_windows(features, mask, convolutions):
    values = []
    for conv in convolutions:
        width = conv.kernel_size[0]
        current = F.relu(conv(features.transpose(1, 2)))
        valid = mask.unfold(1, width, 1).all(-1)
        pooled = current.masked_fill(~valid[:, None], -torch.inf).amax(-1)
        values.append(torch.where(valid.any(-1, keepdim=True), pooled, torch.zeros_like(pooled)))
    return torch.cat(values, -1)


@register_model("LIIMR")
class LIIMR(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        if (
            any(
                type(p[key]) is not int or p[key] < 1
                for key in ("hidden_dim", "filters", "heads", "feedforward_dim")
            )
            or p["hidden_dim"] % p["heads"]
        ):
            raise ValueError("LIIMR dimensions must be positive and hidden_dim divisible by heads.")
        for key in ("beta", "dropout", "classifier_dropout"):
            if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or p[key] < 0:
                raise ValueError(f"LIIMR {key} must be finite and nonnegative.")
        if (
            p["dropout"] >= 1
            or p["classifier_dropout"] >= 1
            or p["text"]["freeze"] is not True
            or p["text"]["output"] != "tokens"
            or p["text"]["mask_padding"] is not True
            or type(p["text"]["max_length"]) is not int
            or p["text"]["max_length"] < 3
            or p["image_activation"] not in {"gelu", "relu"}
        ):
            raise ValueError("LIIMR requires frozen masked BERT tokens (length>=3) and dropout<1.")
        self.text_encoder = material_encoder(config, "text_encoder", _TextEncoder, **p["text"])
        self.image_encoder = material_encoder(config, "image_encoder", _Regions, p["regions"])
        dim, dropout = p["hidden_dim"], p["dropout"]
        self.convolutions = nn.ModuleList(
            nn.Conv1d(self.text_encoder.output_dim, p["filters"], width) for width in (1, 2, 3)
        )
        self.text_mapping = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * p["filters"], dim))
        self.image_mapping = nn.Linear(2048, dim)
        self.image_attention = _ImageLayer(
            dim, p["heads"], p["feedforward_dim"], dropout, p["image_activation"]
        )
        self.text_head = nn.Sequential(nn.Dropout(p["classifier_dropout"]), nn.Linear(dim, 2))
        self.image_head = nn.Sequential(nn.Dropout(p["classifier_dropout"]), nn.Linear(dim, 2))
        self.beta = p["beta"]
        self.recipe = {key: deepcopy(value) for key, value in p.items() if key != "regions"}

    def prepare(self, train_dataset):
        self.image_encoder.inputs_from_rows(train_dataset)

    def _scores(self, batch):
        text = self.text_encoder(batch["text"])
        mask = self.text_encoder.token_mask(batch["text"])
        text = F.normalize(self.text_mapping(_pool_windows(text, mask, self.convolutions)), dim=-1)
        paths = (
            batch["image_path"]
            if hasattr(self.image_encoder, "matrix")
            else self.image_encoder.paths(batch["image_path"])
        )
        image = self.image_mapping(self.image_encoder(paths))
        image = F.normalize(self.image_attention(image).mean(1), dim=-1)
        return _multiplicative_scores(self.text_head(text), self.image_head(image), self.beta)

    def calculate_loss(self, batch):
        scores = self._scores(batch)
        return {"multiplicative": -scores.gather(1, batch["labels"][:, None]).mean()}

    def predict(self, batch):
        return self._scores(batch)

    def get_extra_state(self):
        return {"schema": 1, "recipe": deepcopy(self.recipe)}

    def set_extra_state(self, state):
        if state != {"schema": 1, "recipe": self.recipe}:
            raise ValueError("LIIMR checkpoint has a different computation recipe.")
