"""MMCAN: matching-gated bidirectional co-attention and mutual distillation."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.data.features import file_digest, fingerprint
from mmfact.materials.vilt import prepare_vilt_pair
from mmfact.models_classification._encoders import (
    BertEncoder,
    MAETokenEncoder,
    _load_pretrained_network,
    _PretrainedEncoder,
)
from mmfact.models_classification._transformers import sinusoidal_positions
from mmfact.registry import register_model


def translation_protocol():
    return {
        "provider": "Google Cloud Translation",
        "api": "v2",
        "source": "zh-CN",
        "target": "en",
        "format": "text",
        "model": "nmt",
    }


def _text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Translations(nn.Module):
    """Fixed translations bound to raw text, carried in safe detector checkpoints."""

    def __init__(self, language, manifest):
        super().__init__()
        self.language, self.manifest = language, Path(manifest) if manifest else None
        self.records = {}
        self.loaded = language == "en"

    @staticmethod
    def _validate(record):
        if not isinstance(record, dict) or set(record) != {"text_sha256", "translation"}:
            raise ValueError("MMCAN translation records require text_sha256 and translation.")
        sha, translation = record["text_sha256"], record["translation"]
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
            or not isinstance(translation, str)
            or not translation.strip()
        ):
            raise ValueError("MMCAN translation record has invalid content.")
        return dict(record)

    def load(self):
        parsed = {}
        if self.manifest is None:
            raise ValueError("MMCAN Chinese matching requires a fixed translation manifest.")
        with self.manifest.open(encoding="utf-8") as stream:
            if json.loads(next(stream, "null")) != {
                "schema": 1,
                "generator": translation_protocol(),
            }:
                raise ValueError("MMCAN translation manifest has an incompatible protocol.")
            for line in stream:
                if not line.strip():
                    continue
                record = self._validate(json.loads(line))
                key = record["text_sha256"]
                if key in parsed or (key in self.records and self.records[key] != record):
                    raise ValueError("MMCAN translation records duplicate or conflict.")
                parsed[key] = record
        self.records.update(parsed)
        self.loaded = True

    def translate(self, text):
        if self.language == "en":
            return text
        key = _text_sha(text)
        if not self.loaded or (
            key not in self.records and self.manifest and self.manifest.is_file()
        ):
            self.load()
        if key not in self.records:
            raise ValueError(
                "Missing MMCAN translation; download the required materials with --download-materials."
            )
        return self.records[key]["translation"]

    def get_extra_state(self):
        return {
            "format_version": 1,
            "language": self.language,
            "protocol": translation_protocol(),
            "records": deepcopy(self.records),
            "records_sha256": fingerprint(self.records),
            "loaded": self.loaded,
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("language") != self.language
            or state.get("protocol") != translation_protocol()
            or not isinstance(state.get("records"), dict)
            or type(state.get("loaded")) is not bool
        ):
            raise ValueError("MMCAN translation checkpoint has an incompatible protocol.")
        records = {key: self._validate(record) for key, record in state["records"].items()}
        if any(key != record["text_sha256"] for key, record in records.items()) or fingerprint(
            records
        ) != state.get("records_sha256"):
            raise ValueError("MMCAN translation checkpoint content hash is invalid.")
        self.records, self.loaded = records, state["loaded"]


class _MatchingNetwork(nn.Module):
    def __init__(self, directory):
        super().__init__()
        from safetensors.torch import load_file
        from transformers import ViltModel

        directory = Path(directory)
        self.backbone = _load_pretrained_network(ViltModel, str(directory), None)
        weights = load_file(str(directory / "itm_head.safetensors"))
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("MMCAN ITM head contains nonfinite weights.")
        self.head = nn.Linear(self.backbone.config.hidden_size, 2)
        self.head.load_state_dict(weights, strict=True)

    def forward(self, inputs):
        return self.head(self.backbone(**inputs).pooler_output)


class _ViltMatching(_PretrainedEncoder):
    """Real frozen two-class ITM; pair-content identity covers raw image bytes."""

    packages = ("torch", "transformers", "tokenizers", "Pillow", "safetensors")

    def __init__(self, directory, language, translation_manifest, max_length):
        super().__init__()
        from transformers import ViltProcessor

        self.translations = _Translations(language, translation_manifest)
        self.network = _MatchingNetwork(directory)
        self.processor = ViltProcessor.from_pretrained(directory)
        self.max_length = max_length
        if max_length > self.network.backbone.config.max_position_embeddings:
            raise ValueError("MMCAN matching max_length exceeds ViLT position capacity.")
        self.output_dim = 2
        self.encoding_spec = {
            "encoder": "mmcan-vilt-pretrained-binary-itm",
            "max_length": max_length,
            "config": self.network.backbone.config.to_dict(),
            "image_processor": self.processor.image_processor.to_dict(),
            "tokenizer": json.loads(self.processor.tokenizer.backend_tokenizer.to_str()),
            "special_tokens": self.processor.tokenizer.special_tokens_map,
            "padding_side": self.processor.tokenizer.padding_side,
            "truncation_side": self.processor.tokenizer.truncation_side,
            "language": language,
            "translation_protocol": translation_protocol() if language == "zh" else "identity",
            "patch_sampling": "single-pair-cpu-rng-sha256-v1",
            "image_resize": "vilt-native-grid-min32-v1",
        }
        self._set_frozen(True)

    def pairs(self, texts, paths):
        return [
            json.dumps(
                [self.translations.translate(text), path, file_digest(path)], ensure_ascii=False
            )
            for text, path in zip(texts, paths, strict=True)
        ]

    def inputs_from_rows(self, rows):
        return self.pairs([row["text"] for row in rows], [row["image_path"] for row in rows])

    def _encode(self, pairs):
        outputs = []
        for pair in pairs:
            text, path, digest = json.loads(pair)
            if file_digest(path) != digest:
                raise ValueError("MMCAN image changed after matching cache identity was bound.")
            with Image.open(path) as image:
                inputs = prepare_vilt_pair(self.processor, image, text, self.max_length).to(
                    self.device
                )
            # HF ViLT samples patch order even in eval. A local CPU RNG prevents
            # batch composition / cache construction from changing that sampling.
            seed = int(_text_sha(text + digest)[:15], 16)
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.random.default_generator.manual_seed(seed)
                outputs.append(self.network(inputs))
        return torch.cat(outputs)


def _mean(tokens, mask):
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(1) / weights.sum(1)


class _Attention(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dimension // heads
        self.query, self.key, self.value = (nn.Linear(dimension, dimension) for _ in range(3))
        self.projection = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask):
        batch, length, dimension = query.shape

        def split(values):
            return values.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query(query)), split(self.key(context)), split(self.value(context))
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5
        weights = self.dropout(scores.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1))
        return self.projection((weights @ v).transpose(1, 2).reshape(batch, length, dimension))


class _PeerLayer(nn.Module):
    def __init__(self, dimension, heads, length, matching_dim, dropout, eps):
        super().__init__()
        self.cross = _Attention(dimension, heads, dropout)
        self.self_attention = _Attention(dimension, heads, dropout)
        self.gate = nn.Linear(matching_dim, length)
        self.norms = nn.ModuleList(nn.LayerNorm(dimension, eps=eps) for _ in range(3))
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, query_mask, context_mask, matching):
        gate = self.gate(matching).sigmoid()[:, : query.shape[1], None]
        query = self.norms[0](query + self.dropout(gate * self.cross(query, context, context_mask)))
        query = self.norms[1](query + self.dropout(self.self_attention(query, query, query_mask)))
        query = self.norms[2](query + self.dropout(self.ffn(query)))
        return query * query_mask.unsqueeze(-1).to(query.dtype)


def mutual_kl(text_logits, image_logits):
    """Eq.11: symmetric KL, both live peers, mean over examples / sum over classes."""
    text, image = F.log_softmax(text_logits, -1), F.log_softmax(image_logits, -1)
    return F.kl_div(text, image, reduction="batchmean", log_target=True) + F.kl_div(
        image, text, reduction="batchmean", log_target=True
    )


@register_model("MMCAN")
class MMCAN(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for name in (
            "num_heads",
            "encoder_layers",
            "coattention_layers",
            "encoder_ffn_dim",
            "matching_dim",
            "classifier_hidden",
        ):
            if type(p.get(name)) is not int or p[name] < 1:
                raise ValueError(f"MMCAN {name} must be a positive integer.")
        for name in ("dropout", "kl_weight", "layer_norm_eps"):
            value = p.get(name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name == "dropout" and value >= 1)
                or (name == "layer_norm_eps" and value == 0)
            ):
                raise ValueError(f"MMCAN {name} is outside its finite permitted range.")
        if (
            p["text"].get("freeze") is not True
            or p["text"].get("output") != "tokens"
            or p["text"].get("mask_padding") is not True
            or p["image"].get("freeze") is not True
            or p["image"].get("output") != "patches"
        ):
            raise ValueError("MMCAN requires frozen masked BERT tokens and MAE patches.")
        if p["matching"].get("language") not in {"en", "zh"}:
            raise ValueError("MMCAN matching.language must be en or zh.")
        if type(p["matching"].get("max_length")) is not int or p["matching"]["max_length"] < 2:
            raise ValueError("MMCAN matching.max_length must be an integer >=2.")
        self.parameters_spec, self.kl_weight = deepcopy(p), p["kl_weight"]
        self.shared_inputs = None
        if p.get("shared_materials"):
            from mmfact.materials.mmcan import MMCANSharedInputs

            root = config["preprocessing"].get("materials_root")
            if p["shared_materials"] is not True or not root:
                raise ValueError(
                    "MMCAN shared_materials=true requires preprocessing.materials_root."
                )
            if p["matching"]["max_length"] != 40:
                raise ValueError("MMCAN shared ViLT logits require matching.max_length=40.")
            directory = Path(root) / "datasets" / config["dataset"]["name"]
            self.shared_inputs = MMCANSharedInputs(directory, p["text"])
            self.parameters_spec.pop("shared_materials")
            for key in ("directory", "translation_manifest"):
                self.parameters_spec["matching"].pop(key, None)
            dimension, image_dimension = 768, 768
            lengths = (self.shared_inputs.max_length, self.shared_inputs.sequence_length)
        else:
            self.text_encoder = BertEncoder(**p["text"])
            self.image_encoder = MAETokenEncoder(**p["image"])
            self.matching_encoder = _ViltMatching(**p["matching"])
            dimension, image_dimension = self.text_encoder.output_dim, self.image_encoder.output_dim
            lengths = (self.text_encoder.max_length, self.image_encoder.sequence_length)
        if dimension != image_dimension or dimension % p["num_heads"]:
            raise ValueError("MMCAN BERT/MAE widths must agree and divide among attention heads.")
        self.input_transformers = nn.ModuleList(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    dimension, p["num_heads"], p["encoder_ffn_dim"], p["dropout"], batch_first=True
                ),
                p["encoder_layers"],
                enable_nested_tensor=False,
            )
            for _ in range(2)
        )
        self.matching_projections = nn.ModuleList(
            nn.Sequential(nn.Linear(2, p["matching_dim"]), nn.ReLU())
            for _ in range(p["coattention_layers"])
        )
        self.peers = nn.ModuleList(
            nn.ModuleList(
                _PeerLayer(
                    dimension,
                    p["num_heads"],
                    length,
                    p["matching_dim"],
                    p["dropout"],
                    p["layer_norm_eps"],
                )
                for _ in range(p["coattention_layers"])
            )
            for length in lengths
        )
        self.classifiers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dimension, p["classifier_hidden"]), nn.Linear(p["classifier_hidden"], 2)
            )
            for _ in range(2)
        )

    def forward(self, batch):
        texts, paths = batch["text"], batch["image_path"]
        if (
            not texts
            or len(texts) != len(paths)
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ValueError("MMCAN needs one nonempty text and image per news item.")
        if self.shared_inputs is not None:
            text, image, text_mask, matching = self.shared_inputs(batch)
        else:
            text, image = self.text_encoder(texts), self.image_encoder(paths)
            text_mask = self.text_encoder.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.text_encoder.max_length,
                return_tensors="pt",
            )["attention_mask"].to(device=text.device, dtype=torch.bool)
            matching = self.matching_encoder(self.matching_encoder.pairs(texts, paths))
        image_mask = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
        masks = (text_mask, image_mask)
        values = [
            encoder(tokens + sinusoidal_positions(tokens), src_key_padding_mask=~mask)
            * mask.unsqueeze(-1).to(tokens.dtype)
            for encoder, tokens, mask in zip(
                self.input_transformers, (text, image), masks, strict=True
            )
        ]
        for index, projection in enumerate(self.matching_projections):
            embed = projection(matching)
            text, image = values
            values = [
                self.peers[0][index](text, image, text_mask, image_mask, embed),
                self.peers[1][index](image, text, image_mask, text_mask, embed),
            ]
        return tuple(
            head(_mean(tokens, mask))
            for head, tokens, mask in zip(self.classifiers, values, masks, strict=True)
        )

    def calculate_loss(self, batch):
        text, image = self(batch)
        return {
            "text_classification": F.cross_entropy(text, batch["labels"]),
            "image_classification": F.cross_entropy(image, batch["labels"]),
            "mutual_distillation": self.kl_weight * mutual_kl(text, image),
        }

    def predict(self, batch):
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            text, image = self(batch)
            return torch.logaddexp(F.log_softmax(text, -1), F.log_softmax(image, -1)) - math.log(2)
        finally:
            for module, mode in modes.items():
                module.training = mode

    def get_extra_state(self):
        return {"format_version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("MMCAN checkpoint parameters do not match this configuration.")
