"""NLIN: shared LoRA encoder/decoder, image-language resources and CLIP compensation."""

import json
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder, pretrained_name
from mmfact.materials.resources import material_resources
from mmfact.models_classification._encoders import (
    CLIPImageEncoder,
    CLIPTextEncoder,
    _load_pretrained_network,
)
from mmfact.models_classification._entities import resource_digest
from mmfact.models_classification._nlin_resources import NLINResources
from mmfact.registry import register_model


class _CheckedCLIP:
    """Preserve the public CLIP recipes but reject missing active pretrained weights."""

    def _load(self, model_name, revision, freeze, normalize, output="pooled"):
        from transformers import AutoModel

        if output != "pooled" or freeze is not True or normalize is not False:
            raise ValueError("NLIN requires frozen, unnormalized projected CLIP poolers.")

        def required(network):
            text = self.input_kind == "text"
            prefixes = (
                ("text_model.", "text_projection.")
                if text
                else ("vision_model.", "visual_projection.")
            )
            return {key for key in network.state_dict() if key.startswith(prefixes)}

        model = _load_pretrained_network(AutoModel, model_name, revision, required)
        if model.config.model_type not in {"clip", "chinese_clip"}:
            raise ValueError("NLIN requires an actual CLIP or Chinese-CLIP checkpoint.")
        text = self.input_kind == "text"
        self.network = nn.ModuleDict(
            {
                "encoder": model.text_model if text else model.vision_model,
                "projection": model.text_projection if text else model.visual_projection,
            }
        )
        self.output_dim = model.config.projection_dim
        self._cls_pooling = text and model.config.model_type == "chinese_clip"
        self.normalize, self.output = normalize, output
        self.encoding_spec = {
            "encoder": f"{model.config.model_type}_{self.input_kind}",
            "model_name": str(model_name),
            "revision": getattr(model.config, "_commit_hash", None) or revision,
            "config": model.config.to_dict(),
            "output": "projected_cls" if self._cls_pooling else "projected_pooler",
            "normalize": normalize,
        }
        self._set_frozen(freeze)


class _CLIPText(_CheckedCLIP, CLIPTextEncoder):
    pass


class _CLIPImage(_CheckedCLIP, CLIPImageEncoder):
    pass


class _MappingBlock(nn.Module):
    """Eq.4 / Fig.3: learned-query cross attention, then self attention, each add+norm."""

    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.cross = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm, self.self_norm = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prompts, memory):
        values = self.cross(prompts, memory, memory, need_weights=False)[0]
        values = self.cross_norm(prompts + self.dropout(values))
        refined = self.self_attention(values, values, values, need_weights=False)[0]
        return self.self_norm(values + self.dropout(refined))


def _knowledge(entities):
    return " | ".join(
        f"{entity['label']}: {entity['description']}"
        for entity in entities
        if entity["description"]
    )


def _bounded_context(parts, max_words):
    """Keep every available source under the shared context budget.

    Shared dense labels/news can be much longer than their paper counterparts.
    Prefix truncation then erases trailing OCR/knowledge entirely. Water-fill
    the remaining word budget across nonempty sources, redistributing unused
    capacity from shorter sources without fitting any corpus statistics.
    """
    headers = [heading.split() for heading, _ in parts]
    values = [value.split() for _, value in parts]
    available = max_words - sum(map(len, headers))
    nonempty = sum(bool(value) for value in values)
    if available < nonempty:
        raise ValueError("NLIN context budget cannot retain every nonempty source.")
    counts = [0] * len(values)
    while available:
        active = [index for index, value in enumerate(values) if counts[index] < len(value)]
        if not active:
            break
        for index in active:
            if not available:
                break
            counts[index] += 1
            available -= 1
    return " ".join(
        word
        for heading, value, count in zip(headers, values, counts, strict=True)
        for word in (*heading, *value[:count])
    )


def _contexts(record, max_words):
    dense = " | ".join(" ".join([*item["attributes"], item["object"]]) for item in record["dense"])
    visual = _bounded_context(
        [("Caption:", record["caption"]), ("Dense labelling:", dense),
         ("OCR:", " | ".join(record["ocr_english"])),
         ("Visual background knowledge:", _knowledge(record["visual_entities"]))], max_words
    )
    text = _bounded_context(
        [("News text:", record["news_english"]),
         ("Textual background knowledge:", _knowledge(record["text_entities"]))], max_words
    )
    return visual, text


def _tokenizer_state(tokenizer):
    return {
        "backend": json.loads(tokenizer.backend_tokenizer.to_str()),
        "special_tokens": tokenizer.special_tokens_map,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
    }


@register_model("NLIN")
class NLIN(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, T5Model

        p = config["model"]["parameters"]
        if p["context"].get("allocation") != "component_waterfill_v1":
            raise ValueError("NLIN requires the declared component-preserving context allocation.")
        for group, keys in (
            ("context", ("max_words", "max_length")),
            ("mapping", ("blocks", "prompts", "heads")),
            ("lora", ("rank", "alpha")),
        ):
            if any(type(p[group][key]) is not int or p[group][key] < 1 for key in keys):
                raise ValueError(f"NLIN {group} needs positive integer sizes.")
        for group in ("mapping", "lora"):
            value = p[group]["dropout"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"NLIN {group} dropout must be finite in [0,1).")
        self.parameters_spec = deepcopy(p)
        self.resources = material_resources(config, "nlin", NLINResources, **p["resources"])
        options = p["backbone"]
        model_name = pretrained_name(config, options["model_name"])
        base = _load_pretrained_network(T5Model, model_name, options["revision"])
        if model_name != options["model_name"]:
            base.config._name_or_path = options["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=options["revision"], use_fast=True
        )
        if not self.tokenizer.is_fast or self.tokenizer.padding_side != "right":
            raise ValueError("NLIN requires a serializable fast tokenizer with right padding.")
        self.marker = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        if (
            self.marker is None
            or self.marker == self.tokenizer.unk_token_id
            or self.tokenizer.encode("<extra_id_0>", add_special_tokens=False) != [self.marker]
            or not 0 <= self.marker < base.config.vocab_size
            or base.config.decoder_start_token_id is None
        ):
            raise ValueError(
                "NLIN requires the pretrained <extra_id_0> sentinel and decoder start."
            )
        self.reasoner = get_peft_model(
            base,
            LoraConfig(
                r=p["lora"]["rank"],
                lora_alpha=p["lora"]["alpha"],
                lora_dropout=p["lora"]["dropout"],
                target_modules=["q", "v"],
                init_lora_weights="gaussian",
                bias="none",
            ),
        )
        self.text_encoder = material_encoder(config, "text_encoder", _CLIPText, **p["clip_text"])
        self.image_encoder = material_encoder(
            config, "image_encoder", _CLIPImage, **p["clip_image"]
        )
        if self.text_encoder.output_dim != self.image_encoder.output_dim:
            raise ValueError("NLIN CLIP image and text projection dimensions must match.")
        d = base.config.d_model
        if d % p["mapping"]["heads"]:
            raise ValueError("NLIN T5 dimension must be divisible by mapping heads.")
        self.clip_projection = nn.Linear(self.text_encoder.output_dim, d)
        self.prompts = nn.Parameter(torch.empty(p["mapping"]["prompts"], d))
        nn.init.normal_(self.prompts, std=0.02)
        self.mapping = nn.ModuleList(
            _MappingBlock(d, p["mapping"]["heads"], p["mapping"]["dropout"])
            for _ in range(p["mapping"]["blocks"])
        )
        self.classifier = nn.Linear(d, 1)
        self.identity = {
            "tokenizer": _tokenizer_state(self.tokenizer),
            "backbone": base.config.to_dict(),
            "marker": self.marker,
            "clip_text": deepcopy(self.text_encoder.encoding_spec),
            "clip_image": deepcopy(self.image_encoder.encoding_spec),
        }

    @property
    def device(self):
        return self.prompts.device

    def prepare(self, train_dataset):
        for row in train_dataset:
            self.resources.lookup(row["text"], row["image_path"])

    def _encode(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.parameters_spec["context"]["max_length"],
            return_tensors="pt",
        ).to(self.device)
        values = (
            self.reasoner.get_base_model().encoder(**tokens, return_dict=True).last_hidden_state
        )
        mask = tokens["attention_mask"].bool()
        return values * mask[..., None], mask

    def _memory(self, batch):
        texts, paths = batch["text"], batch["image_path"]
        if not texts or len(texts) != len(paths):
            raise ValueError("NLIN needs a nonempty aligned text/image batch.")
        contexts = [
            _contexts(
                self.resources.lookup(text, path), self.parameters_spec["context"]["max_words"]
            )
            for text, path in zip(texts, paths, strict=True)
        ]
        prompt, prompt_mask = self._encode(
            ["Identify the following news content as true or false:"] * len(texts)
        )
        visual, visual_mask = self._encode([value[0] for value in contexts])
        text, text_mask = self._encode([value[1] for value in contexts])
        mapped = self.clip_projection(
            torch.stack((self.image_encoder(paths), self.text_encoder(texts)), dim=1)
        )
        queries = self.prompts[None].expand(len(texts), -1, -1)
        for block in self.mapping:
            mapped = block(queries, mapped)
        mapped_mask = torch.ones(mapped.shape[:2], dtype=torch.bool, device=self.device)
        return torch.cat((prompt, visual, mapped, text), 1), torch.cat(
            (prompt_mask, visual_mask, mapped_mask, text_mask), 1
        )

    def forward(self, batch):
        memory, mask = self._memory(batch)
        phrase = self.tokenizer(
            ["The news is <extra_id_0>"] * len(memory), padding=True, return_tensors="pt"
        ).to(self.device)
        base = self.reasoner.get_base_model()
        start = phrase["input_ids"].new_full((len(memory), 1), base.config.decoder_start_token_id)
        ids = torch.cat((start, phrase["input_ids"]), -1)
        decoder_mask = torch.cat((torch.ones_like(start), phrase["attention_mask"]), -1)
        marker = ids.eq(self.marker) & decoder_mask.bool()
        if not marker.sum(-1).eq(1).all():
            raise ValueError("NLIN decoder phrase must contain exactly one marker.")
        hidden = base.decoder(
            input_ids=ids,
            attention_mask=decoder_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        logit = self.classifier(hidden[marker]).squeeze(-1)
        return torch.stack((torch.zeros_like(logit), logit), -1)

    def calculate_loss(self, batch):
        logits, labels = self(batch)[:, 1], batch["labels"]
        if (
            labels.dtype != torch.long
            or labels.shape != logits.shape
            or not ((labels == 0) | (labels == 1)).all()
        ):
            raise ValueError("NLIN labels must be a binary long batch vector.")
        return {"classification": F.binary_cross_entropy_with_logits(logits, labels.to(logits))}

    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self(batch)
        finally:
            for module, mode in modes:
                module.training = mode

    def get_extra_state(self):
        payload = {
            "parameters": deepcopy(self.parameters_spec),
            "identity": deepcopy(self.identity),
        }
        return {"version": 1, **payload, "sha256": resource_digest(payload)}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if state != expected:
            raise ValueError("NLIN checkpoint parameters, tokenizer or encoder identities differ.")
