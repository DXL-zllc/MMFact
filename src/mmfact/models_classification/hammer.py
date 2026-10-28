"""HAMMER: complete DGM4 manipulation pretraining followed by news fine-tuning."""

import hashlib
import math
import re
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.materials.encoders import pretrained_name
from mmfact.models_classification._hammer_data import (
    DGM4Data,
    encode_text,
    image_inputs,
)
from mmfact.registry import register_model


def _initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _head(dimension, outputs):
    return nn.Sequential(
        nn.Linear(dimension, 2 * dimension),
        nn.LayerNorm(2 * dimension, eps=1e-5),
        nn.GELU(),
        nn.Linear(2 * dimension, 2 * dimension),
        nn.LayerNorm(2 * dimension, eps=1e-5),
        nn.GELU(),
        nn.Linear(2 * dimension, outputs),
    )


def _resize_positions(values, grid):
    if values.ndim != 3:
        raise ValueError("HAMMER visual positions must be a rank-three tensor.")
    old = math.isqrt(values.shape[1] - 1)
    if values.ndim != 3 or old * old != values.shape[1] - 1 or values.shape[0] != 1:
        raise ValueError("HAMMER visual positions must contain CLS and a square patch grid.")
    if old == grid:
        return values
    patches = values[:, 1:].reshape(1, old, old, values.shape[-1]).permute(0, 3, 1, 2)
    patches = F.interpolate(patches.float(), (grid, grid), mode="bicubic", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, grid * grid, values.shape[-1])
    return torch.cat((values[:, :1], patches.to(values.dtype)), dim=1)


class _SplitBert(nn.Module):
    """One pretrained BERT; upper indexed layers gain bidirectional image fusion."""

    def __init__(self, options):
        super().__init__()
        from transformers import BertModel
        from transformers.models.bert.modeling_bert import BertAttention

        self.bert = BertModel.from_pretrained(
            options["model_name"],
            revision=options["revision"],
            add_pooling_layer=False,
            attn_implementation="eager",
        )
        self.config = self.bert.config
        self.fusion_layer = options["fusion_layer"]
        if not 0 < self.fusion_layer < self.config.num_hidden_layers:
            raise ValueError(
                "HAMMER fusion_layer must split the BERT layers into two nonempty parts."
            )
        if options["max_length"] > self.config.max_position_embeddings:
            raise ValueError("HAMMER text length exceeds the pretrained BERT position limit.")
        if self.config.position_embedding_type != "absolute":
            raise ValueError("HAMMER requires standard absolute-position BERT.")
        cross_config = deepcopy(self.config)
        for index, layer in enumerate(self.bert.encoder.layer):
            if index >= self.fusion_layer:
                layer.crossattention = BertAttention(
                    cross_config, position_embedding_type="absolute"
                )
                layer.crossattention.apply(_initialize)
        self.dropout = nn.Dropout(0.1)
        self.classifier = _head(self.config.hidden_size, 2)
        self.classifier.apply(_initialize)

    def text(self, input_ids, attention_mask):
        values = self.bert.embeddings(input_ids=input_ids)
        mask = self._mask(attention_mask, values.dtype)
        for layer in self.bert.encoder.layer[: self.fusion_layer]:
            values = layer(values, attention_mask=mask)[0]
        return values

    @staticmethod
    def _mask(mask, dtype):
        return (1 - mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min

    def fusion(self, values, attention_mask, image):
        mask = self._mask(attention_mask, values.dtype)
        for layer in self.bert.encoder.layer[self.fusion_layer :]:
            values = layer.attention(values, attention_mask=mask)[0]
            values = layer.crossattention(values, encoder_hidden_states=image)[0]
            values = layer.feed_forward_chunk(values)
        return values

    def forward(self, input_ids, attention_mask, image):
        values = self.fusion(self.text(input_ids, attention_mask), attention_mask, image)
        return self.classifier(self.dropout(values[:, 1:]))


def _paired_giou(predicted, target):
    """Paired cxcywh GIoU, including the source's all-zero target sentinel."""
    predicted, target = _loss_precision(predicted), _loss_precision(target)
    first = torch.cat(
        (predicted[:, :2] - predicted[:, 2:] / 2, predicted[:, :2] + predicted[:, 2:] / 2), dim=-1
    )
    second = torch.cat(
        (target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2), dim=-1
    )
    intersection = (
        (torch.minimum(first[:, 2:], second[:, 2:]) - torch.maximum(first[:, :2], second[:, :2]))
        .clamp_min(0)
        .prod(-1)
    )
    union = predicted[:, 2:].prod(-1) + target[:, 2:].prod(-1) - intersection
    enclosing = (
        (torch.maximum(first[:, 2:], second[:, 2:]) - torch.minimum(first[:, :2], second[:, :2]))
        .clamp_min(0)
        .prod(-1)
    )
    epsilon = torch.finfo(predicted.dtype).eps
    return intersection / union.clamp_min(epsilon) - (enclosing - union) / enclosing.clamp_min(
        epsilon
    )


def _token_loss(student, teacher, labels, alpha):
    student, teacher = _loss_precision(student), _loss_precision(teacher)
    active = labels.ne(-100)
    if not active.any():
        raise ValueError("HAMMER token grounding needs supervised content tokens.")
    logits = student[active]
    hard = F.cross_entropy(logits, labels[active])
    log_student = F.log_softmax(logits, dim=-1)
    log_teacher = F.log_softmax(teacher.detach()[active], dim=-1)
    divergence = (log_student.exp() * (log_student - log_teacher)).sum(-1).mean()
    return (1 - alpha) * hard + alpha * divergence


def _source_mac(
    image, text, image_teacher, text_teacher, image_queue, text_queue, temperature, labels, alpha
):
    # Large queues and small probabilities require FP32 loss arithmetic under
    # AMP. Preserve FP32/FP64 execution exactly, including the formula's gradients.
    with torch.autocast(device_type=image.device.type, enabled=False):
        values = [_loss_precision(value) for value in
                  (image, text, image_teacher, text_teacher, image_queue, text_queue, temperature)]
        return _source_mac_values(*values, labels, alpha)


def _loss_precision(value):
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _source_mac_values(
    image, text, image_teacher, text_teacher, image_queue, text_queue, temperature, labels, alpha
):
    """Four source soft CEs; manipulated cross-modal hard targets are zero."""
    image_teacher, text_teacher = image_teacher.detach(), text_teacher.detach()
    image_keys = torch.cat((image_teacher.T, image_queue.detach().clone()), dim=1)
    text_keys = torch.cat((text_teacher.T, text_queue.detach().clone()), dim=1)
    count = image.shape[0]
    with torch.no_grad():
        intra = image.new_zeros(count, image_keys.shape[1])
        intra[:, :count].fill_diagonal_(1)
        cross = intra * labels.eq(0).to(image.dtype)[:, None]
        targets = [
            alpha * F.softmax(image_teacher @ text_keys / temperature, dim=-1)
            + (1 - alpha) * cross,
            alpha * F.softmax(text_teacher @ image_keys / temperature, dim=-1)
            + (1 - alpha) * cross,
            intra,
            intra,
        ]
    scores = [image @ text_keys, text @ image_keys, image @ image_keys, text @ text_keys]
    return (
        sum(
            -(F.log_softmax(score / temperature, dim=-1) * target).sum(-1).mean()
            for score, target in zip(scores, targets)
        )
        / 4
    )


@register_model("HAMMER")
class HAMMER(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        self._validate(p)
        self.parameters_spec = deepcopy(p)
        self.auxiliary = DGM4Data(p["auxiliary"], config["training"]["seeds"][0])
        from transformers import BertTokenizerFast, ViTModel

        text_options = {**p["text"], "model_name": pretrained_name(config, p["text"]["model_name"])}
        self.tokenizer = BertTokenizerFast.from_pretrained(
            text_options["model_name"], revision=p["text"]["revision"]
        )
        self.text_encoder = _SplitBert(text_options)
        self.visual_encoder = ViTModel.from_pretrained(
            pretrained_name(config, p["image"]["model_name"]),
            revision=p["image"]["revision"],
            add_pooling_layer=False,
            attn_implementation="eager",
        )
        dimension = self.text_encoder.config.hidden_size
        vision = self.visual_encoder.config
        # The published HF DeiT config uses 1e-12; ALBEF/HAMMER specifies 1e-6.
        vision.layer_norm_eps = 1e-6
        for module in self.visual_encoder.modules():
            if isinstance(module, nn.LayerNorm):
                module.eps = 1e-6
        if vision.hidden_size != dimension or vision.num_channels != 3:
            raise ValueError(
                "HAMMER image/text hidden dimensions must agree and images must be RGB."
            )
        patch = vision.patch_size
        if type(patch) is not int or p["image"]["image_size"] % patch:
            raise ValueError("HAMMER square image size must be divisible by its patch size.")
        self.visual_encoder.embeddings.position_embeddings = nn.Parameter(
            _resize_positions(
                self.visual_encoder.embeddings.position_embeddings.detach(),
                p["image"]["image_size"] // patch,
            )
        )
        vision.image_size = p["image"]["image_size"]
        self.visual_encoder.embeddings.patch_embeddings.image_size = (vision.image_size,) * 2
        self.visual_encoder.embeddings.patch_embeddings.num_patches = (
            vision.image_size // patch
        ) ** 2
        self.vision_proj, self.text_proj = (
            nn.Linear(dimension, p["projection_dim"]) for _ in range(2)
        )
        self.itm_head, self.bbox_head, self.cls_head = (_head(dimension, n) for n in (2, 4, 4))
        self.norm_layer_it_cross_atten = nn.LayerNorm(dimension, eps=1e-5)
        self.norm_layer_aggr = nn.LayerNorm(dimension, eps=1e-5)
        heads = self.text_encoder.config.num_attention_heads
        self.it_cross_attn = nn.MultiheadAttention(dimension, heads, dropout=0, batch_first=True)
        self.aggregator = nn.MultiheadAttention(dimension, heads, dropout=0, batch_first=True)
        self.cls_token_local = nn.Parameter(torch.empty(1, 1, dimension))
        nn.init.trunc_normal_(self.cls_token_local, std=0.02)
        for module in (
            self.vision_proj,
            self.text_proj,
            self.itm_head,
            self.bbox_head,
            self.cls_head,
            self.it_cross_attn,
            self.aggregator,
        ):
            module.apply(_initialize)
        self.temp = nn.Parameter(torch.tensor(float(p["temperature"])))
        for name in ("visual_encoder", "text_encoder", "vision_proj", "text_proj"):
            self.add_module(name + "_m", deepcopy(getattr(self, name)))
        self.register_buffer(
            "image_queue", F.normalize(torch.randn(p["projection_dim"], p["queue_size"]), dim=0)
        )
        self.register_buffer(
            "text_queue", F.normalize(torch.randn(p["projection_dim"], p["queue_size"]), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self._stage, self._epoch, self._batch, self._batches = None, 1, 0, 1
        self._auxiliary_accumulation = 1
        self._pending_image_keys, self._pending_text_keys = [], []
        self.initialization = {
            "status": "custom" if p["initialization"]["checkpoint"] is None else "unprepared"
        }
        self._copy_teacher()
        self.set_training_stage(None)

    @staticmethod
    def _validate(p):
        for section in ("text", "image"):
            for key in ("model_name", "revision"):
                if not isinstance(p[section].get(key), str) or not p[section][key].strip():
                    raise ValueError(f"HAMMER {section}.{key} must be a nonempty string.")
        for section, keys in (
            ("text", ("max_length", "max_words", "fusion_layer")),
            ("image", ("image_size",)),
        ):
            for key in keys:
                if type(p[section].get(key)) is not int or p[section][key] < 1:
                    raise ValueError(f"HAMMER {section}.{key} must be a positive integer.")
        if p["text"]["max_length"] < 3:
            raise ValueError("HAMMER text max_length must accommodate CLS, content and SEP.")
        for key in ("projection_dim", "queue_size"):
            if type(p.get(key)) is not int or p[key] < 1:
                raise ValueError(f"HAMMER {key} must be a positive integer.")
        for key, minimum, maximum in (
            ("momentum", 0, 1),
            ("alpha", 0, 1),
            ("temperature", 0.001, 0.5),
        ):
            if (
                type(p.get(key)) not in (int, float)
                or not math.isfinite(p[key])
                or not minimum <= p[key] <= maximum
            ):
                raise ValueError(f"HAMMER {key} is outside its finite valid range.")
        weights = p["loss_weights"]
        if set(weights) != {"mac", "binary", "bbox_l1", "bbox_giou", "token", "multilabel"}:
            raise ValueError("HAMMER requires all six loss weights.")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in weights.values()
        ):
            raise ValueError("HAMMER loss weights must be finite and positive.")
        initialization = p["initialization"]
        if (
            set(initialization) != {"checkpoint", "language_adapted"}
            or type(initialization["language_adapted"]) is not bool
            or (
                initialization["checkpoint"] is not None
                and (
                    not isinstance(initialization["checkpoint"], str)
                    or not initialization["checkpoint"].strip()
                )
            )
        ):
            raise ValueError(
                "HAMMER initialization needs an explicit path or null and a boolean language adaptation."
            )

    @property
    def _pairs(self):
        return [
            (getattr(self, name), getattr(self, name + "_m"))
            for name in ("visual_encoder", "text_encoder", "vision_proj", "text_proj")
        ]

    @torch.no_grad()
    def _copy_teacher(self):
        for online, teacher in self._pairs:
            teacher.load_state_dict(online.state_dict(), strict=True)
            teacher.requires_grad_(False)
            teacher.eval()

    @torch.no_grad()
    def _momentum_update(self):
        momentum = self.parameters_spec["momentum"]
        for online, teacher in self._pairs:
            for source, destination in zip(online.parameters(), teacher.parameters()):
                destination.mul_(momentum).add_(source, alpha=1 - momentum)

    @torch.no_grad()
    def _enqueue(self, image, text):
        size, pointer = self.parameters_spec["queue_size"], int(self.queue_ptr[0])
        count = image.shape[0]
        if count >= size:
            pointer = (pointer + count - size) % size
            image, text = image[-size:], text[-size:]
        offset, remaining = 0, image.shape[0]
        while remaining:
            take = min(remaining, size - pointer)
            self.image_queue[:, pointer : pointer + take] = image[offset : offset + take].T
            self.text_queue[:, pointer : pointer + take] = text[offset : offset + take].T
            pointer = (pointer + take) % size
            offset, remaining = offset + take, remaining - take
        self.queue_ptr[0] = pointer

    def train(self, mode=True):
        super().train(mode)
        for _, teacher in self._pairs:
            teacher.eval()
        return self

    def training_stages(self):
        return (TrainingStage("dgm4_pretrain"), TrainingStage("news_finetune"))

    def set_training_stage(self, stage):
        if stage not in (None, "dgm4_pretrain", "news_finetune"):
            raise ValueError("Unknown HAMMER training stage.")
        self._stage = stage
        self.requires_grad_(stage == "dgm4_pretrain")
        for module in (self.visual_encoder, self.text_encoder.bert, self.itm_head):
            module.requires_grad_(True)
        for _, teacher in self._pairs:
            teacher.requires_grad_(False)
            teacher.eval()

    def training_stage_data(self, stage, train_data, val_data, options):
        if stage == "news_finetune":
            return train_data, val_data
        if stage == "dgm4_pretrain":
            accumulation = options.get("gradient_accumulation_steps", 1)
            if type(accumulation) is not int or accumulation < 1:
                raise ValueError("HAMMER gradient accumulation must be a positive integer.")
            self._auxiliary_accumulation = accumulation
            train = self.auxiliary.loader("train", train_data, options)
            self._batches = len(train)
            return train, self.auxiliary.loader("val", val_data, options)
        raise ValueError("Unknown HAMMER training stage.")

    def on_epoch_start(self, epoch, total_epochs):
        if self._pending_image_keys or self._pending_text_keys:
            raise RuntimeError("HAMMER cannot discard a partial momentum/queue window.")
        self._epoch, self._batch = epoch, 0

    def _begin_auxiliary_microbatch(self, update):
        if update:
            if self._batch >= self._batches:
                raise RuntimeError("HAMMER auxiliary batch counter exceeds its declared loader.")
            if self._batch % self._auxiliary_accumulation == 0:
                with torch.no_grad():
                    self.temp.clamp_(0.001, 0.5)
                self._momentum_update()

    def _finish_auxiliary_microbatch(self, image, text):
        self._pending_image_keys.append(image.detach())
        self._pending_text_keys.append(text.detach())
        self._batch += 1
        if self._batch % self._auxiliary_accumulation == 0 or self._batch == self._batches:
            self._enqueue(torch.cat(self._pending_image_keys), torch.cat(self._pending_text_keys))
            self._pending_image_keys.clear()
            self._pending_text_keys.clear()

    def prepare(self, train_dataset):
        self.auxiliary.prepare()
        path = self.parameters_spec["initialization"]["checkpoint"]
        if path is not None and self.initialization["status"] != "loaded":
            self._load_albef(Path(path).expanduser())
        self._copy_teacher()

    def _inputs(self, batch, *, supervised=False, augment=False):
        p = self.parameters_spec
        tokens = encode_text(
            self.tokenizer,
            batch["text"],
            max_words=p["text"]["max_words"],
            max_length=p["text"]["max_length"],
            fake_text_pos=batch["fake_text_pos"] if supervised else None,
        )
        pixels, boxes = image_inputs(
            batch["image_path"],
            image_size=p["image"]["image_size"],
            training=augment,
            boxes=batch["box_xyxy"] if supervised else None,
        )
        device = self.temp.device
        return (
            {key: value.to(device) for key, value in tokens.items()},
            pixels.to(device),
            boxes.to(device) if boxes is not None else None,
        )

    def _encode(self, tokens, pixels):
        image = self.visual_encoder(pixels).last_hidden_state
        text = self.text_encoder.text(tokens["input_ids"], tokens["attention_mask"])
        return image, text

    def _bbox(self, image, text, mask):
        normalized_image = self.norm_layer_it_cross_atten(image)
        normalized_text = self.norm_layer_it_cross_atten(text)
        interacted = (
            image
            + self.it_cross_attn(
                normalized_image,
                normalized_text,
                normalized_text,
                key_padding_mask=~mask.bool(),
                need_weights=False,
            )[0]
        )
        query = self.norm_layer_aggr(self.cls_token_local.expand(image.shape[0], -1, -1))
        patches = self.norm_layer_aggr(interacted[:, 1:])
        aggregated = self.aggregator(query, patches, patches, need_weights=False)[0][:, 0]
        # Low-precision normalized coordinates can erase small manipulated
        # regions. Keep this small regression head and its geometry in FP32.
        with torch.autocast(device_type=aggregated.device.type, enabled=False):
            return self.bbox_head(_loss_precision(aggregated)).sigmoid()

    def forward(self, batch):
        tokens, pixels, _ = self._inputs(batch, augment=self.training)
        image, text = self._encode(tokens, pixels)
        fused = self.text_encoder.fusion(text, tokens["attention_mask"], image)
        return self.itm_head(fused[:, 0])

    def _auxiliary_loss(self, batch, *, update):
        self._begin_auxiliary_microbatch(update)
        tokens, pixels, target_boxes = self._inputs(batch, supervised=True, augment=update)
        labels = batch["labels"].to(self.temp.device)
        alpha = self.parameters_spec["alpha"]
        if self._epoch == 1:
            windows = math.ceil(self._batches / self._auxiliary_accumulation)
            position = (self._batch // self._auxiliary_accumulation if update
                        else math.ceil(self._batch / self._auxiliary_accumulation))
            alpha *= min(position / windows, 1)
        image, text = self._encode(tokens, pixels)
        image_feature = F.normalize(self.vision_proj(image[:, 0]), dim=-1)
        text_feature = F.normalize(self.text_proj(text[:, 0]), dim=-1)
        with torch.no_grad():
            image_teacher = self.visual_encoder_m(pixels).last_hidden_state
            text_teacher = self.text_encoder_m.text(tokens["input_ids"], tokens["attention_mask"])
            image_key = F.normalize(self.vision_proj_m(image_teacher[:, 0]), dim=-1)
            text_key = F.normalize(self.text_proj_m(text_teacher[:, 0]), dim=-1)
        mac = _source_mac(
            image_feature,
            text_feature,
            image_key,
            text_key,
            self.image_queue,
            self.text_queue,
            self.temp,
            labels,
            alpha,
        )
        fused = self.text_encoder.fusion(text, tokens["attention_mask"], image)
        predicted_boxes = self._bbox(image, text, tokens["attention_mask"])
        # The original TMG path independently resamples online BERT dropout.
        student_tokens = self.text_encoder(tokens["input_ids"], tokens["attention_mask"], image)
        with torch.no_grad():
            teacher_tokens = self.text_encoder_m(
                tokens["input_ids"], tokens["attention_mask"], image_teacher
            )
        losses = {
            "mac": mac,
            "binary": F.cross_entropy(self.itm_head(fused[:, 0]), labels),
            "bbox_l1": F.l1_loss(predicted_boxes, target_boxes, reduction="sum") / labels.numel(),
            "bbox_giou": (1 - _paired_giou(predicted_boxes, target_boxes)).mean(),
            "token": _token_loss(student_tokens, teacher_tokens, tokens["token_labels"], alpha),
            "multilabel": F.binary_cross_entropy_with_logits(
                self.cls_head(fused[:, 0]), batch["manipulation_labels"].to(self.temp.device)
            ),
        }
        if update:
            self._finish_auxiliary_microbatch(image_key, text_key)
        return {
            key: value * self.parameters_spec["loss_weights"][key] for key, value in losses.items()
        }

    def _news_loss(self, batch):
        """Target news needs raw text, images and truth labels, without grounding."""
        inputs = {key: batch[key] for key in ("text", "image_path")}
        return {"binary": F.cross_entropy(self(inputs), batch["labels"])}

    def calculate_loss(self, batch):
        if self._stage == "dgm4_pretrain":
            return self._auxiliary_loss(batch, update=self.training)
        return self._news_loss(batch)

    def validation_loss(self, batch):
        if self._stage == "dgm4_pretrain":
            return self._auxiliary_loss(batch, update=False)
        return self.calculate_loss(batch)

    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            return self(batch)
        finally:
            for module, mode in modes:
                module.training = mode

    def get_extra_state(self):
        if self._pending_image_keys or self._pending_text_keys:
            raise RuntimeError("HAMMER checkpoints must be written at an optimizer-window boundary.")
        return deepcopy(
            {
                "version": 2,
                "parameters": self.parameters_spec,
                "initialization": self.initialization,
                "stage": self._stage,
                "epoch": self._epoch,
                "batch": self._batch,
                "batches": self._batches,
                "auxiliary_accumulation_steps": self._auxiliary_accumulation,
            }
        )

    def set_extra_state(self, state):
        extra = {"auxiliary_accumulation_steps"} if isinstance(state, dict) and state.get("version") == 2 else set()
        if (
            not isinstance(state, dict)
            or set(state)
            != {"version", "parameters", "initialization", "stage", "epoch", "batch", "batches"} | extra
            or type(state["version"]) is not int
            or state["version"] not in (1, 2)
            or type(state.get("auxiliary_accumulation_steps", 1)) is not int
            or state.get("auxiliary_accumulation_steps", 1) < 1
            or state["parameters"] != self.parameters_spec
            or state["stage"] not in (None, "dgm4_pretrain", "news_finetune")
            or any(
                type(state[key]) is not int or state[key] < (0 if key == "batch" else 1)
                for key in ("epoch", "batch", "batches")
            )
        ):
            raise ValueError("HAMMER checkpoint configuration or training state is incompatible.")
        record = state["initialization"]
        if not isinstance(record, dict) or record.get("status") not in (
            "custom",
            "unprepared",
            "loaded",
        ):
            raise ValueError("HAMMER initialization provenance is invalid.")
        if record["status"] == "loaded":
            if (
                set(record) != {"status", "sha256", "loaded", "excluded", "language_adapted"}
                or not isinstance(record["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or type(record["language_adapted"]) is not bool
                or record["language_adapted"]
                != self.parameters_spec["initialization"]["language_adapted"]
                or any(
                    not isinstance(record[key], list)
                    or (key == "loaded" and not record[key])
                    or any(not isinstance(value, str) for value in record[key])
                    for key in ("loaded", "excluded")
                )
            ):
                raise ValueError("HAMMER loaded initialization provenance is invalid.")
        elif set(record) != {"status"}:
            raise ValueError("HAMMER initialization provenance has unexpected fields.")
        self.initialization = deepcopy(record)
        self._auxiliary_accumulation = state.get("auxiliary_accumulation_steps", 1)
        self._pending_image_keys, self._pending_text_keys = [], []
        self._epoch, self._batch, self._batches = (
            state[key] for key in ("epoch", "batch", "batches")
        )
        self.set_training_stage(state["stage"])

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        for name in ("image_queue", "text_queue", "queue_ptr", "temp"):
            value = state.get(prefix + name)
            expected = getattr(self, name)
            if value is not None and (
                not torch.is_tensor(value)
                or value.shape != expected.shape
                or value.dtype != expected.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"HAMMER checkpoint {name} is invalid.")
        pointer = state.get(prefix + "queue_ptr")
        if pointer is not None and not 0 <= int(pointer[0]) < self.parameters_spec["queue_size"]:
            raise ValueError("HAMMER checkpoint queue pointer is outside its ring.")
        super()._load_from_state_dict(state, prefix, *args, **kwargs)

    def _load_albef(self, path):
        """Strict online mapping from published ALBEF schema, never source code execution."""
        if not path.is_file():
            raise ValueError(f"HAMMER ALBEF initialization does not exist: {path}.")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
            handle.seek(0)
            after = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                after.update(chunk)
            if after.digest() != digest.digest():
                raise ValueError("HAMMER ALBEF bytes changed while loading initialization.")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError(
                "HAMMER ALBEF initialization requires the published model state dictionary."
            )
        source = checkpoint["model"]
        if any(not isinstance(key, str) or not key for key in source):
            raise ValueError("HAMMER ALBEF source parameter names must be nonempty strings.")
        current = self.state_dict()
        mapped, loaded = {}, set()
        adapted = self.parameters_spec["initialization"]["language_adapted"]

        def assign(target, key, value=None):
            if key not in source:
                raise ValueError(f"HAMMER ALBEF initialization is missing required key {key}.")
            value = source[key] if value is None else value
            if (
                not torch.is_tensor(value)
                or value.shape != current[target].shape
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"HAMMER ALBEF key {key} has incompatible shape or values.")
            mapped[target] = value
            return key

        for target in current:
            if target.startswith("visual_encoder."):
                name = target.removeprefix("visual_encoder.")
                direct = {
                    "embeddings.cls_token": "cls_token",
                    "embeddings.position_embeddings": "pos_embed",
                    "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
                    "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
                    "layernorm.weight": "norm.weight",
                    "layernorm.bias": "norm.bias",
                }
                if name in direct:
                    key = "visual_encoder." + direct[name]
                    value = source.get(key)
                    if name == "embeddings.position_embeddings" and torch.is_tensor(value):
                        value = _resize_positions(value, math.isqrt(current[target].shape[1] - 1))
                    loaded.add(assign(target, key, value))
                    continue
                match = re.fullmatch(r"encoder.layer.(\d+).(.+)", name)
                if match is None:
                    raise ValueError(f"HAMMER has an unmapped visual parameter {target}.")
                index, suffix = match.groups()
                prefix = f"visual_encoder.blocks.{index}."
                qkv = re.fullmatch(r"attention.attention.(query|key|value).(weight|bias)", suffix)
                if qkv:
                    key = prefix + "attn.qkv." + qkv[2]
                    value = source.get(key)
                    if (
                        not torch.is_tensor(value)
                        or value.ndim != current[target].ndim
                        or value.shape[0] != 3 * current[target].shape[0]
                    ):
                        raise ValueError(f"HAMMER ALBEF concatenated qkv {key} is invalid.")
                    piece = value.chunk(3, dim=0)[("query", "key", "value").index(qkv[1])]
                    loaded.add(assign(target, key, piece))
                else:
                    replacements = {
                        "layernorm_before.": "norm1.",
                        "layernorm_after.": "norm2.",
                        "attention.output.dense.": "attn.proj.",
                        "intermediate.dense.": "mlp.fc1.",
                        "output.dense.": "mlp.fc2.",
                    }
                    for old, new in replacements.items():
                        if suffix.startswith(old):
                            loaded.add(assign(target, prefix + new + suffix.removeprefix(old)))
                            break
                    else:
                        raise ValueError(f"HAMMER has an unmapped visual parameter {target}.")
            elif target.startswith(("vision_proj.", "text_proj.")):
                loaded.add(assign(target, target))
            elif target.startswith("text_encoder.bert.") and (
                not adapted or ".crossattention." in target
            ):
                loaded.add(assign(target, target))
        # All checks finish before the first online weight changes.
        with torch.no_grad():
            for name, value in mapped.items():
                current[name].copy_(value)
        self.initialization = {
            "status": "loaded",
            "sha256": digest.hexdigest(),
            "loaded": sorted(loaded),
            "excluded": sorted(set(source) - loaded),
            "language_adapted": adapted,
        }
        self._copy_teacher()
