"""C3N correlation convolutions with reproducible noun and detected-crop inputs."""

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.data.cache_policy import PreparedInputError
from mmfact.data.features import file_digest, fingerprint, state_digest
from mmfact.materials.encoders import hub_inputs, material_encoder, material_store
from mmfact.models_classification._encoders import (
    CLIPImageEncoder,
    CLIPTextEncoder,
    _PretrainedEncoder,
)
from mmfact.registry import register_model


class _PreparedNouns:
    spec = {"input": "annotations/nouns.jsonl"}

    def __init__(self, records, count):
        self.records, self.count = records, count

    def __call__(self, text):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        nouns = self.records[key].get("nouns")
        if (
            not isinstance(nouns, list)
            or len(nouns) != self.count
            or any(not isinstance(noun, str) for noun in nouns)
        ):
            raise PreparedInputError("Prepared C3N nouns disagree with noun_count.")
        return nouns


def _prepared_inputs(config):
    if not hub_inputs(config):
        return {}, {}
    from mmfact.materials.hub import FeatureMatrix, HubEncoder

    store = material_store(config)
    parameters = config["model"]["parameters"]
    text, image = {}, {}
    if parameters["text"].get("freeze") is False:
        text["nouns"] = _PreparedNouns(
            store.annotations("nouns", key="text_id"), parameters["noun_count"]
        )
    if parameters["image"].get("freeze") is False:
        relative = "inputs/crops/five_crop_boxes"
        try:
            metadata = json.loads((store.root / relative / "metadata.json").read_text("utf-8"))
        except (OSError, ValueError) as error:
            raise PreparedInputError("Prepared C3N crop metadata is missing or invalid.") from error
        if metadata.get("shape", [])[1:] != [parameters["crop_count"], 4]:
            raise PreparedInputError("Prepared C3N boxes disagree with crop_count.")
        image["crop_encoder"] = HubEncoder(
            FeatureMatrix(
                store.root,
                {
                    **metadata,
                    "name": relative,
                    "input_kind": "image",
                    "values": f"{relative}/values.npy",
                    "keys": f"{relative}/keys.json",
                    "settings": {"input": relative},
                    **({"mask": f"{relative}/{metadata['mask']}"} if metadata.get("mask") else {}),
                },
            )
        )
    return text, image


class _NounExtractor:
    """Fixed POS resources; selection never reads labels or fits the news corpus."""

    def __init__(self, language):
        if language == "zh":
            import jieba
            import jieba.posseg

            self.tagger = jieba.posseg.POSTokenizer(jieba.Tokenizer())
            with self.tagger.tokenizer.get_dict_file() as dictionary:
                signature = file_digest(dictionary.name)
            self.spec = {"pos": "jieba.posseg", "dictionary_sha256": signature}
        elif language == "en":
            from nltk.tag import PerceptronTagger
            from nltk.tokenize import TreebankWordTokenizer

            try:
                self.tagger = PerceptronTagger(lang="eng")
            except LookupError as error:
                raise ValueError(
                    "C3N requires the fixed English POS resource: "
                    "python -m nltk.downloader averaged_perceptron_tagger_eng"
                ) from error
            self.tokenize = TreebankWordTokenizer().tokenize
            self.spec = {
                "pos": "nltk.averaged_perceptron_tagger_eng",
                "weights_sha256": fingerprint(
                    [self.tagger.model.weights, sorted(self.tagger.classes), self.tagger.tagdict]
                ),
                "tokenization": "TreebankWordTokenizer",
            }
        else:
            raise ValueError("C3N language must be en or zh.")
        self.language = language

    @lru_cache(maxsize=4096)
    def __call__(self, text):
        tagged = (
            ((item.word, item.flag) for item in self.tagger.cut(text))
            if self.language == "zh"
            else self.tagger.tag(self.tokenize(text))
        )
        prefix = "n" if self.language == "zh" else "NN"
        return tuple(word for word, tag in tagged if tag.startswith(prefix) and word.strip())


class _ContentTextEncoder(CLIPTextEncoder):
    packages = (*CLIPTextEncoder.packages, "jieba", "nltk")

    def __init__(self, noun_count, noun_length=20, language="en", nouns=None, **options):
        if noun_count < 1 or noun_length < 2:
            raise ValueError("C3N needs at least one noun slot and noun_length >= 2.")
        super().__init__(**options)
        capacity = self.encoding_spec["config"]["text_config"]["max_position_embeddings"]
        if noun_length > capacity:
            raise ValueError("C3N noun_length exceeds the CLIP text position capacity.")
        self.noun_count, self.noun_length = noun_count, noun_length
        self.nouns = _NounExtractor(language) if nouns is None else nouns
        if nouns is not None:
            self.packages = CLIPTextEncoder.packages
        self.encoding_spec.update(
            content="global_then_nouns_v1",
            noun_count=noun_count,
            noun_length=noun_length,
            language=language,
            noun_extraction=self.nouns.spec,
            padding="encode_empty_string",
        )

    def _encode(self, texts):
        global_features = super()._encode(texts)
        words = []
        for text in texts:
            selected = list(self.nouns(text)[: self.noun_count])
            words.extend(selected + [""] * (self.noun_count - len(selected)))
        inputs = self.tokenizer(
            words,
            padding=True,
            truncation=True,
            max_length=self.noun_length,
            return_tensors="pt",
        ).to(self.device)
        local_features = self._project(inputs).reshape(len(texts), self.noun_count, -1)
        return torch.cat((global_features.unsqueeze(1), local_features), dim=1)


def _select_boxes(boxes, scores, width, height, count, threshold, duplicate_iou):
    selected = []
    for index in scores.argsort(descending=True, stable=True).tolist():
        if not torch.isfinite(scores[index]) or scores[index] < threshold:
            continue
        box = boxes[index].detach().cpu().float()
        if not torch.isfinite(box).all():
            continue
        box[0::2] = box[0::2].clamp(0, width)
        box[1::2] = box[1::2].clamp(0, height)
        box[:2], box[2:] = box[:2].floor(), box[2:].ceil()
        if (box[2:] <= box[:2]).any():
            continue
        if selected:
            previous = torch.stack(selected)
            intersection = (
                (torch.minimum(previous[:, 2:], box[2:]) - torch.maximum(previous[:, :2], box[:2]))
                .clamp_min(0)
                .prod(-1)
            )
            union = (previous[:, 2:] - previous[:, :2]).prod(-1)
            union = union + (box[2:] - box[:2]).prod() - intersection
            if (intersection / union > duplicate_iou).any():
                continue
        selected.append(box)
        if len(selected) == count:
            break
    result = torch.full((count, 4), -1.0)
    if selected:
        result[: len(selected)] = torch.stack(selected)
    return result


class _DetectedCrops(_PretrainedEncoder):
    """Cache small bounding boxes even when the downstream CLIP tower is tuned."""

    input_kind = "image"
    packages = ("torch", "torchvision", "Pillow")

    def __init__(self, count, weights="COCO_V1", threshold=0.5, duplicate_iou=0.95):
        super().__init__()
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        if count < 1 or not 0 <= threshold <= 1 or not 0 <= duplicate_iou <= 1:
            raise ValueError("C3N crop count and detector thresholds are invalid.")
        selected = FasterRCNN_ResNet50_FPN_Weights[weights] if weights is not None else None
        self.network = fasterrcnn_resnet50_fpn(weights=selected, weights_backbone=None)
        self.count, self.threshold, self.duplicate_iou = count, threshold, duplicate_iou
        self.encoding_spec = {
            "encoder": "c3n_fasterrcnn_resnet50_fpn_boxes_v1",
            "weights": weights,
            "count": count,
            "threshold": threshold,
            "duplicate_iou": duplicate_iou,
            "order": "descending_score_stable",
            "coordinates": "original_pixels_floor_ceil_clamped",
            "missing": -1,
            "transform": repr(self.network.transform),
        }
        self._set_frozen(True)

    @torch.no_grad()
    def _encode(self, paths):
        from torchvision.transforms.functional import pil_to_tensor

        results = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                pixels = pil_to_tensor(image).to(self.device).float() / 255
                prediction = self.network([pixels])[0]
                results.append(
                    _select_boxes(
                        prediction["boxes"],
                        prediction["scores"],
                        *image.size,
                        self.count,
                        self.threshold,
                        self.duplicate_iou,
                    )
                )
        return torch.stack(results).to(self.device)


class _ContentImageEncoder(CLIPImageEncoder):
    def __init__(
        self,
        crop_count,
        detector,
        encoding_batch_size=32,
        crop_encoder=None,
        pixel_cache=None,
        **options,
    ):
        if encoding_batch_size < 1:
            raise ValueError("C3N image encoding_batch_size must be positive.")
        super().__init__(**options)
        # Optional exact speed-up: store the resized uint8 crops on disk and apply the
        # processor's own rescale/normalize at load time. It never enters encoding_spec
        # because the produced pixel tensors are bit-identical to the uncached path.
        self.pixel_cache = None if pixel_cache is None else Path(pixel_cache)
        if self.pixel_cache is not None:
            self.pixel_cache.mkdir(parents=True, exist_ok=True)
            self._pixel_cache_tag = hashlib.sha256(
                json.dumps(self.processor.to_dict(), sort_keys=True, default=str).encode()
            ).hexdigest()
        self.crop_encoder = (
            _DetectedCrops(crop_count, **detector) if crop_encoder is None else crop_encoder
        )
        self.crop_count, self.encoding_batch_size = crop_count, encoding_batch_size
        self.encoding_spec.update(
            content="global_then_detected_crops_v1",
            crop_extraction=self.crop_encoder.encoding_spec,
            padding="zero_preprocessed_pixels_then_encode",
        )
        self._refresh_detector_spec(self, None)
        self.register_load_state_dict_post_hook(self._refresh_detector_spec)

    @staticmethod
    def _refresh_detector_spec(module, _incompatible_keys):
        if hasattr(module.crop_encoder, "network"):
            module.encoding_spec["detector_weights_sha256"] = state_digest(
                module.crop_encoder.network
            )

    def _crops(self, path, regions):
        with Image.open(path) as image:
            image = image.convert("RGB")
            return [image, *(image.crop(box.tolist()) for box in regions if box[0] >= 0)]

    def _cached_values(self, path, regions):
        """Resized uint8 crops cached on disk; rescale/normalize reuse the processor's own code."""
        import numpy as np
        from transformers.image_utils import ChannelDimension

        key = hashlib.sha256(
            f"{self._pixel_cache_tag}|{path}|".encode() + regions.numpy().tobytes()
        ).hexdigest()
        file = self.pixel_cache / f"{key[:40]}.npy"
        try:
            resized = np.load(file)
        except (OSError, ValueError):
            resized = np.asarray(
                self.processor(
                    images=self._crops(path, regions),
                    do_rescale=False,
                    do_normalize=False,
                    return_tensors="np",
                )["pixel_values"]
            )
            assert resized.dtype == np.uint8
            temporary = file.with_name(file.name + f".{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, resized)
            temporary.replace(file)
        processor, first = self.processor, ChannelDimension.FIRST
        finished = [
            processor.normalize(
                processor.rescale(image, scale=processor.rescale_factor, input_data_format=first),
                mean=processor.image_mean,
                std=processor.image_std,
                input_data_format=first,
            )
            for image in resized
        ]
        return torch.from_numpy(np.stack(finished))

    def _encode(self, paths):
        boxes = self.crop_encoder(paths).detach().cpu()
        pixels = []
        for path, regions in zip(paths, boxes):
            if self.pixel_cache is not None:
                values = self._cached_values(path, regions)
            else:
                values = self.processor(images=self._crops(path, regions), return_tensors="pt")[
                    "pixel_values"
                ]
            pixels.append(F.pad(values, (0, 0, 0, 0, 0, 0, 0, 1 + self.crop_count - len(values))))
        features = [
            self._project({"pixel_values": chunk.to(self.device)})
            for chunk in torch.cat(pixels).split(self.encoding_batch_size)
        ]
        return torch.cat(features).reshape(len(paths), 1 + self.crop_count, -1)


def _mlp(input_dim, dimensions):
    layers = []
    for dimension in dimensions:
        layers.extend((nn.Linear(input_dim, dimension), nn.ReLU()))
        input_dim = dimension
    return nn.Sequential(*layers)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dimension, heads, text_slots, image_slots, ffn_multiplier):
        super().__init__()
        self.text_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.image_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.text_norms = nn.ModuleList([nn.LayerNorm([text_slots, dimension]) for _ in range(2)])
        self.image_norms = nn.ModuleList([nn.LayerNorm([image_slots, dimension]) for _ in range(2)])
        self.text_ffn = nn.Sequential(
            _mlp(dimension, [ffn_multiplier * dimension]),
            nn.Linear(ffn_multiplier * dimension, dimension),
        )
        self.image_ffn = nn.Sequential(
            _mlp(dimension, [ffn_multiplier * dimension]),
            nn.Linear(ffn_multiplier * dimension, dimension),
        )

    def forward(self, text, image):
        text_delta = self.text_attention(text, image, image, need_weights=False)[0]
        image_delta = self.image_attention(image, text, text, need_weights=False)[0]
        text = self.text_norms[0](text + text_delta)
        image = self.image_norms[0](image + image_delta)
        return (
            self.text_norms[1](text + self.text_ffn(text)),
            self.image_norms[1](image + self.image_ffn(image)),
        )


@register_model("C3N")
class C3N(AbstractDetector):
    """Author-network core, with explicitly adapted POS and object-detection resources."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        sizes = [
            p[key]
            for key in (
                "noun_count",
                "crop_count",
                "num_layers",
                "num_heads",
                "conv_channels",
                "ffn_multiplier",
            )
        ]
        dimensions = [p[key] for key in ("unimodal_dims", "correlation_dims", "fusion_dims")]
        kernels = p["conv_kernels"]
        if (
            any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in sizes
            )
            or any(not values for values in [*dimensions, kernels])
            or any(
                not isinstance(value, int) or value < 1
                for values in [*dimensions, kernels]
                for value in values
            )
            or max(kernels) > p["noun_count"] + 1
        ):
            raise ValueError("C3N convolution kernels and network dimensions are invalid.")
        text_options, image_options = p["text"], p["image"]
        if text_options["model_name"] != image_options["model_name"] or text_options.get(
            "revision", "main"
        ) != image_options.get("revision", "main"):
            raise ValueError("C3N requires paired CLIP checkpoint names and revisions.")
        if text_options.get("normalize", False) or image_options.get("normalize", False):
            raise ValueError("C3N requires unnormalized projected CLIP inputs.")
        if (
            text_options.get("language", "en") not in {"en", "zh"}
            or min(text_options.get("max_length", 77), text_options.get("noun_length", 20)) < 2
            or image_options.get("encoding_batch_size", 32) < 1
        ):
            raise ValueError("C3N input lengths, language, or encoding batch size are invalid.")
        detector_options = image_options["detector"]
        if (
            not 0 <= detector_options.get("threshold", 0.5) <= 1
            or not 0 <= detector_options.get("duplicate_iou", 0.95) <= 1
        ):
            raise ValueError("C3N detector thresholds are invalid.")
        if (
            not 0 <= p["dropout"] < 1
            or not math.isfinite(p["similarity_scale"])
            or p["similarity_scale"] <= 0
        ):
            raise ValueError("C3N dropout and similarity_scale are invalid.")
        text_slots, image_slots = p["noun_count"] + 1, p["crop_count"] + 1
        text_inputs, image_inputs = _prepared_inputs(config)
        self.text_encoder = material_encoder(
            config,
            "text_encoder",
            _ContentTextEncoder,
            noun_count=p["noun_count"],
            **text_inputs,
            **p["text"],
        )
        self.image_encoder = material_encoder(
            config,
            "image_encoder",
            _ContentImageEncoder,
            crop_count=p["crop_count"],
            **image_inputs,
            **p["image"],
        )
        dimension = self.text_encoder.output_dim
        if dimension != self.image_encoder.output_dim or dimension % p["num_heads"]:
            raise ValueError("C3N CLIP projections must match and divide num_heads.")
        self.enhancement = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    dimension, p["num_heads"], text_slots, image_slots, p["ffn_multiplier"]
                )
                for _ in range(p["num_layers"])
            ]
        )
        self.convolutions = nn.ModuleList(
            [nn.Conv2d(1, p["conv_channels"], (kernel, image_slots)) for kernel in kernels]
        )
        self.text_projection = _mlp(dimension, p["unimodal_dims"])
        self.image_projection = _mlp(dimension, p["unimodal_dims"])
        self.correlation_projection = _mlp(len(kernels) * p["conv_channels"], p["correlation_dims"])
        self.fusion = _mlp(2 * p["unimodal_dims"][-1] + p["correlation_dims"][-1], p["fusion_dims"])
        self.classifier = nn.Linear(p["fusion_dims"][-1], 2)
        self.dropout = nn.Dropout(p["dropout"])
        self.similarity_scale = p["similarity_scale"]

    def _correlation(self, text, image):
        similarity = self.similarity_scale * (
            F.normalize(text, dim=-1) @ F.normalize(image, dim=-1).transpose(1, 2)
        )
        return torch.cat(
            [F.relu(conv(similarity.unsqueeze(1))).mean((-2, -1)) for conv in self.convolutions],
            dim=-1,
        )

    def forward(self, batch):
        text = self.text_encoder(batch["text"])
        image = self.image_encoder(batch["image_path"])
        for layer in self.enhancement:
            text, image = layer(text, image)
        branches = (
            self.text_projection(text[:, 0]),
            self.image_projection(image[:, 0]),
            self.correlation_projection(self._correlation(text, image)),
        )
        fused = self.fusion(torch.cat([self.dropout(branch) for branch in branches], dim=-1))
        return self.classifier(self.dropout(fused))

    def calculate_loss(self, batch):
        return F.cross_entropy(self(batch), batch["labels"])

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)
