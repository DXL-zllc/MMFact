"""Shared input encoders; method-specific fusion and losses stay in each model."""

import hashlib
import json
import math
import re
from collections import Counter

import torch
from PIL import Image
from torch import nn


def _word_hash(word):
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "little")


def _module_state_keys(network, module):
    if module is None:
        return set()
    prefix = next(name for name, child in network.named_modules() if child is module)
    return {f"{prefix}.{key}" if prefix else key for key in module.state_dict()}


def _load_pretrained_network(
    model_class, model_name, revision, required_keys=None, **model_options
):
    network, information = model_class.from_pretrained(
        model_name, revision=revision, output_loading_info=True, **model_options
    )
    required = set(network.state_dict()) if required_keys is None else required_keys(network)
    failures = {
        "missing_keys": [key for key in information.get("missing_keys", ()) if key in required],
        "mismatched_keys": information.get("mismatched_keys", ()),
        "error_msgs": information.get("error_msgs", ()),
    }
    failures = {name: values for name, values in failures.items() if values}
    if failures:
        raise ValueError(f"Pretrained encoder has incomplete or incompatible weights: {failures}")
    # Unused pretraining heads are valid extras; active backbone weights must exist.
    return network


class WordEncoder(nn.Module):
    """Fit Word2Vec on training text, and checkpoint both vectors and vocabulary."""

    pad_id, unk_id = 0, 1

    def __init__(
        self,
        dimension=32,
        max_length=100,
        max_vocab_size=20000,
        language="en",
        trainable=True,
        window=5,
        epochs=5,
        seed=42,
    ):
        super().__init__()
        if min(dimension, max_length, window, epochs) < 1 or max_vocab_size < 3:
            raise ValueError(
                "WordEncoder sizes must be positive; vocabulary capacity must be >= 3."
            )
        if language not in {"en", "zh"}:
            raise ValueError("WordEncoder language must be en or zh.")
        self.dimension, self.max_length, self.max_vocab_size = dimension, max_length, max_vocab_size
        self.language, self.window, self.epochs, self.seed = language, window, epochs, seed
        self.embedding = nn.Embedding(max_vocab_size, dimension, padding_idx=self.pad_id)
        self.embedding.requires_grad_(trainable)
        self.vocabulary = {"<pad>": self.pad_id, "<unk>": self.unk_id}
        self.ready = False
        self.preprocessing = None

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def _tokenize(self, text):
        if self.language == "zh":
            import jieba

            if not hasattr(self, "_segmenter"):
                self._segmenter = jieba.Tokenizer()
            return [word for word in self._segmenter.lcut(text) if word.strip()]
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def prepare(self, texts):
        if self.preprocessing is not None:
            self.preprocessing.prepare_words(self, texts)
        else:
            self._fit(texts)

    def _fit(self, texts):
        from gensim.models import Word2Vec

        sentences = [self._tokenize(text)[: self.max_length] for text in texts]
        counts = Counter(word for sentence in sentences for word in sentence)
        words = sorted(counts, key=lambda word: (-counts[word], word))[: self.max_vocab_size - 2]
        if not words:
            raise ValueError("Training text produced an empty vocabulary.")
        self.vocabulary = {"<pad>": self.pad_id, "<unk>": self.unk_id}
        self.vocabulary.update({word: index + 2 for index, word in enumerate(words)})
        sentences = [
            [word for word in sentence if word in self.vocabulary] for sentence in sentences
        ]
        vectors = Word2Vec(
            sentences=sentences,
            vector_size=self.dimension,
            window=self.window,
            min_count=1,
            workers=1,
            epochs=self.epochs,
            seed=self.seed,
            hashfxn=_word_hash,
        ).wv
        values = torch.from_numpy(vectors[words]).to(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[2 : len(words) + 2].copy_(values)
            self.embedding.weight[self.unk_id].copy_(values.mean(dim=0))
        self.ready = True

    def forward(self, texts):
        if not self.ready:
            raise ValueError(
                "Prepare the word encoder on training data or load a checkpoint first."
            )
        ids = (
            self.preprocessing.word_ids(self, texts)
            if self.preprocessing is not None
            else self._token_ids(texts)
        )
        return self.embedding(ids), ids, ids.ne(self.pad_id)

    def _token_ids(self, texts):
        ids = torch.zeros(
            len(texts), self.max_length, dtype=torch.long, device=self.embedding.weight.device
        )
        for index, text in enumerate(texts):
            words = self._tokenize(text)[: self.max_length]
            tokens = [self.vocabulary.get(word, self.unk_id) for word in words] or [self.unk_id]
            ids[index, : len(tokens)] = torch.tensor(tokens, device=ids.device)
        return ids

    def get_extra_state(self):
        return {"vocabulary": self.vocabulary, "ready": self.ready}

    def set_extra_state(self, state):
        self.vocabulary, self.ready = state["vocabulary"], state["ready"]
        if self.preprocessing is not None:
            self.preprocessing.reset()


class _PretrainedEncoder(nn.Module):
    """Frozen feature extractors also keep dropout and normalization in eval mode."""

    input_kind = "text"
    cache_features = True
    variable_inputs = False
    packages = ("torch", "transformers", "tokenizers")

    def __init__(self):
        super().__init__()
        self.preprocessing = None
        self.register_buffer("_device", torch.empty(0), persistent=False)
        self.register_load_state_dict_post_hook(self._clear_cache)

    @staticmethod
    def _clear_cache(module, _incompatible_keys):
        if module.preprocessing is not None:
            module.preprocessing.reset()

    @staticmethod
    def _load_network(model_class, model_name, revision):
        return _load_pretrained_network(model_class, model_name, revision)

    def _set_frozen(self, freeze):
        self.freeze = freeze
        self.network.requires_grad_(not freeze)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.network.eval()
        return self

    @property
    def device(self):
        return next(self.network.parameters(), self._device).device

    def forward(self, inputs):
        if self.freeze and self.preprocessing is not None:
            return self.preprocessing.features(self, inputs)
        return self._encode(inputs)

    def inputs_from_rows(self, rows):
        """Select fixed, permitted inputs; do not fit resources or inspect labels."""
        key = "image_path" if self.input_kind == "image" else "text"
        return [row[key] for row in rows]


class RGBEncoder(_PretrainedEncoder):
    """Fixed RGB pixels with explicit resize, crop and normalization rules."""

    input_kind = "image"
    packages = ("torch", "torchvision", "Pillow")

    def __init__(
        self,
        image_size=224,
        minimum_size=64,
        center_crop=False,
        resize_size=256,
        normalize=False,
        mean=None,
        std=None,
    ):
        super().__init__()
        for name, value, minimum in (
            ("image_size", image_size, 1),
            ("minimum_size", minimum_size, 0),
            ("resize_size", resize_size, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"RGB {name} must be an integer >= {minimum}.")
        if not isinstance(center_crop, bool) or not isinstance(normalize, bool):
            raise ValueError("RGB center_crop and normalize must be boolean.")
        if center_crop and resize_size < image_size:
            raise ValueError("RGB resize_size must be >= image_size for center crop.")
        if (mean is None) != (std is None):
            raise ValueError("RGB mean and std must be provided together.")
        self.mean = tuple((0.485, 0.456, 0.406) if mean is None else mean)
        self.std = tuple((0.229, 0.224, 0.225) if std is None else std)
        if (
            len(self.mean) != 3
            or len(self.std) != 3
            or any(not math.isfinite(value) for value in (*self.mean, *self.std))
            or any(value <= 0 for value in self.std)
        ):
            raise ValueError("RGB mean/std must contain three finite values; std must be positive.")
        self.network = nn.Identity()
        self.image_size, self.minimum_size = image_size, minimum_size
        self.center_crop, self.resize_size, self.normalize = center_crop, resize_size, normalize
        self.encoding_spec = {
            "encoder": "bmr-rgb",
            "image_size": image_size,
            "minimum_size": minimum_size,
            "resize": "PIL-bilinear",
            "range": [0, 1],
            "normalize": False,
            "small_image": "zero-before-resize",
        }
        # Preserve existing BMR/MIMoE cache identities when processing is unchanged.
        if center_crop:
            self.encoding_spec.update(
                resize="torchvision-shortest-edge-PIL-bilinear",
                resize_size=resize_size,
                center_crop=image_size,
            )
        if normalize:
            self.encoding_spec.pop("range")
            self.encoding_spec.update(
                normalize=True,
                pre_normalization_range=[0, 1],
                mean=list(self.mean),
                std=list(self.std),
            )
        self._set_frozen(True)

    def _encode(self, paths):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import (
            center_crop,
            normalize,
            pil_to_tensor,
            resize,
        )

        pixels = []
        for path in paths:
            with Image.open(path) as image:
                if min(image.size) < self.minimum_size:
                    value = torch.zeros(3, self.image_size, self.image_size)
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
                    value = pil_to_tensor(image).float().div(255)
                if self.normalize:
                    value = normalize(value, self.mean, self.std)
                pixels.append(value)
        return torch.stack(pixels).to(self.device)


class MAETokenEncoder(_PretrainedEncoder):
    """All ordered MAE tokens; optionally augment pixels during training."""

    input_kind = "image"
    packages = ("torch", "torchvision", "transformers", "Pillow")

    def __init__(
        self,
        model_name="facebook/vit-mae-base",
        revision="main",
        freeze=True,
        image_size=224,
        minimum_size=64,
        augment=True,
        color_jitter=(0.2, 0.2, 0.2, 0.1),
        flip_probability=0.5,
        center_crop=False,
        resize_size=256,
        normalize=False,
        mean=None,
        std=None,
        output="tokens",
    ):
        super().__init__()
        from torchvision import transforms
        from transformers import ViTMAEModel

        if output not in {"tokens", "patches", "cls"}:
            raise ValueError("MAE output must be tokens, patches, or cls.")
        self.pixels = RGBEncoder(
            image_size, minimum_size, center_crop, resize_size, normalize, mean, std
        )
        self.network = self._load_network(ViTMAEModel, model_name, revision)
        if self.network.config.image_size != image_size:
            raise ValueError("image_size must match the MAE checkpoint.")
        self.network.config.mask_ratio = 0.0
        self.output_dim = self.network.config.hidden_size
        self.patch_count = (image_size // self.network.config.patch_size) ** 2
        self.sequence_length = {
            "tokens": self.patch_count + 1,
            "patches": self.patch_count,
            "cls": 1,
        }[output]
        self.augment, self.output = augment, output
        self.augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(flip_probability),
                transforms.ColorJitter(*color_jitter),
            ]
        )
        self.encoding_spec = {
            "encoder": "bmr-mae-tokens",
            "model_name": str(model_name),
            "revision": getattr(self.network.config, "_commit_hash", None) or revision,
            "config": self.network.config.to_dict(),
            "pixels": self.pixels.encoding_spec,
            "output": "last_hidden_state",
            "patch_order": "identity-noise-no-masking",
            "augmentation": "none-for-fixed-eval-features",
        }
        if output != "tokens":
            self.encoding_spec["output"] = f"last_hidden_state-{output}"
        self._set_frozen(freeze)

    def _tokens(self, values):
        # HF MAE shuffles even at mask_ratio=0 unless noise explicitly fixes the order.
        noise = torch.arange(self.patch_count, device=values.device).expand(len(values), -1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            tokens = self.network(pixel_values=values, noise=noise).last_hidden_state
            if self.output == "cls":
                return tokens[:, 0]
            return tokens[:, 1:] if self.output == "patches" else tokens

    def _encode(self, paths):
        return self._tokens(self.pixels(paths))

    def forward(self, paths):
        if self.training and self.augment:
            pixels = self.pixels(paths)
            if self.pixels.normalize:
                mean = pixels.new_tensor(self.pixels.mean)[:, None, None]
                std = pixels.new_tensor(self.pixels.std)[:, None, None]
                empty = (pixels == -mean / std).flatten(1).all(1)
                # Color transforms operate in RGB space, before normalization.
                pixels = (pixels * std + mean).clamp(0, 1)
                pixels = pixels.masked_fill(empty[:, None, None, None], 0)
            # Random transforms are per image; a missing/small image remains exactly zero.
            values = torch.stack(
                [self.augmentation(image) if image.count_nonzero() else image for image in pixels]
            )
            if self.pixels.normalize:
                values = (values - mean) / std
            return self._tokens(values)
        return super().forward(paths)


class _ImageEncoder(_PretrainedEncoder):
    input_kind = "image"
    packages = ("torch", "torchvision", "Pillow")

    def _encode(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        inputs = torch.stack(images).to(self.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            return self.network(inputs)


class VGGEncoder(_ImageEncoder):
    """ImageNet VGG-19: fc2 before/after ReLU, or final logits."""

    def __init__(
        self,
        output="fc2",
        freeze=True,
        weights="IMAGENET1K_V1",
        center_crop=True,
        normalize=True,
        mean=None,
        std=None,
    ):
        super().__init__()
        from torchvision import transforms
        from torchvision.models import VGG19_Weights, vgg19

        if output not in {"fc2", "fc2_linear", "logits"}:
            raise ValueError("VGG output must be fc2, fc2_linear, or logits.")
        if (mean is None) != (std is None):
            raise ValueError("VGG mean and std must be provided together.")
        selected = VGG19_Weights[weights] if weights is not None else None
        self.network = vgg19(weights=selected)
        if output in {"fc2", "fc2_linear"}:
            self.output_dim = self.network.classifier[3].out_features
            self.network.classifier = self.network.classifier[: 5 if output == "fc2" else 4]
        else:
            self.output_dim = self.network.classifier[-1].out_features
        transform = (selected or VGG19_Weights.IMAGENET1K_V1).transforms()
        if not center_crop or not normalize or mean is not None:
            steps = [
                transforms.Resize(
                    256 if center_crop else (224, 224), interpolation=transform.interpolation
                ),
            ]
            if center_crop:
                steps.append(transforms.CenterCrop(224))
            steps.append(transforms.ToTensor())
            if normalize:
                steps.append(
                    transforms.Normalize(
                        mean=transform.mean if mean is None else mean,
                        std=transform.std if std is None else std,
                    )
                )
            transform = transforms.Compose(steps)
        self.transform = transform
        self.encoding_spec = {
            "encoder": "vgg19",
            "weights": weights,
            "output": output,
            "transform": repr(transform),
        }
        self._set_frozen(freeze)


class ResNetEncoder(_ImageEncoder):
    """ImageNet ResNet pooled, spatial, or original classification features."""

    def __init__(
        self,
        freeze=True,
        weights="IMAGENET1K_V1",
        architecture="resnet34",
        output="avgpool",
        stem_maxpool=True,
        center_crop=True,
        grid_size=4,
    ):
        super().__init__()
        from torchvision import models

        if architecture not in {"resnet18", "resnet34", "resnet50", "resnet101"}:
            raise ValueError(
                "ResNet architecture must be resnet18, resnet34, resnet50, or resnet101."
            )
        if output not in {"avgpool", "logits", "spatial"}:
            raise ValueError("ResNet output must be avgpool, logits, or spatial.")
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError("ResNet grid_size must be a positive integer.")
        if not isinstance(stem_maxpool, bool) or not isinstance(center_crop, bool):
            raise ValueError("ResNet stem_maxpool and center_crop must be boolean.")
        weight_type = getattr(models, f"ResNet{architecture.removeprefix('resnet')}_Weights")
        selected = weight_type[weights] if weights is not None else None
        self.network = getattr(models, architecture)(weights=selected)
        if not stem_maxpool:
            self.network.maxpool = nn.Identity()
        self.output_dim = (
            self.network.fc.out_features if output == "logits" else self.network.fc.in_features
        )
        if output == "avgpool":
            self.network.fc = nn.Identity()
        elif output == "spatial":
            self.network = nn.Sequential(
                *list(self.network.children())[:-2], nn.AdaptiveAvgPool2d((grid_size, grid_size))
            )
        transform = (selected or weight_type.IMAGENET1K_V1).transforms()
        if not center_crop:
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224), interpolation=transform.interpolation),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=transform.mean, std=transform.std),
                ]
            )
        self.transform = transform
        self.encoding_spec = {
            "encoder": architecture,
            "weights": weights,
            "output": output,
            "transform": repr(self.transform),
        }
        if not stem_maxpool:
            self.encoding_spec["stem_maxpool"] = False
        if output == "spatial":
            self.encoding_spec.update(pooling="layer4-adaptive-mean", grid_size=grid_size)
        self._set_frozen(freeze)


class BertEncoder(_PretrainedEncoder):
    """PLM outputs; explicit Longformer views retain global CLS attention."""

    def _required_weights(self, network):
        if self.output == "embeddings":
            return _module_state_keys(network, network.get_input_embeddings())
        required = set(network.state_dict())
        if self.output != "pooled":
            required -= _module_state_keys(network, getattr(network, "pooler", None))
        if self.output == "layer_groups":
            layers = getattr(getattr(network, "encoder", None), "layer", ())
            for layer in layers[max(max(group) for group in self.layer_groups) :]:
                required -= _module_state_keys(network, layer)
        return required

    def _load_network(self, model_class, model_name, revision):
        # Longformer's MLM checkpoint has no trained pooler. Excluding it also
        # prevents random unused pooler weights from invalidating shared caches.
        options = {"add_pooling_layer": False} if self.global_attention is not None else {}
        return _load_pretrained_network(
            model_class, model_name, revision, self._required_weights, **options
        )

    def __init__(
        self,
        model_name="bert-base-uncased",
        freeze=True,
        max_length=200,
        revision="main",
        output="pooled",
        mask_padding=False,
        layer_groups=None,
        global_attention=None,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        if output not in {
            "pooled",
            "cls",
            "tokens",
            "flatten",
            "cls_layers",
            "embeddings",
            "layer_groups",
            "mean",
        }:
            raise ValueError("Unsupported BERT output mode.")
        if global_attention not in (None, "cls"):
            raise ValueError("PLM global_attention must be null or cls.")
        if output == "layer_groups":
            if (
                not isinstance(layer_groups, (list, tuple))
                or not layer_groups
                or any(
                    not isinstance(group, (list, tuple))
                    or not group
                    or any(
                        isinstance(layer, bool) or not isinstance(layer, int) or layer < 1
                        for layer in group
                    )
                    or len(set(group)) != len(group)
                    for group in layer_groups
                )
            ):
                raise ValueError(
                    "BERT layer_groups must contain nonempty groups of unique positive block indices."
                )
        elif layer_groups is not None:
            raise ValueError("BERT layer_groups requires output='layer_groups'.")
        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("BERT max_length must be an integer >= 2.")
        self.output, self.global_attention = output, global_attention
        self.layer_groups = tuple(tuple(group) for group in layer_groups or ())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.network = self._load_network(AutoModel, model_name, revision)
        is_longformer = self.network.config.model_type == "longformer"
        if is_longformer:
            if global_attention != "cls" or output not in {"cls", "mean", "tokens"}:
                raise ValueError(
                    "Longformer requires global_attention='cls' and cls, mean, or tokens output."
                )
            if self.tokenizer.padding_side != "right":
                raise ValueError("Longformer CLS views require right padding.")
        elif global_attention is not None:
            raise ValueError("global_attention is only supported by Longformer.")
        if any(max(group) > self.network.config.num_hidden_layers for group in self.layer_groups):
            raise ValueError("BERT layer_groups exceeds the checkpoint's Transformer block count.")
        capacity = self.network.config.max_position_embeddings
        if is_longformer:
            capacity -= self.network.config.pad_token_id + 1
        if max_length > capacity:
            raise ValueError("BERT max_length exceeds the checkpoint's position capacity.")
        self.max_length = max_length
        self.output_dim = self.network.config.hidden_size
        self.output, self.mask_padding = output, mask_padding
        self.sequence_length = (
            self.network.config.num_hidden_layers + 1 if output == "cls_layers" else max_length
        )
        if output == "flatten":
            self.output_dim *= max_length
        self.encoding_spec = {
            "encoder": "bert",
            "model_name": str(model_name),
            "revision": getattr(self.network.config, "_commit_hash", None) or revision,
            "config": self.network.config.to_dict(),
            "max_length": max_length,
            "pooling": "pooler_output" if output == "pooled" else output,
            "tokenizer": json.loads(self.tokenizer.backend_tokenizer.to_str()),
            "special_tokens": self.tokenizer.special_tokens_map,
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
        }
        if output in {"tokens", "flatten", "embeddings", "layer_groups"}:
            self.encoding_spec.update(padding="max_length", mask_padding=mask_padding)
        if output == "layer_groups":
            self.encoding_spec.update(
                layer_groups=[list(group) for group in self.layer_groups],
                pooling="sum-transformer-blocks-excluding-input-embedding",
            )
        if output == "embeddings":
            self.encoding_spec["pooling"] = "input-word-embeddings-no-position-type-or-norm"
        if output == "mean":
            self.encoding_spec["pooling"] = "masked-mean-including-special-tokens"
        if is_longformer:
            self.encoding_spec.update(encoder="longformer", global_attention="first-token-cls")
        self._set_frozen(freeze)

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length"
            if self.output in {"tokens", "flatten", "embeddings", "layer_groups"}
            else True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def token_mask(self, texts):
        """Return the exact tokenizer mask, including retained special tokens."""
        return self._tokenize(texts)["attention_mask"].to(self.device).bool()

    def _encode(self, texts):
        inputs = self._tokenize(texts).to(self.device)
        if self.global_attention == "cls":
            global_mask = torch.zeros_like(inputs["attention_mask"])
            global_mask[:, 0] = 1
            inputs["global_attention_mask"] = global_mask
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            if self.output == "embeddings":
                tokens = self.network.get_input_embeddings()(inputs["input_ids"])
                if self.mask_padding:
                    tokens = tokens * inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
                return tokens
            result = self.network(
                **inputs, output_hidden_states=self.output in {"cls_layers", "layer_groups"}
            )
            if self.output == "pooled":
                if result.pooler_output is None:
                    raise ValueError("BERT pooled output requires a checkpoint with a pooler.")
                return result.pooler_output
            if self.output == "cls_layers":
                return torch.stack([layer[:, 0] for layer in result.hidden_states], dim=1)
            if self.output == "layer_groups":
                groups = torch.stack(
                    [
                        torch.stack([result.hidden_states[i] for i in group]).sum(0)
                        for group in self.layer_groups
                    ],
                    dim=1,
                )
                if self.mask_padding:
                    groups = groups * inputs["attention_mask"][:, None, :, None].to(groups.dtype)
                return groups
            if self.output == "cls":
                return result.last_hidden_state[:, 0]
            if self.output == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).to(result.last_hidden_state.dtype)
                return (result.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
            tokens = result.last_hidden_state
            if self.mask_padding:
                tokens = tokens * inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
            return tokens.flatten(1) if self.output == "flatten" else tokens


class SwinTokenEncoder(_PretrainedEncoder):
    """Normalized final Swin patch tokens, using the checkpoint's native processor."""

    input_kind = "image"
    packages = ("torch", "transformers", "torchvision", "Pillow")

    def __init__(
        self,
        model_name="microsoft/swin-base-patch4-window7-224",
        revision="main",
        freeze=False,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, SwinModel

        self.processor = AutoImageProcessor.from_pretrained(
            model_name, revision=revision, use_fast=False
        )
        self.network = SwinModel.from_pretrained(model_name, revision=revision)
        config = self.network.config
        self.output_dim = config.hidden_size
        sizes = (
            (config.image_size,) * 2 if isinstance(config.image_size, int) else config.image_size
        )
        patches = (
            (config.patch_size,) * 2 if isinstance(config.patch_size, int) else config.patch_size
        )
        reduction = 2 ** (len(config.depths) - 1)
        self.sequence_length = math.prod(
            math.ceil(size / patch / reduction) for size, patch in zip(sizes, patches)
        )
        self.encoding_spec = {
            "encoder": "swin-tokens",
            "model_name": str(model_name),
            "revision": getattr(config, "_commit_hash", None) or revision,
            "config": config.to_dict(),
            "processor": self.processor.to_dict(),
            "output": "last_hidden_state",
            "cls_token": False,
        }
        self._set_frozen(freeze)

    def _encode(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            return self.network(**inputs).last_hidden_state


class _CLIPEncoder(_PretrainedEncoder):
    """Keep only one CLIP tower and its pretrained projection in checkpoints."""

    def _load(self, model_name, revision, freeze, normalize, output="pooled"):
        from transformers import AutoModel

        if output not in {"pooled", "tokens"}:
            raise ValueError("CLIP output must be pooled or tokens.")
        model = AutoModel.from_pretrained(model_name, revision=revision)
        if model.config.model_type not in {"clip", "chinese_clip"}:
            raise ValueError("CLIP encoders require a CLIP or Chinese-CLIP checkpoint.")
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
        if output == "tokens":
            self.encoding_spec["output"] = "projected_last_hidden_state"
        self._set_frozen(freeze)

    def _project(self, inputs):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            result = self.network["encoder"](**inputs)
            # Chinese-CLIP has no BERT pooler: its joint forward projects raw CLS.
            if self.output == "tokens":
                values = result.last_hidden_state
            else:
                values = (
                    result.last_hidden_state[:, 0] if self._cls_pooling else result.pooler_output
                )
            projected = self.network["projection"](values)
            return nn.functional.normalize(projected, dim=-1) if self.normalize else projected


class CLIPTextEncoder(_CLIPEncoder):
    """Pretrained projected text features; normalization is an explicit choice."""

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        freeze=True,
        max_length=77,
        revision="main",
        normalize=False,
        output="pooled",
        mask_padding=True,
    ):
        super().__init__()
        from transformers import AutoTokenizer

        if not isinstance(mask_padding, bool):
            raise ValueError("CLIP mask_padding must be boolean.")
        self._load(model_name, revision, freeze, normalize, output)
        if not isinstance(max_length, int) or not (
            2
            <= max_length
            <= self.encoding_spec["config"]["text_config"]["max_position_embeddings"]
        ):
            raise ValueError("CLIP text length must fit the checkpoint's position capacity.")
        self.max_length = max_length
        self.mask_padding = mask_padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.encoding_spec.update(
            max_length=max_length,
            tokenizer=json.loads(self.tokenizer.backend_tokenizer.to_str()),
            special_tokens=self.tokenizer.special_tokens_map,
            padding_side=self.tokenizer.padding_side,
            truncation_side=self.tokenizer.truncation_side,
        )
        if output == "tokens":
            self.sequence_length = max_length
            self.encoding_spec.update(padding="max_length", mask_padding=mask_padding)

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length" if self.output == "tokens" else True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def token_mask(self, texts):
        """Use tokenizer masks, including special tokens; never infer padding from features."""
        return self._tokenize(texts)["attention_mask"].to(self.device).bool()

    def _encode(self, texts):
        inputs = self._tokenize(texts).to(self.device)
        result = self._project(inputs)
        if self.output == "tokens" and self.mask_padding:
            result = result * inputs["attention_mask"].unsqueeze(-1).to(result.dtype)
        return result


class CLIPImageEncoder(_CLIPEncoder):
    """Use the selected checkpoint's image processor and projected vision pooler."""

    input_kind = "image"
    packages = ("torch", "transformers", "torchvision", "Pillow")

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        freeze=True,
        revision="main",
        normalize=False,
        square_resize=False,
        output="pooled",
    ):
        super().__init__()
        from transformers import AutoImageProcessor

        if not isinstance(square_resize, bool):
            raise ValueError("CLIP square_resize must be boolean.")
        self._load(model_name, revision, freeze, normalize, output)
        if output == "tokens":
            vision = self.encoding_spec["config"]["vision_config"]
            self.sequence_length = (vision["image_size"] // vision["patch_size"]) ** 2 + 1
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, revision=revision, use_fast=False
        )
        if square_resize:
            size = self.encoding_spec["config"]["vision_config"]["image_size"]
            self.processor.size = {"height": size, "width": size}
            self.processor.do_resize = True
            self.processor.do_center_crop = False
        self.encoding_spec["processor"] = self.processor.to_dict()

    def _encode(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(
            images=images, input_data_format="channels_last", return_tensors="pt"
        ).to(self.device)
        return self._project(inputs)


class SentenceEncoder(_PretrainedEncoder):
    """Ordered sentence means from a pretrained PLM; padded sentences stay zero.

    The release leaves token-to-sentence pooling unspecified. Use masked token
    means (including special tokens) and deterministic punctuation splitting.
    """

    packages = ("torch", "transformers", "tokenizers", "sentencepiece")

    def __init__(
        self,
        model_name="google-bert/bert-base-uncased",
        freeze=True,
        max_length=512,
        revision="main",
        max_sentences=50,
        language="en",
        sentence_batch_size=16,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        if min(max_length, max_sentences, sentence_batch_size) < 1:
            raise ValueError("Sentence input lengths and sentence_batch_size must be positive.")
        if language not in {"en", "zh"}:
            raise ValueError("Sentence language must be en or zh.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.network = _load_pretrained_network(
            AutoModel,
            model_name,
            revision,
            lambda network: (
                set(network.state_dict())
                - _module_state_keys(network, getattr(network, "pooler", None))
            ),
        )
        self.output_dim = self.network.config.hidden_size
        self.max_length, self.max_sentences = max_length, max_sentences
        self.language, self.sentence_batch_size = language, sentence_batch_size
        self.encoding_spec = {
            "encoder": "ordered-sentence-plm",
            "model_name": str(model_name),
            "revision": getattr(self.network.config, "_commit_hash", None) or revision,
            "config": self.network.config.to_dict(),
            "max_length": max_length,
            "max_sentences": max_sentences,
            "language": language,
            "sentence_splitter": "punctuation-v1",
            "pooling": "masked_mean_including_special_tokens",
            "tokenizer": json.loads(self.tokenizer.backend_tokenizer.to_str()),
            "special_tokens": self.tokenizer.special_tokens_map,
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
        }
        self._set_frozen(freeze)

    def _sentences(self, text):
        pattern = r"(?<=[。！？!?])\s*|\n+" if self.language == "zh" else r"(?<=[.!?])\s+|\n+"
        return [part.strip() for part in re.split(pattern, text) if part.strip()][
            : self.max_sentences
        ] or [""]

    def _encode(self, texts):
        sentences = [self._sentences(text) for text in texts]
        flat = [sentence for article in sentences for sentence in article]
        features = []
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            for start in range(0, len(flat), self.sentence_batch_size):
                inputs = self.tokenizer(
                    flat[start : start + self.sentence_batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                # No memory is passed between sentences/articles.
                options = {"use_mems": False} if self.network.config.model_type == "xlnet" else {}
                hidden = self.network(**inputs, **options).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                features.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
            encoded = torch.cat(features)
            chunks = encoded.split([len(article) for article in sentences])
            return torch.stack(
                [
                    torch.nn.functional.pad(chunk, (0, 0, 0, self.max_sentences - len(chunk)))
                    for chunk in chunks
                ]
            )


class XLNetEncoder(SentenceEncoder):
    """Compatibility name for explicitly configured XLNet sentence resources."""

    def __init__(self, model_name="xlnet-base-cased", **options):
        super().__init__(model_name=model_name, **options)
        self.encoding_spec["encoder"] = "xlnet"
