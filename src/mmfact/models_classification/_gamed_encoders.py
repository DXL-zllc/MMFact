"""GAMED's ordered semantic tokens and trainable image-pattern backbone."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.models_classification._encoders import BertEncoder, MAETokenEncoder


def _load_analyzer(model_class, model_name, revision, **options):
    network, information = model_class.from_pretrained(
        model_name, revision=revision, output_loading_info=True, **options
    )
    failures = {
        name: information[name]
        for name in ("missing_keys", "mismatched_keys", "error_msgs")
        if information.get(name)
    }
    if failures:
        raise ValueError(
            f"GAMED pretrained analyzer has incomplete or incompatible weights: {failures}"
        )
    # Pretraining decoder, MLM and unused pooler keys may legitimately remain unused.
    return network


def _validate_resource(model_name, revision):
    if not isinstance(model_name, (str, Path)) or not str(model_name).strip():
        raise ValueError("GAMED analyzer model_name must be a nonempty model ID or local path.")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("GAMED analyzer revision must be a nonempty string.")


def _validate_text_options(model_name, revision, freeze, max_length):
    _validate_resource(model_name, revision)
    if type(freeze) is not bool or type(max_length) is not int or max_length < 2:
        raise ValueError("GAMED text requires boolean freeze and integer max_length >= 2.")


def _validate_image_options(
    model_name,
    revision,
    freeze,
    image_size,
    minimum_size,
    augment,
    rotation_degrees,
    scale,
    flip_probability,
    center_crop,
    resize_size,
    normalize,
):
    _validate_resource(model_name, revision)
    if (
        type(freeze) is not bool
        or type(image_size) is not int
        or image_size < 1
        or type(minimum_size) is not int
        or minimum_size < 0
    ):
        raise ValueError(
            "GAMED image requires boolean freeze, image_size > 0 and minimum_size >= 0."
        )
    if not isinstance(augment, bool):
        raise ValueError("GAMED image augment must be boolean.")
    if (
        type(center_crop) is not bool
        or type(resize_size) is not int
        or resize_size < image_size
        or normalize is not True
    ):
        raise ValueError("GAMED MAE needs a valid fixed crop and ImageNet normalization.")
    if (
        isinstance(rotation_degrees, bool)
        or not isinstance(rotation_degrees, (int, float))
        or not math.isfinite(rotation_degrees)
        or rotation_degrees < 0
    ):
        raise ValueError("GAMED image rotation_degrees must be finite and nonnegative.")
    if (
        not isinstance(scale, (list, tuple))
        or len(scale) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in scale
        )
        or scale[0] > scale[1]
    ):
        raise ValueError("GAMED image scale must contain two ordered positive finite values.")
    if (
        isinstance(flip_probability, bool)
        or not isinstance(flip_probability, (int, float))
        or not math.isfinite(flip_probability)
        or not 0 <= flip_probability <= 1
    ):
        raise ValueError("GAMED image flip_probability must be finite and in [0,1].")


def _validate_pattern_options(model_name, revision, pretrained, checkpoint, epsilon):
    if model_name != "inception_resnet_v2.tf_in1k":
        raise ValueError("GAMED pattern requires inception_resnet_v2.tf_in1k.")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("GAMED pattern revision must be a nonempty string.")
    if not isinstance(pretrained, bool):
        raise ValueError("GAMED pattern pretrained must be boolean.")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("GAMED pattern epsilon must be finite and positive.")
    if checkpoint is not None and (
        not isinstance(checkpoint, (str, Path)) or not Path(checkpoint).is_file()
    ):
        raise ValueError("GAMED pattern checkpoint must be an existing local weight file.")


def _validate_gamed_analyzers(text, image, pattern):
    """Validate all merged analyzer settings before loading the first pretrained tower."""
    expected = (
        {"model_name", "revision", "freeze", "max_length"},
        {
            "model_name",
            "revision",
            "freeze",
            "image_size",
            "minimum_size",
            "augment",
            "rotation_degrees",
            "scale",
            "flip_probability",
            "center_crop",
            "resize_size",
            "normalize",
        },
        {"model_name", "revision", "pretrained", "checkpoint", "epsilon"},
    )
    for name, options, keys in zip(("text", "image", "pattern"), (text, image, pattern), expected):
        if not isinstance(options, dict) or set(options) != keys:
            raise ValueError(f"GAMED {name} analyzer settings must match the configured interface.")
    _validate_text_options(**text)
    _validate_image_options(**image)
    _validate_pattern_options(**pattern)
    if text["freeze"] is not True or image["freeze"] is not True:
        raise ValueError("GAMED requires frozen text and MAE analyzers.")


def _pixels(values):
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 4
        or not values.is_floating_point()
        or values.shape[1] != 3
        or min(values.shape) < 1
        or not torch.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("GAMED requires finite RGB pixels [B,3,H,W] in [0,1].")
    return values


class _GAMEDTextTokens(BertEncoder):
    """Fixed-length semantic tokens with the tokenizer's exact padding mask."""

    @staticmethod
    def _load_network(model_class, model_name, revision):
        return _load_analyzer(model_class, model_name, revision, add_pooling_layer=False)

    def __init__(
        self,
        model_name="google-bert/bert-base-uncased",
        revision="86b5e0934494bd15c9632b12f734a8a67f723594",
        freeze=True,
        max_length=197,
    ):
        _validate_text_options(model_name, revision, freeze, max_length)
        super().__init__(
            model_name=model_name,
            revision=revision,
            freeze=freeze,
            max_length=max_length,
            output="tokens",
            mask_padding=True,
        )

    def token_mask(self, texts):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return inputs["attention_mask"].to(device=self.device, dtype=torch.bool)


class _GAMEDImageTokens(MAETokenEncoder):
    """Augment only IS; cache deterministic normalized, unmasked MAE tokens."""

    @staticmethod
    def _load_network(model_class, model_name, revision):
        return _load_analyzer(model_class, model_name, revision)

    def __init__(
        self,
        model_name="facebook/vit-mae-base",
        revision="25b184bea5538bf5c4c852c79d221195fdd2778d",
        freeze=True,
        image_size=224,
        minimum_size=0,
        augment=True,
        rotation_degrees=10.0,
        scale=(0.9, 1.1),
        flip_probability=0.5,
        center_crop=True,
        resize_size=256,
        normalize=True,
    ):
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        _validate_image_options(
            model_name,
            revision,
            freeze,
            image_size,
            minimum_size,
            augment,
            rotation_degrees,
            scale,
            flip_probability,
            center_crop,
            resize_size,
            normalize,
        )
        super().__init__(
            model_name=model_name,
            revision=revision,
            freeze=freeze,
            image_size=image_size,
            minimum_size=minimum_size,
            augment=False,
            normalize=False,
            center_crop=center_crop,
            resize_size=resize_size,
        )
        self.pixels.cache_features = False
        self.augment = augment
        self.augmentation = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=rotation_degrees,
                    scale=tuple(scale),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.RandomHorizontalFlip(flip_probability),
            ]
        )
        self.register_buffer(
            "_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False
        )
        # Describe the effective fixed view, even though augmentation needs raw RGB.
        self.encoding_spec["pixels"] = {
            **self.pixels.encoding_spec,
            "normalize": True,
            "pre_normalization_range": [0, 1],
            "mean": list(self.pixels.mean),
            "std": list(self.pixels.std),
        }
        self.encoding_spec["pixels"].pop("range")

    def view(self, paths, pixels=None):
        values = _pixels(self.pixels(paths) if pixels is None else pixels)
        if values.shape[0] != len(paths) or values.shape[-2:] != (
            self.pixels.image_size,
            self.pixels.image_size,
        ):
            raise ValueError("GAMED image views must match the input paths and MAE image size.")
        if self.training and self.augment:
            return torch.stack([self.augmentation(image) for image in values])
        return values

    def encode_view(self, pixels):
        values = _pixels(pixels)
        if values.shape[-2:] != (self.pixels.image_size, self.pixels.image_size):
            raise ValueError("GAMED image view size must match the MAE checkpoint.")
        values = values.to(next(self.network.parameters()))
        return self._tokens((values - self._mean) / self._std)

    def _encode(self, paths):
        # Cache construction always encodes the fixed view, even under parent train().
        return self.encode_view(self.pixels(paths))

    def forward(self, paths, pixels=None):
        if self.training and self.augment:
            return self.encode_view(self.view(paths, pixels))
        if self.freeze and self.preprocessing is not None:
            return super().forward(paths)
        return self.encode_view(self.view(paths, pixels))


class _GAMEDBayarConv(nn.Module):
    """Keep each 5x5 center at -1 and its surrounding 24 coefficients summing to 1."""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.rand(3, 3, 5, 5))
        mask = torch.ones(1, 1, 5, 5)
        mask[..., 2, 2] = 0
        self.register_buffer("mask", mask)

    def kernel(self):
        surroundings = self.weight * self.mask
        total = surroundings.sum((-2, -1), keepdim=True)
        valid = total.abs() >= self.epsilon
        denominator = torch.where(valid, total, 1)
        surroundings = torch.where(valid, surroundings / denominator, self.mask / 24)
        return surroundings - (1 - self.mask)

    def forward(self, pixels):
        return F.conv2d(pixels, self.kernel(), padding=2)


class _GAMEDPattern(nn.Module):
    """Early Bayar filtering followed by the complete trainable Inception-ResNet-v2."""

    def __init__(
        self,
        model_name="inception_resnet_v2.tf_in1k",
        revision="836b07d7d599da247de92f148c7a07fda18afced",
        pretrained=True,
        checkpoint=None,
        epsilon=1e-6,
    ):
        super().__init__()
        _validate_pattern_options(model_name, revision, pretrained, checkpoint, epsilon)
        import timm

        load_weights = pretrained or checkpoint is not None
        overlay = (
            {"file": str(checkpoint)}
            if checkpoint is not None
            else {
                "hf_hub_id": f"timm/{model_name}@{revision}",
                "hf_hub_filename": "model.safetensors",
            }
        )
        self.cnn = timm.create_model(
            model_name,
            pretrained=load_weights,
            num_classes=1000 if load_weights else 0,
            global_pool="avg",
            pretrained_cfg_overlay=overlay if load_weights else None,
        )
        # Load the full ImageNet state strictly before discarding its classifier.
        if load_weights:
            self.cnn.reset_classifier(0, global_pool="avg")
        self.output_dim = self.cnn.num_features
        self.bayar = _GAMEDBayarConv(epsilon)
        self.register_buffer("_mean", torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1))

    def forward(self, pixels):
        values = _pixels(pixels).to(self.bayar.weight)
        return self.cnn((self.bayar(values) - self._mean) / self._std)
