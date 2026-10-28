"""BMR: frozen token analyzers, multi-task experts and bootstrapped views."""

import math
import random
import re

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.data.features import file_digest
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._encoders import MAETokenEncoder as _SemanticTokens
from mmfact.models_classification._encoders import RGBEncoder as _Pixels
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.models_classification._transformers import ViTExpert as _Expert
from mmfact.models_classification._transformers import (
    sinusoidal_positions as _positions,
)
from mmfact.registry import register_model


class _TextTokens(BertEncoder):
    def __init__(self, language="en", minimum_words=5, **options):
        super().__init__(output="tokens", mask_padding=False, **options)
        self.language, self.minimum_words = language, minimum_words
        if language == "zh":
            import jieba

            self.segmenter = jieba.Tokenizer()
        self.encoding_spec.update(
            short_text={
                "language": language,
                "minimum_words": minimum_words,
                "placeholder": "No text provided.",
                "recipe": "words-v1",
            }
        )
        if language == "zh":
            self.packages = (*self.packages, "jieba")
            with self.segmenter.get_dict_file() as dictionary:
                self.encoding_spec["short_text"]["dictionary_sha256"] = file_digest(dictionary.name)

    def get_extra_state(self):
        return dict(self.encoding_spec["short_text"])

    def set_extra_state(self, state):
        if state != self.encoding_spec["short_text"]:
            raise ValueError("BMR short-text resources differ from the checkpoint.")

    def _encode(self, texts):
        def complete(text):
            words = self.segmenter.lcut(text) if self.language == "zh" else text.split()
            count = sum(bool(re.search(r"\w", word)) for word in words)
            return text if count >= self.minimum_words else "No text provided."

        return super()._encode([complete(text) for text in texts])


class _MLP(nn.Sequential):
    def __init__(self, inputs, hidden, outputs):
        super().__init__(
            nn.Linear(inputs, hidden),
            nn.ELU(),
            SingletonBatchNorm(hidden),
            nn.Linear(hidden, outputs),
        )


class _BayarConv(nn.Module):
    """Normalize each 5x5 kernel's surroundings to one, with center fixed at -1."""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(3, 3, 5, 5))
        mask = torch.ones(1, 1, 5, 5)
        mask[..., 2, 2] = 0
        self.register_buffer("mask", mask)
        self.epsilon = epsilon

    def kernel(self):
        values = self.weight * self.mask
        total = values.sum((-2, -1), keepdim=True)
        valid = total.abs() >= self.epsilon
        values = torch.where(valid, values / torch.where(valid, total, 1), self.mask / 24)
        return values - (1 - self.mask)

    def forward(self, pixels):
        return F.conv2d(pixels, self.kernel(), padding=2)


class _Pattern(nn.Module):
    def __init__(self, dimension, hidden, epsilon):
        super().__init__()
        from torchvision.models import inception_v3

        self.bayar = _BayarConv(epsilon)
        self.cnn = inception_v3(
            weights=None, aux_logits=False, transform_input=False, init_weights=False, dropout=0.0
        )
        features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.projection = _MLP(features, hidden, dimension)

    def forward(self, pixels):
        return self.projection(self.cnn(self.bayar(pixels)))


class _Mixture(nn.Module):
    def __init__(self, dimension, experts, heads, multiplier, tasks):
        super().__init__()
        self.experts = nn.ModuleList(
            [_Expert(dimension, heads, multiplier) for _ in range(experts)]
        )
        self.attention = nn.ModuleList([_MLP(dimension, dimension, 1) for _ in range(tasks)])
        self.gates = nn.ModuleList([_MLP(dimension, dimension, experts) for _ in range(tasks)])

    def forward(self, values, tasks=None):
        tasks = range(len(self.gates)) if tasks is None else tasks
        positioned = values + _positions(values)
        expert_values = torch.stack([expert(positioned) for expert in self.experts], 1)
        outputs = []
        for task in tasks:
            scores = self.attention[task](values.flatten(0, 1)).reshape(*values.shape[:2], 1)
            weights = self.gates[task]((scores * values).sum(1))
            outputs.append(torch.einsum("be,beld->bld", weights, expert_values))
        return tuple(outputs)


@register_model("BMR")
class BMR(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        positive = (
            "num_experts",
            "num_heads",
            "ffn_multiplier",
            "classifier_hidden",
            "routing_dim",
        )
        if any(type(p[key]) is not int or p[key] < 1 for key in positive):
            raise ValueError("BMR dimensions, heads and expert counts must be positive integers.")
        if p["routing_dim"] % p["num_heads"]:
            raise ValueError("BMR routing_dim must be divisible by num_heads.")
        if (
            p["text"]["language"] not in {"en", "zh"}
            or type(p["text"]["minimum_words"]) is not int
            or p["text"]["minimum_words"] < 0
            or type(p["text"]["max_length"]) is not int
            or p["text"]["max_length"] < 2
        ):
            raise ValueError("BMR requires en/zh language and a nonnegative word minimum.")
        if (
            not p["text"]["freeze"]
            or not p["image"]["freeze"]
            or min(p["alpha"], p["beta"]) < 0
            or not math.isfinite(p["alpha"] + p["beta"])
            or p["bayar_epsilon"] <= 0
            or not math.isfinite(p["bayar_epsilon"])
            or type(p["pair_seed"]) is not int
        ):
            raise ValueError("BMR requires frozen analyzers and valid loss/Bayar coefficients.")
        image = p["image"]
        if (
            image.get("output", "tokens") != "tokens"
            or p["pattern_image"].get("normalize") is not False
            or type(image["image_size"]) is not int
            or image["image_size"] < 16
            or type(image["minimum_size"]) is not int
            or image["minimum_size"] < 0
            or not isinstance(image["augment"], bool)
            or not 0 <= image["flip_probability"] <= 1
            or len(image["color_jitter"]) != 4
            or any(not math.isfinite(x) or x < 0 for x in image["color_jitter"])
            or image["color_jitter"][3] > 0.5
        ):
            raise ValueError("Invalid BMR image/augmentation settings.")
        self.alpha, self.beta = p["alpha"], p["beta"]
        self.pair_seed = p["pair_seed"]
        self.text_encoder, self.image_encoder = (
            material_encoder(config, "text_encoder", _TextTokens, **p["text"]),
            material_encoder(config, "image_encoder", _SemanticTokens, **image),
        )
        self.pattern_pixels = material_encoder(
            config, "pattern_pixels", _Pixels, **p["pattern_image"]
        )
        dimension = p["routing_dim"]
        if self.text_encoder.output_dim != dimension or self.image_encoder.output_dim != dimension:
            raise ValueError("BMR MAE/BERT hidden sizes must equal routing_dim.")
        options = (dimension, p["num_experts"], p["num_heads"], p["ffn_multiplier"])
        self.image_mixture, self.text_mixture = _Mixture(*options, 2), _Mixture(*options, 2)
        self.fusion, self.bootstrap = _Mixture(*options, 1), _Mixture(*options, 1)
        hidden = p["classifier_hidden"]
        self.pattern = _Pattern(dimension, dimension, p["bayar_epsilon"])
        self.coarse = nn.ModuleDict(
            {name: _MLP(dimension, hidden, 1) for name in ("is", "ip", "t")}
        )
        self.reweight = nn.ModuleDict({name: _MLP(1, hidden, 1) for name in ("is", "ip", "t", "m")})
        self.consistency, self.classifier = _MLP(dimension, hidden, 1), _MLP(dimension, hidden, 1)
        self.irrelevance = nn.Parameter(torch.ones(1, dimension))
        self.pair_pool = ()

    def prepare(self, train_dataset):
        real = [row for row in train_dataset if row["label"] == 0]
        if len(real) < 2:
            raise ValueError("BMR consistency training needs at least two Real training examples.")
        order = list(range(len(real)))
        random.Random(self.pair_seed).shuffle(order)
        self.pair_pool = tuple(
            (
                real[index]["text"],
                real[index]["image_path"],
                real[order[(place + 1) % len(real)]]["image_path"],
            )
            for place, index in enumerate(order)
        )

    def _fused(self, batch, consistency_only=False):
        tasks = (1,) if consistency_only else None
        images = self.image_mixture(self.image_encoder(batch["image_path"]), tasks)
        texts = self.text_mixture(self.text_encoder(batch["text"]), tasks)
        fused = self.fusion(torch.cat((images[-1], texts[-1]), 1))[0][:, 0]
        return images[0][:, 0], texts[0][:, 0], fused

    def forward(self, batch):
        semantic, text, fused = self._fused(batch)
        pattern = self.pattern(self.pattern_pixels(batch["image_path"]))
        views = {"is": semantic, "ip": pattern, "t": text}
        logits = {name: self.coarse[name](view).squeeze(-1) for name, view in views.items()}
        weighted = {
            name: view * self.reweight[name](logits[name].sigmoid()[:, None]).sigmoid()
            for name, view in views.items()
        }
        matching = self.consistency(fused).squeeze(-1)
        unrelated = self.irrelevance * self.reweight["m"](matching.sigmoid()[:, None])
        tokens = torch.stack((weighted["is"], weighted["ip"], fused, unrelated, weighted["t"]), 1)
        logits["final"] = self.classifier(self.bootstrap(tokens)[0][:, 0]).squeeze(-1)
        return logits

    def _consistency_loss(self, count):
        if not self.pair_pool:
            raise ValueError("Call BMR.prepare(train_dataset) before training consistency.")
        indices = torch.randint(2 * len(self.pair_pool), (count,)).tolist()
        pairs = [self.pair_pool[index // 2] for index in indices]
        batch = {
            "text": [pair[0] for pair in pairs],
            "image_path": [pair[1 + index % 2] for pair, index in zip(pairs, indices)],
        }
        fused = self._fused(batch, consistency_only=True)[-1]
        logits = self.consistency(fused).squeeze(-1)
        targets = logits.new_tensor([1 - index % 2 for index in indices])
        return F.binary_cross_entropy_with_logits(logits, targets)

    def calculate_loss(self, batch):
        consistency = self._consistency_loss(len(batch["labels"]))
        logits, targets = self(batch), batch["labels"].float()
        return {
            "classification": F.binary_cross_entropy_with_logits(logits["final"], targets),
            "coarse": self.alpha
            * torch.stack(
                [
                    F.binary_cross_entropy_with_logits(logits[name], targets)
                    for name in ("is", "ip", "t")
                ]
            ).mean(),
            "consistency": self.beta * consistency,
        }

    def predict(self, batch):
        training = self.training
        try:
            self.eval()
            logits = self(batch)["final"]
            return torch.stack((torch.zeros_like(logits), logits), -1)
        finally:
            self.train(training)
