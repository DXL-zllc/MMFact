"""FSRU (AAAI 2024): spectrum compression, co-selection and dual contrast."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import RGBEncoder, WordEncoder
from mmfact.models_classification._transformers import sinusoidal_positions
from mmfact.registry import register_model


class _Compression(nn.Module):
    """Eq.5: real power spectrum times a bank of complex-valued filters."""

    def __init__(self, length, dimension, filters):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(filters, length, dimension, 2))
        angles = (2 * torch.arange(1, filters + 1) - 1) * math.pi / (2 * filters)
        self.register_buffer("coefficients", angles.cos())

    def forward(self, spectrum):
        power = spectrum.abs().square() / spectrum.shape[1]
        weights = torch.view_as_complex(self.bank.contiguous())
        compressed = (weights * self.coefficients[:, None, None]).sum(0)
        return power * compressed


class _SpectrumGate(nn.Module):
    def __init__(self, length, dimension):
        super().__init__()
        self.selection = nn.Parameter(torch.randn(length, dimension, 2))
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, spectrum):
        weighted = spectrum * torch.view_as_complex(self.selection.contiguous())
        return self.projection(weighted.mean(1).real).unsqueeze(1)


class _AddNorm(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        values = self.norm1(values)
        return self.norm2(values + self.ffn(self.dropout(values)))


class _SpectrumLayer(nn.Module):
    """Both CSC gates read the unselected USC spectra, as in Eq.6–7."""

    def __init__(self, text_length, image_length, dimension, filters, dropout):
        super().__init__()
        text_bins, image_bins = text_length // 2 + 1, image_length // 2 + 1
        self.text_compression = _Compression(text_bins, dimension, filters)
        self.image_compression = _Compression(image_bins, dimension, filters)
        self.text_gate = _SpectrumGate(text_bins, dimension)
        self.image_gate = _SpectrumGate(image_bins, dimension)
        self.text_norm, self.image_norm = (
            _AddNorm(dimension, dropout),
            _AddNorm(dimension, dropout),
        )

    def forward(self, text, image):
        # FFT kernels require full precision for arbitrary sequence lengths.
        with torch.autocast(device_type=text.device.type, enabled=False):
            text_input, image_input = text, image
            if text.dtype in {torch.float16, torch.bfloat16}:
                text, image = text.float(), image.float()
            text_spectrum = self.text_compression(torch.fft.rfft(text, dim=1, norm="ortho"))
            image_spectrum = self.image_compression(torch.fft.rfft(image, dim=1, norm="ortho"))
            selected_text = text_spectrum * self.image_gate(image_spectrum)
            selected_image = image_spectrum * self.text_gate(text_spectrum)
            text = torch.fft.irfft(selected_text, n=text.shape[1], dim=1, norm="ortho")
            image = torch.fft.irfft(selected_image, n=image.shape[1], dim=1, norm="ortho")
        return (
            self.text_norm(text.to(text_input.dtype) + text_input),
            self.image_norm(image.to(image_input.dtype) + image_input),
        )


def _js_divergence(text, image):
    """Per-news JS divergence of probability distributions, never batch-averaged."""
    log_p, log_q = F.log_softmax(text.float(), -1), F.log_softmax(image.float(), -1)
    log_mean = torch.logaddexp(log_p, log_q) - math.log(2)
    divergence = (log_p.exp() * (log_p - log_mean) + log_q.exp() * (log_q - log_mean)) / 2
    return divergence.sum(-1, keepdim=True).clamp_min(0).to(text.dtype)


def _supervised_contrast(features, labels, temperature):
    """Eq.10: class-mean sums of pairwise objectives; exclude self pairs."""
    values = F.normalize(features.float(), dim=-1)
    similarity = (values @ values.T) / temperature
    diagonal = torch.eye(len(values), device=values.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positives.any(-1)
    if not valid.any():
        return features.sum() * 0
    # L1 follows the cited Lin et al. (COLING 2022), Eq.11–12.
    logits = similarity[valid].masked_fill(diagonal[valid], -torch.inf)
    pair_losses = torch.logsumexp(logits, -1, keepdim=True) - similarity[valid]
    anchor_losses = pair_losses.masked_fill(~positives[valid], 0).sum(-1)
    valid_labels = labels[valid]
    return sum(anchor_losses[valid_labels == label].mean() for label in valid_labels.unique())


def _intermodal_contrast(text, image, temperature):
    logits = F.normalize(text.float(), dim=-1) @ F.normalize(image.float(), dim=-1).T
    logits = logits / temperature
    targets = torch.arange(len(text), device=text.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


@register_model("FSRU")
class FSRU(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        options = config["model"]["parameters"]
        self.text = material_encoder(config, "text", WordEncoder, **options["text"])
        self.image = material_encoder(config, "image", RGBEncoder, **options["image"])
        dimension, patch = options["hidden_dim"], options["patch_size"]
        for name in ("hidden_dim", "patch_size", "num_layers", "num_filters"):
            if type(options[name]) is not int or options[name] < 1:
                raise ValueError(f"FSRU {name} must be a positive integer.")
        if self.image.image_size % patch:
            raise ValueError("FSRU image_size must be divisible by patch_size.")
        dropout = options["dropout"]
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("FSRU dropout must be finite and in [0,1).")
        self.alpha, self.beta = float(options["alpha"]), float(options["beta"])
        self.full_temperature, self.self_temperature = (
            float(options["full_temperature"]),
            float(options["self_temperature"]),
        )
        if any(not math.isfinite(v) or v < 0 for v in (self.alpha, self.beta)):
            raise ValueError("FSRU loss weights must be finite and nonnegative.")
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.full_temperature, self.self_temperature)
        ):
            raise ValueError("FSRU temperatures must be finite and positive.")
        self.text_projection = nn.Sequential(
            nn.Linear(self.text.dimension, dimension), nn.LayerNorm(dimension)
        )
        self.patch_projection = nn.Conv2d(3, dimension, patch, stride=patch)
        patches = (self.image.image_size // patch) ** 2
        self.image_position = nn.Parameter(torch.empty(1, patches, dimension))
        self.input_dropout = nn.Dropout(dropout)
        self.spectrum = nn.ModuleList(
            _SpectrumLayer(
                self.text.max_length, patches, dimension, options["num_filters"], dropout
            )
            for _ in range(options["num_layers"])
        )
        self.text_weight = nn.Parameter(torch.randn(dimension, dimension))
        self.image_weight = nn.Parameter(torch.randn(dimension, dimension))
        self.classifier = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.LayerNorm(4 * dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
            nn.LayerNorm(dimension),
            nn.ReLU(),
            nn.Linear(dimension, 2),
        )
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.image_position, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def prepare(self, train_dataset):
        self.text.prepare([row["text"] for row in train_dataset])

    def _forward(self, batch):
        embedded, _, _ = self.text(batch["text"])
        text = self.text_projection(embedded)
        text = self.input_dropout(text + sinusoidal_positions(text))
        image = self.patch_projection(self.image(batch["image_path"]))
        image = self.input_dropout(image.flatten(2).transpose(1, 2) + self.image_position)
        for layer in self.spectrum:
            text, image = layer(text, image)
        text, image = text.amax(1), image.amax(1)
        divergence = _js_divergence(text, image)
        mixed = text @ self.text_weight + image @ self.image_weight
        fused = (1 - divergence) * mixed + divergence * (text + image)
        return text, image, self.classifier(fused)

    def calculate_loss(self, batch):
        text, image, logits = self._forward(batch)
        labels = batch["labels"]
        supervised = _supervised_contrast(text, labels, self.full_temperature)
        supervised = supervised + _supervised_contrast(image, labels, self.full_temperature)
        return {
            "classification": F.cross_entropy(logits, labels),
            "intra_modal": self.alpha * supervised,
            "inter_modal": self.beta * _intermodal_contrast(text, image, self.self_temperature),
        }

    def predict(self, batch):
        return self._forward(batch)[-1]
