"""Disclosed block-DCT resource and trainable Event-Radar pattern view."""

import math

import torch
from PIL import Image
from torch import nn

from mmfact.models_classification._encoders import _PretrainedEncoder


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _probability(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1).")


def _dct_basis(size, *, dtype=torch.float32, device=None):
    """Orthonormal DCT-II matrix: frequency rows, sample columns."""
    _positive_integer("DCT patch_size", size)
    samples = torch.arange(size, dtype=torch.float64, device=device) + 0.5
    frequencies = torch.arange(size, dtype=torch.float64, device=device)
    basis = torch.cos(math.pi * frequencies[:, None] * samples[None] / size)
    basis *= math.sqrt(2 / size)
    basis[0] /= math.sqrt(2)
    return basis.to(dtype=dtype)


def _block_dct(pixels, basis):
    """Transform each nonoverlapping square patch; keep signed spatial block order."""
    if (
        pixels.ndim != 4
        or basis.ndim != 2
        or basis.shape[0] != basis.shape[1]
        or basis.shape[0] < 1
        or pixels.shape[-2] % basis.shape[0]
        or pixels.shape[-1] % basis.shape[0]
    ):
        raise ValueError("DCT requires [B,C,H,W] pixels divisible by its square basis.")
    size = basis.shape[0]
    blocks = pixels.unfold(2, size, size).unfold(3, size, size)
    transformed = basis @ blocks @ basis.T
    return transformed.permute(0, 1, 2, 4, 3, 5).reshape_as(pixels)


class DCTEncoder(_PretrainedEncoder):
    """Frozen RGB [0,1], bicubic square resize, signed orthonormal block DCT-II."""

    input_kind = "image"
    packages = ("torch", "torchvision", "Pillow")

    def __init__(self, image_size=168, patch_size=24):
        super().__init__()
        _positive_integer("DCT image_size", image_size)
        _positive_integer("DCT patch_size", patch_size)
        if image_size % patch_size:
            raise ValueError("DCT image_size must be divisible by patch_size.")
        self.image_size, self.patch_size = image_size, patch_size
        self.network = nn.Identity()
        self.register_buffer("_basis", _dct_basis(patch_size), persistent=False)
        self.encoding_spec = {
            "encoder": "event-radar-dct",
            "schema": 1,
            "color": "PIL-RGB",
            "exif_transpose": False,
            "image_size": image_size,
            "resize": "PIL-square-bicubic",
            "pixels": "uint8/255",
            "dct": "II",
            "normalization": "ortho",
            "patch_size": patch_size,
            "stride": patch_size,
            "layout": "channel-first-original-block-positions-frequency-rows-columns",
            "coefficients": "signed",
            "dtype": "torch.float32",
        }
        self._set_frozen(True)

    def _encode(self, paths):
        from torchvision.transforms.functional import pil_to_tensor

        pixels = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
                pixels.append(pil_to_tensor(image).float().div(255))
        if not pixels:
            return torch.empty(0, 3, self.image_size, self.image_size, device=self.device)
        values = torch.stack(pixels).to(self.device)
        return _block_dct(values, self._basis.to(values))


class _PatternBlock(nn.Module):
    def __init__(self, dim, heads, attention_dropout, residual_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=attention_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, values):
        normalized = self.norm1(values)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        values = values + self.dropout(attended)
        return values + self.dropout(self.mlp(self.norm2(values)))


class _PatternView(nn.Module):
    """Author pattern structure with corrected per-news attention axis."""

    def __init__(
        self,
        image_size=168,
        patch_size=24,
        dim=64,
        depth=3,
        heads=8,
        view_dim=32,
        attention_dropout=0.5,
        residual_dropout=0.5,
        pool_dropout=0.1,
    ):
        super().__init__()
        for name, value in (
            ("image_size", image_size),
            ("patch_size", patch_size),
            ("dim", dim),
            ("depth", depth),
            ("heads", heads),
            ("view_dim", view_dim),
        ):
            _positive_integer(f"Pattern {name}", value)
        if image_size % patch_size or dim % heads:
            raise ValueError("Pattern image_size/patch_size and dim/heads must divide evenly.")
        for name, value in (
            ("attention_dropout", attention_dropout),
            ("residual_dropout", residual_dropout),
            ("pool_dropout", pool_dropout),
        ):
            _probability(f"Pattern {name}", value)
        self.image_size, self.output_dim = image_size, view_dim
        self.projection = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.stem_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        self.position = nn.Parameter(torch.empty(1, (image_size // patch_size) ** 2 + 1, dim))
        self.blocks = nn.Sequential(
            *[_PatternBlock(dim, heads, attention_dropout, residual_dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(pool_dropout)
        self.feature = nn.Sequential(nn.Linear(dim, 64), nn.LeakyReLU(), nn.Linear(64, view_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, values):
        if values.ndim != 4 or tuple(values.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError("Pattern input must be [B,3,image_size,image_size].")
        tokens = self.stem_norm(self.projection(values).flatten(2).transpose(1, 2))
        tokens = torch.cat((self.cls_token.expand(len(values), -1, -1), tokens), dim=1)
        tokens = self.norm(self.blocks(tokens + self.position))
        return self.feature(self.dropout(tokens).mean(dim=1))
