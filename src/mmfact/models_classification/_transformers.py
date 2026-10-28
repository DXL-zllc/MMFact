"""Token expert blocks shared by BMR and MIMoE."""

import math

import torch
from torch import nn


class ViTExpert(nn.Module):
    """Single pre-norm ViT block, matching timm Block's unmasked attention."""

    def __init__(self, dimension, heads, multiplier):
        super().__init__()
        self.heads, self.scale = heads, (dimension // heads) ** -0.5
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.qkv, self.projection = (
            nn.Linear(dimension, 3 * dimension, bias=False),
            nn.Linear(dimension, dimension),
        )
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * multiplier),
            nn.GELU(),
            nn.Linear(dimension * multiplier, dimension),
        )

    def forward(self, values):
        batch, length, dimension = values.shape
        q, k, v = (
            self.qkv(self.norm1(values))
            .reshape(batch, length, 3, self.heads, dimension // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attention = ((q @ k.transpose(-1, -2)) * self.scale).softmax(-1)
        mixed = (attention @ v).transpose(1, 2).reshape(batch, length, dimension)
        values = values + self.projection(mixed)
        return values + self.ffn(self.norm2(values))


def sinusoidal_positions(values):
    length, dimension = values.shape[1:]
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(0, dimension, 2, device=values.device).float() / dimension
    )
    angles = torch.arange(length, device=values.device).float()[:, None] * frequencies
    return torch.stack((angles.sin(), angles.cos()), -1).flatten(1)[:, :dimension].to(values)
