"""Gaussian feature heads and training-marginal KL used by ambiguity models."""

import math

import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import functional as F


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, 2 * latent_dim)
        )

    def forward(self, features):
        mean, raw_scale = self.network(features).chunk(2, dim=-1)
        return Independent(Normal(mean, F.softplus(raw_scale) + 1e-7), 1)


def mixture_kl(first, second, samples, seed):
    """Monte Carlo KL of uniform Gaussian mixtures, in stable float64 CPU chunks."""
    mean, scale = (value.detach().cpu().double() for value in first)
    other_mean, other_scale = (value.detach().cpu().double() for value in second)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(len(mean), samples, mean.shape[-1], generator=generator, dtype=mean.dtype)
    points = (mean[:, None] + scale[:, None] * noise).flatten(0, 1)

    def log_density(values, centers, scales):
        terms = -0.5 * ((values[:, None] - centers) / scales).square() - scales.log()
        return terms.sum(-1).logsumexp(-1) - math.log(len(centers))

    total = torch.zeros((), dtype=mean.dtype)
    for chunk in points.split(256):
        total += (
            log_density(chunk, mean, scale) - log_density(chunk, other_mean, other_scale)
        ).sum()
    return total / len(points)
