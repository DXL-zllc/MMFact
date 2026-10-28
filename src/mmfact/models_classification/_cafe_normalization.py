"""Deterministic marginal KL integration for CAFE's Gaussian mixtures.

The distribution is still the uniform training-reference mixture in paper Eq4.
Tensor-product Gauss-Hermite integrates each component's expectation; increasing
orders must agree before a denominator can be used. This avoids turning a noisy
negative Monte Carlo estimate into an artificial near-zero denominator.
"""

from functools import lru_cache
import math

import numpy as np
import torch


@lru_cache(maxsize=32)
def _rule(order, dimension):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    grids = np.meshgrid(*([nodes] * dimension), indexing="ij")
    grid = np.stack(grids, axis=-1).reshape(-1, dimension)
    weight_grids = np.meshgrid(*([weights] * dimension), indexing="ij")
    mass = np.prod(np.stack(weight_grids, axis=-1), axis=-1).reshape(-1)
    mass /= math.pi ** (dimension / 2)
    return torch.from_numpy(grid), torch.from_numpy(mass)


def _parameters(values):
    mean, scale = (value.detach().to(device="cpu", dtype=torch.float64) for value in values)
    if (
        mean.ndim != 2
        or mean.shape != scale.shape
        or not mean.numel()
        or not torch.isfinite(mean).all()
        or not torch.isfinite(scale).all()
        or not scale.gt(0).all()
    ):
        raise ValueError("CAFE marginal integration requires finite means and positive scales.")
    return mean, scale


def _log_density(points, mean, scale):
    terms = -0.5 * ((points[:, None, :] - mean) / scale).square() - scale.log()
    return (
        terms.sum(-1).logsumexp(-1)
        - math.log(len(mean))
        - mean.shape[-1] * math.log(2 * math.pi) / 2
    )


def _at_order(first, second, order):
    mean, scale = first
    other_mean, other_scale = second
    # Current CAFE has two latent dimensions. Keep accidental high-dimensional
    # tensor products bounded rather than silently exhausting host memory.
    if len(mean) * order ** mean.shape[-1] > 2**21:
        raise ValueError("CAFE quadrature exceeds its point budget; reduce latent dimensionality.")
    grid, mass = _rule(order, mean.shape[-1])
    points = (mean[:, None, :] + math.sqrt(2) * scale[:, None, :] * grid).reshape(
        -1, mean.shape[-1]
    )
    mass = mass.repeat(len(mean)) / len(mean)
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(points), 256):
        values = points[start : start + 256]
        ratio = _log_density(values, mean, scale) - _log_density(values, other_mean, other_scale)
        total += (mass[start : start + 256] * ratio).sum()
    return total


@torch.inference_mode()
def marginal_kl_pair(first, second, *, orders, rtol, atol):
    """Return both KLs, last order difference and order; never sample or refit.

    The fixed train reference and fitted encoder parameters define the mixture.
    Like the previous fitted normalizer, these statistics have no gradient;
    conditional per-news Gaussian KLs retain their classifier gradient path.
    """
    if (
        not isinstance(orders, (list, tuple))
        or len(orders) < 2
        or any(type(order) is not int or order < 3 for order in orders)
        or any(left >= right for left, right in zip(orders, orders[1:]))
        or not math.isfinite(rtol)
        or not math.isfinite(atol)
        or rtol <= 0
        or atol <= 0
    ):
        raise ValueError("CAFE quadrature requires increasing orders and positive tolerances.")
    first, second = _parameters(first), _parameters(second)
    if first[0].shape[-1] != second[0].shape[-1]:
        raise ValueError("CAFE modality mixtures must share a latent dimension.")
    previous = None
    for order in orders:
        current = torch.stack((_at_order(first, second, order), _at_order(second, first, order)))
        if not torch.isfinite(current).all():
            raise ValueError("CAFE marginal quadrature produced a non-finite divergence.")
        if previous is not None:
            error = (current - previous).abs()
            if current.ge(-atol).all() and error.le(atol + rtol * current.abs()).all():
                return current, error, order
        previous = current
    raise ValueError(
        f"CAFE marginal KL quadrature did not converge at order {orders[-1]}: "
        f"KL={current.tolist()}, last differences={error.tolist()}"
    )
