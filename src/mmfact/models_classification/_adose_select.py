"""Label-free ADOSE LUS/MDC over a frozen snapshot of the target pool."""

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _variance_grid(variances):
    if isinstance(variances, Mapping):
        if set(variances) != {"count", "min", "max"}:
            raise ValueError("ADOSE variances require count, min and max.")
        count, low, high = (variances[key] for key in ("count", "min", "max"))
        if type(count) is not int or count < 1:
            raise ValueError("ADOSE variance count must be a positive integer.")
        for value in (low, high):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("ADOSE variance bounds must be finite numbers.")
        if not 0 < low <= high <= 1 or (count == 1 and low != high) or (count > 1 and low == high):
            raise ValueError("ADOSE variances must be increasing within (0,1].")
        return torch.linspace(low, high, count, dtype=torch.float64)
    if not isinstance(variances, (list, tuple)) or not variances:
        raise ValueError("ADOSE variances must be a nonempty grid.")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in variances
    ):
        raise ValueError("ADOSE variances must be finite and within (0,1].")
    if any(first >= second for first, second in zip(variances, variances[1:])):
        raise ValueError("ADOSE variances must be strictly increasing.")
    return torch.tensor(variances, dtype=torch.float64)


def _diversity(features):
    """Whole-pool mean cosine distances, with zero-vector self-distance zero."""
    count = len(features[0])
    if not count:
        return torch.empty(0, dtype=torch.float64)
    scores = []
    for values in features:
        normalized = F.normalize(values, dim=-1)
        score = 1 - normalized @ normalized.mean(0)
        score -= values.norm(dim=-1).eq(0).to(score.dtype) / count
        scores.append(score.clamp(0, 2))
    return torch.stack(scores).mean(0)


@torch.no_grad()
def select_samples(
    features,
    heads,
    k,
    *,
    multiplier,
    variances,
    samples_per_variance,
    seed,
    chunk_size,
):
    """Return pool indices and safe audit data; never mutate or call model heads.

    Features are the three expert inputs immediately before their final Linear.
    All hypotheses perturb three weight matrices together; biases remain fixed.
    CPU float64 snapshots and a local generator bound memory and RNG side effects.
    """
    for name, value, minimum in (
        ("k", k, 0),
        ("multiplier", multiplier, 1),
        ("samples_per_variance", samples_per_variance, 1),
        ("chunk_size", chunk_size, 1),
        ("seed", seed, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"ADOSE {name} must be an integer >= {minimum}.")
    if seed >= 2**64:
        raise ValueError("ADOSE seed must be smaller than 2**64.")
    grid = _variance_grid(variances)
    if not isinstance(features, (tuple, list)) or len(features) != 3:
        raise ValueError("ADOSE selection needs exactly three expert feature matrices.")
    if not isinstance(heads, (tuple, list)) or len(heads) != 3:
        raise ValueError("ADOSE selection needs exactly three final Linear heads.")
    count = (
        features[0].shape[0]
        if isinstance(features[0], torch.Tensor) and features[0].ndim == 2
        else -1
    )
    snapshots, weights, biases = [], [], []
    for values, head in zip(features, heads, strict=True):
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 2
            or not values.is_floating_point()
            or len(values) != count
            or not isinstance(head, nn.Linear)
            or head.out_features != 2
            or values.shape[1] != head.in_features
            or not torch.isfinite(values).all()
            or not torch.isfinite(head.weight).all()
            or (head.bias is not None and not torch.isfinite(head.bias).all())
        ):
            raise ValueError("ADOSE expert features/heads must have compatible finite shapes.")
        snapshots.append(values.detach().to(device="cpu", dtype=torch.float64))
        weights.append(head.weight.detach().to(device="cpu", dtype=torch.float64).clone())
        biases.append(
            None
            if head.bias is None
            else head.bias.detach().to(device="cpu", dtype=torch.float64).clone()
        )
    if k > count:
        raise ValueError("ADOSE query budget cannot exceed the remaining target pool.")

    def predictions(current_weights):
        result = []
        for start in range(0, count, chunk_size):
            scores = sum(
                F.log_softmax(F.linear(values[start : start + chunk_size], weight, bias), dim=-1)
                for values, weight, bias in zip(snapshots, current_weights, biases, strict=True)
            )
            result.append(scores.argmax(-1))
        return torch.cat(result) if result else torch.empty(0, dtype=torch.long)

    ldm = torch.ones(count, dtype=torch.float64)
    diversity = _diversity(snapshots)
    candidates = torch.empty(0, dtype=torch.long)
    selected = candidates
    if k:
        baseline = predictions(weights)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for variance in grid:
            for _ in range(samples_per_variance):
                perturbed = [
                    weight
                    + torch.randn(weight.shape, generator=generator, dtype=weight.dtype)
                    * variance.sqrt()
                    for weight in weights
                ]
                changed = predictions(perturbed).ne(baseline)
                rho = changed.to(ldm.dtype).mean()
                ldm[changed] = torch.minimum(ldm[changed], rho)
        candidates = torch.argsort(ldm, stable=True)[: min(multiplier * k, count)]
        # Resolve diversity ties by original pool order, not by LDM ordering.
        candidates = candidates.sort().values
        selected = candidates[
            torch.argsort(diversity[candidates], descending=True, stable=True)[:k]
        ]
    return selected, {
        "ldm": ldm.tolist(),
        "diversity": diversity.tolist(),
        "candidates": candidates.tolist(),
        "variances": grid.tolist(),
        "samples_per_variance": samples_per_variance,
        "seed": seed,
    }
