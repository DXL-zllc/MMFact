"""Unit-Lorentz fusion, simultaneous graph propagation and manifold prototypes."""

import math

import geoopt
import torch
from torch import nn
from torch.nn import functional as F


def _inner(first, second):
    return (first[..., 1:] * second[..., 1:]).sum(-1, keepdim=True) - first[..., :1] * second[
        ..., :1
    ]


def _cross(first, second):
    return first[..., 1:] @ second[..., 1:].T - first[..., :1] @ second[..., :1].T


def _project(points):
    spatial = points[..., 1:]
    return torch.cat(((1 + spatial.square().sum(-1, keepdim=True)).sqrt(), spatial), -1)


def _normalize(points):
    if not torch.isfinite(points).all() or (points[..., :1] <= 0).any():
        raise ValueError("MHR aggregation requires finite future-timelike support.")
    # Positive homogeneous rescaling prevents small gates from underflowing the
    # norm. It leaves the Lorentz-normalized result and attention unchanged.
    scaled = points / points.abs().amax(-1, keepdim=True)
    squared = -_inner(scaled, scaled)
    if (squared <= 0).any():
        raise ValueError("MHR aggregation support is zero or spacelike.")
    return scaled / squared.sqrt()


class _CappedLorentz(geoopt.Lorentz):
    """Official geometry with the source's unit-curvature radius cap of ten."""

    def __init__(self):
        super().__init__(k=torch.tensor(1.0, dtype=torch.float64), learnable=False)

    def _cap(self, tangent, dim):
        radius = self.norm(tangent, dim=dim, keepdim=True)
        return tangent * (10 / radius).clamp_max(1)

    def expmap0(self, tangent, *, project=True, dim=-1):
        return super().expmap0(self._cap(tangent, dim), project=project, dim=dim)

    def expmap(self, point, tangent, *, norm_tan=True, project=True, dim=-1):
        if norm_tan:
            tangent = self.proju(point, tangent, dim=dim)
        return super().expmap(
            point, self._cap(tangent, dim), norm_tan=False, project=project, dim=dim
        )

    retr = expmap


class _LorentzLinear(nn.Module):
    """Preserve each source initialization and spatial recipe; correct positive time."""

    def __init__(self, input_dim, output_dim, *, fusion=False, dropout=0.1):
        super().__init__()
        self.affine = nn.Linear(input_dim, output_dim, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.fusion = fusion
        bound = 0.02 if fusion else 1 / math.sqrt(output_dim)
        nn.init.uniform_(self.affine.weight, -bound, bound)
        with torch.no_grad():
            self.affine.weight[:, 0] = 0
            self.affine.bias.zero_()
        self.logscale = nn.Parameter(
            torch.tensor(2.3 if fusion else math.log(10), dtype=torch.float64)
        )

    def forward(self, points):
        affine = self.affine(self.dropout(points))
        time = affine[..., :1].sigmoid() * self.logscale.exp() + 1.1
        spatial = affine[..., 1:]
        squared = spatial.square().sum(-1, keepdim=True)
        denominator = squared + 1e-9 if self.fusion else squared.clamp_min(1e-8)
        spatial = spatial * ((time.square() - 1) / denominator).sqrt()
        return _project(torch.cat((time, spatial), -1))


class _LorentzGraphLayer(nn.Module):
    """One shared transform, then old-reference/self aggregation for each query."""

    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.mapping = _LorentzLinear(input_dim, output_dim, dropout=dropout)
        self.query = _LorentzLinear(output_dim, output_dim)
        self.key = _LorentzLinear(output_dim, output_dim)
        self.scale = nn.Parameter(torch.tensor(math.sqrt(output_dim), dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(20.0, dtype=torch.float64))

    def _weights(self, score, adjacency):
        return ((2 + 2 * score) / self.scale + self.bias).sigmoid() * adjacency

    def forward(self, query, reference, reference_adjacency, query_adjacency):
        reference, query = self.mapping(reference), self.mapping(query)
        reference_query, query_query = self.query(reference), self.query(query)
        reference_key, query_key = self.key(reference), self.key(query)
        reference_weights = self._weights(
            _cross(reference_query, reference_key), reference_adjacency
        )
        query_weights = self._weights(_cross(query_query, reference_key), query_adjacency[:, :-1])
        self_weights = self._weights(_inner(query_query, query_key), query_adjacency[:, -1:])
        next_reference = _normalize(reference_weights @ reference)
        next_query = _normalize(query_weights @ reference + self_weights * query)
        return next_query, next_reference


class _MHRHead(nn.Module):
    """Double geometry with one manifold owner and independent inductive queries."""

    def __init__(self, text_dim, image_dim, hidden_dim, dim, num_layers, dropout):
        super().__init__()
        if (
            any(type(value) is not int or value < 1 for value in (text_dim, image_dim, hidden_dim))
            or type(dim) is not int
            or dim < 2
            or type(num_layers) is not int
            or num_layers < 2
            or type(dropout) not in (int, float)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                "MHR requires positive input widths, dim/layers>=2 and dropout in [0,1)."
            )
        self.text_dim, self.image_dim, self.dim = text_dim, image_dim, dim
        self.manifold = _CappedLorentz()
        self.text_projection = nn.Linear(text_dim, hidden_dim, bias=False, dtype=torch.float64)
        self.image_projection = nn.Linear(image_dim, hidden_dim, bias=False, dtype=torch.float64)
        for projection in (self.text_projection, self.image_projection):
            nn.init.xavier_uniform_(projection.weight)
        self.fusion = _LorentzLinear(hidden_dim + 1, dim + 1, fusion=True)
        self.layers = nn.ModuleList(
            _LorentzGraphLayer(dim + 1 if index == 0 else dim, dim, dropout)
            for index in range(num_layers - 1)
        )
        self.output = _LorentzLinear(dim, dim, fusion=True)
        self.prototypes = geoopt.ManifoldParameter(
            self.manifold.random_normal(2, dim, std=1 / math.sqrt(dim)), manifold=self.manifold
        )
        self.bias = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.register_load_state_dict_pre_hook(self._before_load)
        self.register_load_state_dict_post_hook(self._after_load)

    def _modalities(self, text, image):
        values = []
        for features, width, projection in (
            (text, self.text_dim, self.text_projection),
            (image, self.image_dim, self.image_projection),
        ):
            if (
                not isinstance(features, torch.Tensor)
                or not features.is_floating_point()
                or features.ndim != 2
                or features.shape[1] != width
                or not features.shape[0]
                or not torch.isfinite(features).all()
            ):
                raise ValueError("MHR requires finite nonempty matrices of raw modal features.")
            features = features.to(projection.weight)
            norms = features.norm(dim=-1, keepdim=True)
            if (norms == 0).any():
                raise ValueError("MHR raw features must be nonzero.")
            spatial = projection(features / norms).relu()
            values.append(self.manifold.expmap0(F.pad(spatial, (1, 0))))
        if values[0].shape != values[1].shape:
            raise ValueError("MHR text and image feature batches must align.")
        return self.fusion(_normalize((values[0] + values[1]) / 2))

    def forward(
        self,
        query_text,
        query_image,
        reference_text,
        reference_image,
        reference_adjacency,
        query_adjacency,
    ):
        query = self._modalities(query_text, query_image)
        reference = self._modalities(reference_text, reference_image)
        adjacencies = []
        for adjacency, shape in (
            (reference_adjacency, (reference.shape[0], reference.shape[0])),
            (query_adjacency, (query.shape[0], reference.shape[0] + 1)),
        ):
            if (
                not isinstance(adjacency, torch.Tensor)
                or not adjacency.is_floating_point()
                or tuple(adjacency.shape) != shape
                or not torch.isfinite(adjacency).all()
                or (adjacency < 0).any()
            ):
                raise ValueError("MHR requires finite nonnegative row-normalized adjacencies.")
            adjacency = adjacency.to(query)
            if not torch.allclose(
                adjacency.sum(-1),
                torch.ones(shape[0], device=query.device, dtype=query.dtype),
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("MHR adjacency rows must sum to one.")
            adjacencies.append(adjacency)
        reference_adjacency, query_adjacency = adjacencies
        if (reference_adjacency.diag() <= 0).any() or (query_adjacency[:, -1] <= 0).any():
            raise ValueError("MHR requires a self edge for every reference and query.")
        for layer in self.layers:
            query, reference = layer(query, reference, reference_adjacency, query_adjacency)
        logits = 2 + 2 * _cross(self.output(query), self.prototypes) + self.bias
        if not torch.isfinite(logits).all():
            raise ValueError("MHR logits must be finite.")
        return logits

    def validate_state(self):
        if any(
            value.dtype != torch.float64 or not torch.isfinite(value).all()
            for value in self.parameters()
        ):
            raise ValueError("MHR geometry parameters must be finite float64.")
        if (
            self.manifold.k.shape != ()
            or self.manifold.k.item() != 1
            or self.manifold.k.requires_grad
        ):
            raise ValueError("MHR requires fixed unit Lorentz curvature.")
        if (
            not isinstance(self.prototypes, geoopt.ManifoldParameter)
            or self.prototypes.manifold is not self.manifold
            or tuple(self.prototypes.shape) != (2, self.dim)
        ):
            raise ValueError("MHR prototypes must retain the owned manifold association.")
        tolerance = (
            16 * torch.finfo(torch.float64).eps * self.prototypes.square().sum(-1, keepdim=True)
        ).clamp_min(1e-10)
        if (self.prototypes[..., :1] <= 0).any() or (
            _inner(self.prototypes, self.prototypes).add(1).abs() > tolerance
        ).any():
            raise ValueError("MHR prototypes must lie on the future unit hyperboloid.")

    def _before_load(self, module, state, prefix, *args):
        if any(
            isinstance(value, torch.Tensor)
            and (value.dtype != torch.float64 or not torch.isfinite(value).all())
            for key, value in state.items()
            if key.startswith(prefix)
        ):
            raise ValueError("MHR geometry checkpoint tensors must be finite float64.")

    def _after_load(self, module, incompatible):
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            self.validate_state()
