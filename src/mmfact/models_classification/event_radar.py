"""Event-Radar's three-view graph, evidence and credibility fusion mechanisms."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.materials.resources import material_resources
from mmfact.models_classification._event_dct import DCTEncoder, _PatternView
from mmfact.models_classification._event_emotion import emotion_layout
from mmfact.models_classification._event_inputs import (
    _event_nodes,
    _EventImage,
    _EventText,
)
from mmfact.models_classification._event_resources import EventResourceStore
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fraction(value, name, *, include_one=False):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        or (value == 1 and not include_one)
    ):
        interval = "[0,1]" if include_one else "[0,1)"
        raise ValueError(f"{name} must be finite and in {interval}")
    return value


def _view_mlp(input_dim, view_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.LeakyReLU(), nn.Linear(64, view_dim))


def _normalized_adjacency(adjacency):
    """Normalize each news/relation independently, including its existing loops."""
    degree = adjacency.sum(-1)
    inverse = degree.clamp_min(1e-8).rsqrt() * degree.gt(0)
    return adjacency * inverse.unsqueeze(-1) * inverse.unsqueeze(-2)


def _event_relation(indices, role_mask, num_nodes):
    """Global-to-role stars with deduplicated indices and participating-node loops."""
    members = F.one_hot(indices.clamp_min(0), num_nodes).bool() & role_mask.unsqueeze(-1)
    center, participants = members[:, 0], members.any(1)
    edges = center.unsqueeze(-1) & participants.unsqueeze(-2)
    return edges | edges.transpose(-1, -2) | torch.diag_embed(participants)


class _EventGraph(nn.Module):
    """Parallel full/text/image GCNs followed by four paired-role comparisons."""

    def __init__(
        self,
        node_dim=512,
        embedding_dim=64,
        view_dim=32,
        num_layers=2,
        adjacency_weight=0.6,
        dropout=0.1,
        branch_state="shared",
    ):
        super().__init__()
        if branch_state not in ("shared", "separate"):
            raise ValueError("graph branch_state must be 'shared' or 'separate'")
        self.branch_state = branch_state
        for name, value in (
            ("node_dim", node_dim),
            ("embedding_dim", embedding_dim),
            ("view_dim", view_dim),
            ("num_layers", num_layers),
        ):
            _positive_int(value, name)
        _fraction(adjacency_weight, "adjacency_weight", include_one=True)
        _fraction(dropout, "dropout")
        self.node_dim = node_dim
        self.adjacency_weight = adjacency_weight
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(3)])
                for _ in range(num_layers)
            ]
        )
        self.dynamic = nn.ParameterList(
            [nn.Parameter(torch.empty(node_dim, node_dim)) for _ in range(num_layers - 1)]
        )
        for layer in self.layers:
            for convolution in layer:
                nn.init.xavier_uniform_(convolution.weight)
                nn.init.zeros_(convolution.bias)
        for weight in self.dynamic:
            nn.init.xavier_uniform_(weight)
        self.node_norm = SingletonBatchNorm(node_dim)
        self.comparison = nn.Sequential(
            nn.Linear(4 * node_dim, node_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(node_dim, embedding_dim),
        )
        self.pair_norm = SingletonBatchNorm(4 * embedding_dim)
        self.ambiguity = nn.Sequential(
            nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.output_norm = SingletonBatchNorm(embedding_dim)
        self.feature = _view_mlp(embedding_dim, view_dim)

    def _validate(self, nodes, node_mask, text_indices, image_indices, text_mask, image_mask):
        if (
            nodes.ndim != 3
            or nodes.shape[-1] != self.node_dim
            or min(nodes.shape[:2]) < 1
            or not nodes.is_floating_point()
            or node_mask.shape != nodes.shape[:2]
            or node_mask.dtype != torch.bool
        ):
            raise ValueError("nodes must be [B,N,D] floats with a matching boolean node mask")
        if not torch.isfinite(nodes[node_mask]).all():
            raise ValueError("valid graph nodes must be finite")
        for indices, mask in ((text_indices, text_mask), (image_indices, image_mask)):
            if (
                indices.shape != (len(nodes), 4)
                or indices.dtype != torch.long
                or mask.shape != indices.shape
                or mask.dtype != torch.bool
                or not mask[:, 0].all()
                or (indices < -1).any()
                or (indices >= nodes.shape[1]).any()
                or (indices[mask] < 0).any()
                or not node_mask.gather(1, indices.clamp_min(0))[mask].all()
            ):
                raise ValueError(
                    "four role indices/masks must select valid nodes, including global"
                )

    @staticmethod
    def _initial_graph(nodes, node_mask):
        valid_edges = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        values = nodes.masked_fill(~node_mask.unsqueeze(-1), 0)
        unit = F.normalize(values, dim=-1, eps=1e-8)
        adjacency = ((unit @ unit.transpose(-1, -2) + 1) / 2).clamp(0, 1)
        adjacency = adjacency.masked_fill(~valid_edges, 0)
        identity = torch.eye(values.shape[1], device=values.device, dtype=values.dtype)
        adjacency = adjacency * (1 - identity)
        adjacency = adjacency + torch.diag_embed(node_mask.to(values.dtype))
        return valid_edges, values, adjacency

    def _dynamic_update(self, index, adjacency, values, valid_edges):
        update = torch.sigmoid(values @ self.dynamic[index] @ values.transpose(-1, -2))
        adjacency = self.adjacency_weight * adjacency + (1 - self.adjacency_weight) * update
        return adjacency.masked_fill(~valid_edges, 0)

    def _propagate_separate(self, nodes, node_mask, text_relation, image_relation):
        """Paper Eq.1-3: full, text-event and image-event states each recur on their own.

        The branches meet only through the edge weights, which the full-graph state
        updates dynamically. Only H_P^L and H_I^L reach the comparison, so the full
        branch is not run after the last dynamic update.
        """
        valid_edges, values, adjacency = self._initial_graph(nodes, node_mask)
        relations = (valid_edges, text_relation, image_relation)
        states = [values, values, values]
        for index, layer in enumerate(self.layers):
            for branch, (relation, convolution) in enumerate(zip(relations, layer)):
                if branch == 0 and index >= len(self.dynamic):
                    continue
                weights = adjacency * relation
                if index == 0 and branch > 0:
                    weights = relation.to(values.dtype)
                message = F.relu(convolution(_normalized_adjacency(weights) @ states[branch]))
                states[branch] = message * relation.any(-1).unsqueeze(-1)
            if index < len(self.dynamic):
                adjacency = self._dynamic_update(index, adjacency, states[0], valid_edges)
        return states[1], states[2]

    def _propagate(self, nodes, node_mask, text_relation, image_relation):
        valid_edges, values, adjacency = self._initial_graph(nodes, node_mask)
        relations = (valid_edges, text_relation, image_relation)
        for index, layer in enumerate(self.layers):
            messages = []
            for branch, (relation, convolution) in enumerate(zip(relations, layer)):
                weights = adjacency * relation
                if index == 0 and branch > 0:
                    weights = relation.to(values.dtype)
                message = F.relu(convolution(_normalized_adjacency(weights) @ values))
                messages.append(message * relation.any(-1).unsqueeze(-1))
            values = sum(messages).masked_fill(~node_mask.unsqueeze(-1), 0)
            if index < len(self.dynamic):
                adjacency = self._dynamic_update(index, adjacency, values, valid_edges)
        return values

    def forward(
        self, nodes, node_mask, text_indices, image_indices, text_role_mask, image_role_mask
    ):
        self._validate(
            nodes, node_mask, text_indices, image_indices, text_role_mask, image_role_mask
        )
        text_relation = _event_relation(text_indices, text_role_mask, nodes.shape[1])
        image_relation = _event_relation(image_indices, image_role_mask, nodes.shape[1])
        if self.branch_state == "separate":
            text_values, image_values = self._propagate_separate(
                nodes, node_mask, text_relation, image_relation
            )
            text_nodes, image_nodes = text_relation.any(-1), image_relation.any(-1)
            rows = self.node_norm(torch.cat((text_values[text_nodes], image_values[image_nodes])))
            text_states = torch.zeros_like(text_values)
            image_states = torch.zeros_like(image_values)
            count = int(text_nodes.sum())
            text_states[text_nodes] = rows[:count]
            image_states[image_nodes] = rows[count:]
        else:
            values = self._propagate(nodes, node_mask, text_relation, image_relation)
            text_states = torch.zeros_like(values)
            text_states[node_mask] = self.node_norm(values[node_mask])
            image_states = text_states

        def gather(states, indices, mask):
            index = indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.node_dim)
            return states.gather(1, index) * mask.unsqueeze(-1)

        text = gather(text_states, text_indices, text_role_mask)
        image = gather(image_states, image_indices, image_role_mask)
        pairs = self.comparison(torch.cat((text, image, text - image, text * image), -1))
        present = text_role_mask | image_role_mask
        pairs = pairs * present.unsqueeze(-1)
        pairs = self.pair_norm(pairs.flatten(1)).reshape_as(pairs)
        pairs = pairs * present.unsqueeze(-1)
        return self.feature(self.output_norm(self.ambiguity(pairs.flatten(1))))


class _EmotionView(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, view_dim=32):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("embedding_dim", embedding_dim),
            ("view_dim", view_dim),
        ):
            _positive_int(value, name)
        self.input_dim = input_dim
        self.normalization = SingletonBatchNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.feature = _view_mlp(embedding_dim, view_dim)

    def forward(self, emotions):
        if (
            emotions.ndim != 2
            or emotions.shape != (len(emotions), self.input_dim)
            or len(emotions) == 0
            or not emotions.is_floating_point()
            or not torch.isfinite(emotions).all()
        ):
            raise ValueError("emotions must be finite [B,input_dim] floats")
        return self.feature(self.encoder(self.normalization(emotions)))


class _CredibilityFusion(nn.Module):
    """Independent Beta evidence and direct credibility-weighted attention."""

    def __init__(self, view_dim=32, heads=4, dropout=0.1, classifier="mlp"):
        super().__init__()
        _positive_int(view_dim, "view_dim")
        _positive_int(heads, "heads")
        _fraction(dropout, "dropout")
        if view_dim < 2 or view_dim % heads:
            raise ValueError("view_dim must be at least two and divisible by heads")
        if classifier not in ("mlp", "linear"):
            raise ValueError("fusion classifier must be 'mlp' or 'linear'")
        self.view_dim = view_dim
        self.classifier_type = classifier
        self.evidence = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(view_dim, view_dim),
                    nn.ReLU(),
                    nn.Linear(view_dim, view_dim // 2),
                    nn.Linear(view_dim // 2, 2),
                    nn.Softplus(),
                )
                for _ in range(3)
            ]
        )
        self.attention = nn.MultiheadAttention(view_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        if classifier == "linear":
            # Paper Eq.11: y = W_o [x, x~] + b_o.
            self.classifier = nn.Linear(3 * view_dim + 36, 2)
        else:
            self.embedding = nn.Linear(3 * view_dim + 36, view_dim)
            self.classifier = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(view_dim, view_dim),
                nn.LeakyReLU(),
                nn.Linear(view_dim, 2),
            )

    def forward(self, views):
        if (
            views.ndim != 3
            or views.shape[1:] != (3, self.view_dim)
            or len(views) == 0
            or not views.is_floating_point()
            or not torch.isfinite(views).all()
        ):
            raise ValueError("views must be finite [B,3,view_dim] floats")
        alpha = torch.stack(
            [head(views[:, index]) + 1 for index, head in enumerate(self.evidence)], 1
        )
        credibility = 1 - 2 / alpha.sum(-1)
        weighted = views * credibility.unsqueeze(-1)
        fused, attention = self.attention(weighted, weighted, weighted, need_weights=True)
        structure = F.pad(attention, (0, 3, 0, 3)).flatten(1)
        features = torch.cat((self.dropout(fused).flatten(1), structure), -1)
        if self.classifier_type == "linear":
            # L_c compares each view with the fused x of Eq.9, averaged over its three tokens.
            logits, embedding = self.classifier(features), fused.mean(1)
        else:
            embedding = self.embedding(features)
            logits = self.classifier(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "alpha": alpha,
            "credibility": credibility,
            "attention": attention,
        }


def _credible_loss(alpha, labels):
    """Paper Eq.12: mean across news, sum across the three independent views."""
    if (
        alpha.ndim != 3
        or alpha.shape[1:] != (3, 2)
        or len(alpha) == 0
        or not alpha.is_floating_point()
        or not torch.isfinite(alpha).all()
        or (alpha < 1).any()
        or labels.shape != (len(alpha),)
        or labels.dtype != torch.long
        or ((labels < 0) | (labels > 1)).any()
    ):
        raise ValueError("credible loss requires beta[B,3,2]>=1 and binary long labels[B]")
    selected = alpha.gather(-1, labels[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    return (torch.digamma(alpha.sum(-1)) - torch.digamma(selected)).sum(-1).mean()


def _contrastive_loss(embedding, views, credibility, temperature=0.5):
    """Corrected local ratio: single exp, lowest credibility, no detached q."""
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("contrastive temperature must be positive and finite")
    if (
        views.ndim != 3
        or views.shape[1] != 3
        or len(views) == 0
        or embedding.shape != (len(views), views.shape[-1])
        or credibility.shape != views.shape[:2]
        or not all(value.is_floating_point() for value in (embedding, views, credibility))
        or not all(torch.isfinite(value).all() for value in (embedding, views, credibility))
        or ((credibility < 0) | (credibility > 1)).any()
    ):
        raise ValueError(
            "contrastive inputs must be finite z[B,F], views[B,3,F], and q[B,3] in [0,1]"
        )
    lowest = credibility.argmin(-1, keepdim=True)
    q = credibility.gather(-1, lowest)
    weights = (1 - q).expand_as(credibility).scatter(-1, lowest, q)
    similarity = F.cosine_similarity(embedding.unsqueeze(1), views, dim=-1, eps=1e-8) / temperature
    return (
        similarity.logsumexp(-1) - (similarity + weights.clamp_min(1e-8).log()).logsumexp(-1)
    ).mean()


def _validate_configuration(parameters):
    text, image = parameters["text"], parameters["image"]
    if (
        text.get("freeze") is not True
        or image.get("freeze") is not True
        or text.get("output") != "tokens"
        or text.get("mask_padding") is not True
        or image.get("output", "pooled") != "pooled"
        or text.get("normalize", False) is not False
        or image.get("normalize", False) is not False
        or text.get("model_name") != image.get("model_name")
        or text.get("revision", "main") != image.get("revision", "main")
    ):
        raise ValueError("Event-Radar requires paired frozen raw CLIP tokens and pooled images.")
    for options in (text, image):
        if any(
            not isinstance(options.get(key, default), str) or not options.get(key, default).strip()
            for key, default in (("model_name", ""), ("revision", "main"))
        ):
            raise ValueError("Event-Radar CLIP names and revisions must be nonempty strings.")
    for key in ("credible_weight", "contrastive_weight", "contrastive_temperature"):
        value = parameters[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (key == "contrastive_temperature" and value == 0)
        ):
            raise ValueError(
                "Event-Radar loss weights must be nonnegative and temperature positive."
            )


@register_model("Event-Radar")
class EventRadar(AbstractDetector):
    """Three full trainable views with explicitly adapted, fixed content resources."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate_configuration(p)
        self.parameters_spec = deepcopy(p)
        self.resources = material_resources(
            config, "event_radar", EventResourceStore, **p["resources"]
        )
        self.text_encoder = material_encoder(config, "text_encoder", _EventText, **p["text"])
        self.image_encoder = material_encoder(
            config, "image_encoder", _EventImage, self.resources, **p["image"]
        )
        if self.text_encoder.output_dim != self.image_encoder.output_dim:
            raise ValueError("Event-Radar CLIP projected node dimensions must match.")
        self.dct_encoder = material_encoder(
            config,
            "dct_encoder",
            DCTEncoder,
            image_size=p["pattern"]["image_size"],
            patch_size=p["pattern"]["patch_size"],
        )
        self.graph = _EventGraph(
            node_dim=self.text_encoder.output_dim,
            embedding_dim=p["embedding_dim"],
            view_dim=p["view_dim"],
            **p["graph"],
        )
        self.emotion = _EmotionView(
            len(emotion_layout(p["resources"]["language"])), p["embedding_dim"], p["view_dim"]
        )
        self.pattern = _PatternView(view_dim=p["view_dim"], **p["pattern"])
        self.fusion = _CredibilityFusion(view_dim=p["view_dim"], **p["fusion"])
        self.credible_weight = p["credible_weight"]
        self.contrastive_weight = p["contrastive_weight"]
        self.contrastive_temperature = p["contrastive_temperature"]

    def get_extra_state(self):
        return {"format_version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("parameters") != self.parameters_spec
        ):
            raise ValueError("Event-Radar checkpoint parameters differ from this model.")

    def forward(self, batch):
        texts, paths = batch["text"], batch["image_path"]
        if not texts or len(texts) != len(paths):
            raise ValueError("Event-Radar requires equal nonempty text and image batches.")
        records = [self.resources.lookup(text, path) for text, path in zip(texts, paths)]
        graph = self.graph(
            *_event_nodes(self.text_encoder, self.image_encoder, texts, paths, records)
        )
        emotions = graph.new_tensor([record["emotion"] for record in records])
        views = torch.stack(
            (graph, self.emotion(emotions), self.pattern(self.dct_encoder(paths))), 1
        )
        return {**self.fusion(views), "views": views}

    def calculate_loss(self, batch):
        result = self(batch)
        return {
            "classification": F.cross_entropy(result["logits"], batch["labels"]),
            "credible": self.credible_weight * _credible_loss(result["alpha"], batch["labels"]),
            "contrastive": self.contrastive_weight
            * _contrastive_loss(
                result["embedding"],
                result["views"],
                result["credibility"],
                self.contrastive_temperature,
            ),
        }

    @torch.no_grad()
    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            return self(batch)["logits"]
        finally:
            for module, mode in modes:
                module.training = mode
