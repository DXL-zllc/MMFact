"""MHR Lorentz detector with a content-addressed, train-only reference graph."""

import hashlib
import json
import math
import re
from copy import deepcopy
from importlib.metadata import version

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.config import _validate_json_value
from mmfact.data.features import _value_digest, file_digest, fingerprint, state_digest
from mmfact.materials.encoders import hub_inputs, material_encoder
from mmfact.materials.training import (
    hub_training_resource,
    load_prepared_resource,
    resource_rows,
    resource_tensor,
)
from mmfact.models_classification._encoders import BertEncoder, MAETokenEncoder
from mmfact.models_classification._mhr_geometry import _MHRHead
from mmfact.registry import register_model


class _MAECLS(MAETokenEncoder):
    """Cache only the unmasked final-LN CLS, without also caching RGB pixels."""

    def __init__(self, **options):
        super().__init__(**options, output="cls")
        self.pixels.cache_features = False


def _content_keys(texts, paths, *, prepared=False):
    if (
        not isinstance(texts, (list, tuple))
        or not isinstance(paths, (list, tuple))
        or not texts
        or len(texts) != len(paths)
        or any(not isinstance(value, str) or not value for value in (*texts, *paths))
    ):
        raise ValueError("MHR requires aligned nonempty raw text and image-path strings.")
    if prepared:
        from pathlib import Path

        images = {path: Path(path).stem for path in dict.fromkeys(paths)}
        if any(not _sha(value) for value in images.values()):
            raise ValueError("MHR prepared image paths require canonical SHA256 filenames.")
    else:
        images = {path: file_digest(path) for path in dict.fromkeys(paths)}
    digests = [
        {"text": hashlib.sha256(text.encode("utf-8")).hexdigest(), "image": images[path]}
        for text, path in zip(texts, paths, strict=True)
    ]
    return [fingerprint(item) for item in digests], digests, images


def _check_image_cache(encoder, images):
    context = encoder.preprocessing
    if context is not None:
        for path, digest in images.items():
            if path in context.input_keys and context.input_keys[path] != digest:
                raise ValueError("MHR image bytes changed after cache binding; reconfigure inputs.")


def _encoder_identity(encoder):
    if hasattr(encoder, "matrix"):
        return {"spec": encoder.encoding_spec, "catalog_sha256": encoder.matrix.identity}
    return {
        "spec": json.loads(json.dumps(encoder.encoding_spec, allow_nan=False)),
        "weights_sha256": state_digest(encoder.network),
        "versions": {package: version(package) for package in encoder.packages},
    }


def _weight_stamp(encoder):
    if hasattr(encoder, "matrix"):
        return (("catalog", encoder.matrix.identity),)
    return tuple(
        (name, id(value), value._version)
        for name, value in (*encoder.network.named_parameters(), *encoder.network.named_buffers())
    )


def _features(text, image):
    if (
        not isinstance(text, torch.Tensor)
        or not isinstance(image, torch.Tensor)
        or text.ndim != 2
        or image.ndim != 2
        or not text.shape[0]
        or text.shape[0] != image.shape[0]
        or not text.is_floating_point()
        or not image.is_floating_point()
    ):
        raise ValueError("MHR requires aligned nonempty raw modal feature matrices.")
    values = []
    for value in (text, image):
        value = value.double()
        norms = value.norm(dim=-1, keepdim=True)
        if not torch.isfinite(value).all() or not torch.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("MHR raw modal features must be finite and nonzero.")
        values.append(value / norms)
    return F.normalize(torch.cat(values, -1), dim=-1)


def _neighbors(query, reference, count, excluded):
    """Stable binary positive-cosine topK; columns are already canonical keys."""
    scores = query @ reference.T
    valid = scores > 0
    for row, column in enumerate(excluded):
        if column is not None:
            valid[row, column] = False
    indices = scores.masked_fill(~valid, -torch.inf).argsort(descending=True, stable=True)[
        :, : min(count, len(reference))
    ]
    edges = torch.zeros_like(scores)
    edges.scatter_(1, indices, valid.gather(1, indices).to(edges))
    return edges


def _reference_adjacency(features, count):
    edges = _neighbors(features, features, count, range(len(features)))
    edges.fill_diagonal_(1)
    return edges / edges.sum(-1, keepdim=True)


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


class _ReferenceGraph(nn.Module):
    """Portable immutable raw anchors; learned reference paths stay in the head."""

    def __init__(self, text_dim, image_dim, reference_size, neighbors):
        super().__init__()
        self.text_dim, self.image_dim = text_dim, image_dim
        self.options = {"reference_size": reference_size, "neighbors": neighbors}
        self.recipe = {
            **self.options,
            "content": "exact-utf8-and-original-image-sha256",
            "anchors": "sorted-unique-content-keys",
            "features": "float64-modal-l2-concat-l2",
            "edges": "positive-cosine-stable-topk-binary-directed-self-row-degree",
            "queries": "exclude-identical-content-independent-self-last",
        }
        self._resource = None
        self._views = None

    @property
    def ready(self):
        return self._resource is not None

    @property
    def reference_text(self):
        return None if not self.ready else self._resource["reference_text"]

    @property
    def reference_image(self):
        return None if not self.ready else self._resource["reference_image"]

    @property
    def adjacency(self):
        return None if not self.ready else self._resource["adjacency"]

    @property
    def anchor_keys(self):
        return [] if not self.ready else list(self._resource["anchor_keys"])

    @property
    def provenance(self):
        if not self.ready:
            return None
        return deepcopy(
            {
                key: value
                for key, value in self._resource.items()
                if not isinstance(value, torch.Tensor)
            }
        )

    def prepare(self, rows, text_encoder, image_encoder, batch_size):
        texts, paths = [], []
        for row in rows:
            texts.append(row["text"])
            paths.append(row["image_path"])
        keys, digests, images = _content_keys(
            texts, paths, prepared=hasattr(image_encoder, "matrix")
        )
        _check_image_cache(image_encoder, images)
        indices = {key: index for index, key in reversed(list(enumerate(keys)))}
        anchors = sorted(indices)[: self.options["reference_size"]]
        selected = [indices[key] for key in anchors]
        modal = [[], []]
        for offset in range(0, len(selected), batch_size):
            chunk = selected[offset : offset + batch_size]
            for outputs, encoder, inputs in (
                (modal[0], text_encoder, texts),
                (modal[1], image_encoder, paths),
            ):
                outputs.append(encoder([inputs[index] for index in chunk]).detach().float().cpu())
        text, image = (torch.cat(outputs) for outputs in modal)
        resource = {
            "recipe": deepcopy(self.recipe),
            "num_samples": len(keys),
            "num_unique": len(indices),
            "training_sha256": fingerprint(sorted(keys)),
            "anchor_keys": anchors,
            "anchor_digests": [digests[index] for index in selected],
            "reference_text": text,
            "reference_image": image,
            "adjacency": _reference_adjacency(_features(text, image), self.options["neighbors"]),
            "encoders": {
                "text": _encoder_identity(text_encoder),
                "image": _encoder_identity(image_encoder),
            },
        }
        resource["sha256"] = _value_digest(resource)
        self._validate_resource(resource)
        self._resource, self._views = resource, None

    def device_views(self, device):
        if not self.ready:
            raise ValueError("MHR reference graph is unprepared.")
        device = torch.device(device)
        if self._views is None or self._views[0].device != device:
            text, image, adjacency = (
                self._resource[key].to(device)
                for key in ("reference_text", "reference_image", "adjacency")
            )
            self._views = (text, image, adjacency, _features(text, image))
        return self._views

    def query_adjacency(self, text, image, keys):
        if text.shape[1:] != (self.text_dim,) or image.shape[1:] != (self.image_dim,):
            raise ValueError("MHR query raw feature widths do not match the reference graph.")
        reference = self.device_views(text.device)[3]
        indices = {key: index for index, key in enumerate(self.anchor_keys)}
        with torch.no_grad():
            edges = _neighbors(
                _features(text, image),
                reference,
                self.options["neighbors"],
                [indices.get(key) for key in keys],
            )
            edges = torch.cat((edges, torch.ones_like(edges[:, :1])), -1)
            return edges / edges.sum(-1, keepdim=True)

    def _validate_resource(self, resource):
        expected = {
            "recipe",
            "num_samples",
            "num_unique",
            "training_sha256",
            "anchor_keys",
            "anchor_digests",
            "reference_text",
            "reference_image",
            "adjacency",
            "encoders",
            "sha256",
        }
        if not isinstance(resource, dict) or set(resource) != expected:
            raise ValueError("Invalid MHR graph resource schema.")
        _validate_json_value(
            {
                key: value
                for key, value in resource.items()
                if key not in {"reference_text", "reference_image", "adjacency"}
            },
            "MHR graph",
        )
        total, unique = resource["num_samples"], resource["num_unique"]
        if (
            fingerprint(resource["recipe"]) != fingerprint(self.recipe)
            or type(total) is not int
            or type(unique) is not int
            or not 1 <= unique <= total
            or not _sha(resource["training_sha256"])
        ):
            raise ValueError("MHR graph recipe or training identity is invalid.")
        keys, digests = resource["anchor_keys"], resource["anchor_digests"]
        count = min(self.options["reference_size"], unique)
        if (
            not isinstance(keys, list)
            or len(keys) != count
            or any(not _sha(key) for key in keys)
            or keys != sorted(set(keys))
            or not isinstance(digests, list)
            or len(digests) != count
        ):
            raise ValueError("MHR anchor keys must be unique, sorted and match training counts.")
        for key, digest in zip(keys, digests, strict=True):
            if (
                not isinstance(digest, dict)
                or set(digest) != {"text", "image"}
                or any(not _sha(value) for value in digest.values())
                or fingerprint(digest) != key
            ):
                raise ValueError("MHR anchor content digests do not match their keys.")
        for name, width, dtype in (
            ("reference_text", self.text_dim, torch.float32),
            ("reference_image", self.image_dim, torch.float32),
            ("adjacency", count, torch.float64),
        ):
            value = resource[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.device.type != "cpu"
                or value.dtype != dtype
                or tuple(value.shape) != (count, width)
                or value.requires_grad
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"MHR {name} must be a finite CPU {dtype} matrix of correct shape."
                )
        expected_adjacency = _reference_adjacency(
            _features(resource["reference_text"], resource["reference_image"]),
            self.options["neighbors"],
        )
        if not torch.equal(resource["adjacency"], expected_adjacency):
            raise ValueError(
                "MHR adjacency does not match the graph reconstructed from raw anchors."
            )
        encoders = resource["encoders"]
        if not isinstance(encoders, dict) or set(encoders) != {"text", "image"}:
            raise ValueError("MHR graph requires both encoder identities.")
        for identity in encoders.values():
            if (
                isinstance(identity, dict)
                and set(identity) == {"spec", "catalog_sha256"}
                and isinstance(identity["spec"], dict)
                and identity["spec"]
                and _sha(identity["catalog_sha256"])
            ):
                continue
            if (
                not isinstance(identity, dict)
                or set(identity) != {"spec", "weights_sha256", "versions"}
                or not isinstance(identity["spec"], dict)
                or not identity["spec"]
                or not _sha(identity["weights_sha256"])
                or not isinstance(identity["versions"], dict)
                or not identity["versions"]
                or any(
                    not isinstance(key, str) or not key or not isinstance(value, str) or not value
                    for key, value in identity["versions"].items()
                )
            ):
                raise ValueError("MHR encoder identity is invalid.")
        if not _sha(resource["sha256"]) or resource["sha256"] != _value_digest(
            {key: value for key, value in resource.items() if key != "sha256"}
        ):
            raise ValueError("MHR graph checksum does not match its complete payload.")

    def get_extra_state(self):
        return {
            "version": 1,
            "text_dim": self.text_dim,
            "image_dim": self.image_dim,
            "options": deepcopy(self.options),
            "resource": deepcopy(self._resource),
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "text_dim", "image_dim", "options", "resource"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or type(state["text_dim"]) is not int
            or state["text_dim"] != self.text_dim
            or type(state["image_dim"]) is not int
            or state["image_dim"] != self.image_dim
            or not isinstance(state["options"], dict)
            or any(type(value) is not int for value in state["options"].values())
            or fingerprint(state["options"]) != fingerprint(self.options)
        ):
            raise ValueError("MHR graph checkpoint dimensions or options differ.")
        _validate_json_value(state["options"], "MHR graph options")
        resource = state["resource"]
        if resource is not None:
            self._validate_resource(resource)
        self._resource, self._views = deepcopy(resource), None

    def _apply(self, fn, recurse=True):
        self._views = None
        return super()._apply(fn, recurse)


def _validate(p):
    _validate_json_value(p, "MHR parameters")
    if set(p) != {"text", "image", "hidden_dim", "dim", "num_layers", "dropout", "graph"}:
        raise ValueError("MHR requires only text, image, geometry and graph parameters.")
    text, image, graph = p["text"], p["image"], p["graph"]
    if (
        not isinstance(text, dict)
        or set(text) - {"global_attention"}
        != {"model_name", "revision", "freeze", "max_length", "output"}
        or text.get("global_attention") not in (None, "cls")
        or text["freeze"] is not True
        or text["output"] != "cls"
        or type(text["max_length"]) is not int
        or text["max_length"] < 2
        or not isinstance(image, dict)
        or set(image)
        != {
            "model_name",
            "revision",
            "freeze",
            "image_size",
            "minimum_size",
            "augment",
            "center_crop",
            "resize_size",
            "normalize",
            "mean",
            "std",
        }
        or image["freeze"] is not True
        or image["augment"] is not False
        or type(image["center_crop"]) is not bool
        or type(image["resize_size"]) is not int
        or image["resize_size"] < image["image_size"]
        or image["normalize"] is not True
        or type(image["minimum_size"]) is not int
        or image["minimum_size"] != 0
        or type(image["image_size"]) is not int
        or image["image_size"] < 1
    ):
        raise ValueError("MHR requires frozen CLS text and normalized deterministic MAE inputs.")
    for options in (text, image):
        if any(
            not isinstance(options[key], str) or not options[key].strip()
            for key in ("model_name", "revision")
        ):
            raise ValueError("MHR checkpoints and revisions must be nonempty strings.")
    if any(
        not isinstance(image[key], list)
        or len(image[key]) != 3
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (key == "std" and value <= 0)
            for value in image[key]
        )
        for key in ("mean", "std")
    ):
        raise ValueError("MHR image normalization requires three finite values and positive std.")
    if (
        any(
            type(p[key]) is not int or p[key] < minimum
            for key, minimum in (("hidden_dim", 1), ("dim", 2), ("num_layers", 2))
        )
        or type(p["dropout"]) not in (int, float)
        or not math.isfinite(p["dropout"])
        or not 0 <= p["dropout"] < 1
        or not isinstance(graph, dict)
        or set(graph) != {"reference_size", "neighbors"}
        or any(type(value) is not int or value < 1 for value in graph.values())
    ):
        raise ValueError("MHR geometry/graph sizes must be positive and dropout in [0,1).")


@register_model("MHR")
class MHR(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = deepcopy(config["model"]["parameters"])
        _validate(p)
        self.parameters_spec = p
        self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **p["text"])
        self.image_encoder = material_encoder(config, "image_encoder", _MAECLS, **p["image"])
        self.graph = _ReferenceGraph(
            self.text_encoder.output_dim, self.image_encoder.output_dim, **p["graph"]
        )
        self.head = _MHRHead(
            self.text_encoder.output_dim,
            self.image_encoder.output_dim,
            p["hidden_dim"],
            p["dim"],
            p["num_layers"],
            p["dropout"],
        )
        self._graph_valid, self._encoder_stamps, self._graph_stamp = False, None, None
        for encoder in (self.text_encoder, self.image_encoder):
            encoder.register_load_state_dict_post_hook(self._invalidate_graph)
            if not hasattr(encoder, "matrix"):
                encoder.network.register_load_state_dict_post_hook(self._invalidate_graph)
        self.graph.register_load_state_dict_post_hook(self._invalidate_graph)
        self.register_load_state_dict_pre_hook(self._before_load)
        self.register_load_state_dict_post_hook(self._after_load)

    def _reset_encoder_caches(self):
        for encoder in (self.text_encoder, self.image_encoder):
            if encoder.preprocessing is not None:
                encoder.preprocessing.reset()

    def _invalidate_graph(self, _module, _incompatible_keys):
        self._reset_encoder_caches()
        self._graph_valid, self._encoder_stamps, self._graph_stamp = False, None, None

    def _before_load(self, *_args):
        self._graph_valid, self._encoder_stamps, self._graph_stamp = False, None, None

    def _check_frozen(self):
        if any(value.dtype != torch.float64 for value in self.head.parameters()):
            raise ValueError("MHR geometry weights must stay float64.")
        for name, encoder in (("text", self.text_encoder), ("image", self.image_encoder)):
            if hasattr(encoder, "matrix"):
                if encoder.freeze is not True or any(p.requires_grad for p in encoder.parameters()):
                    raise ValueError("MHR Hub encoders must remain immutable.")
                continue
            if encoder.freeze is not True or any(
                value.requires_grad for value in encoder.network.parameters()
            ):
                raise ValueError(
                    "MHR graph requires frozen encoders; prepare after changing weights."
                )
            if any(value.dtype != torch.float32 for value in encoder.network.parameters()):
                raise ValueError(
                    "MHR frozen encoder weights must stay float32; geometry stays float64."
                )
            options = self.parameters_spec[name]
            if name == "text" and (
                encoder.output != "cls"
                or encoder.max_length != options["max_length"]
                or getattr(encoder, "global_attention", None) != options.get("global_attention")
            ):
                raise ValueError("MHR text recipe changed; reconstruct and prepare the model.")
            if name == "image" and (
                encoder.augment is not False
                or encoder.pixels.cache_features is not False
                or encoder.pixels.image_size != options["image_size"]
                or encoder.pixels.minimum_size != 0
                or encoder.pixels.center_crop != options["center_crop"]
                or encoder.pixels.resize_size != options["resize_size"]
                or encoder.pixels.normalize is not True
                or encoder.pixels.mean != tuple(options["mean"])
                or encoder.pixels.std != tuple(options["std"])
                or encoder.network.config.mask_ratio != 0
            ):
                raise ValueError("MHR image recipe changed; reconstruct and prepare the model.")

    def _resource_stamp(self):
        return (
            id(self.graph._resource),
            *[
                (id(value), value._version)
                for value in (
                    self.graph.reference_text,
                    self.graph.reference_image,
                    self.graph.adjacency,
                )
            ],
        )

    def _bind_graph(self):
        self._check_frozen()
        actual = {
            "text": _encoder_identity(self.text_encoder),
            "image": _encoder_identity(self.image_encoder),
        }
        if fingerprint(self.graph._resource["encoders"]) != fingerprint(actual):
            raise ValueError(
                "MHR reference graph does not match actual encoder weights, recipes or versions."
            )
        self._encoder_stamps = tuple(
            _weight_stamp(encoder) for encoder in (self.text_encoder, self.image_encoder)
        )
        self._graph_stamp = self._resource_stamp()
        self._graph_valid = True

    def _after_load(self, _module, incompatible_keys):
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            raise ValueError("MHR requires a complete strict model checkpoint.")
        self.head.validate_state()
        if self.graph.ready:
            self._bind_graph()

    @torch.no_grad()
    def prepare(self, train_dataset):
        self._graph_valid = False
        self._check_frozen()
        self._reset_encoder_caches()
        resource = hub_training_resource(self.config, "mhr_reference_graph")
        if resource is not None:
            rows = resource_rows(resource, "pair_id")
            train = list(train_dataset)
            keys, _, _ = _content_keys(
                [row["text"] for row in train],
                [row["image_path"] for row in train],
                prepared=True,
            )
            digests = [{"text": row["text_id"], "image": row["image_id"]} for row in rows]
            anchor_keys = [fingerprint(digest) for digest in digests]
            state = {
                "recipe": deepcopy(self.graph.recipe),
                "num_samples": len(keys),
                "num_unique": len(set(keys)),
                "training_sha256": fingerprint(sorted(keys)),
                "anchor_keys": anchor_keys,
                "anchor_digests": digests,
                **{
                    name: resource_tensor(resource, name)
                    for name in ("reference_text", "reference_image", "adjacency")
                },
                "encoders": {
                    "text": _encoder_identity(self.text_encoder),
                    "image": _encoder_identity(self.image_encoder),
                },
            }
            if not set(anchor_keys).issubset(keys):
                raise ValueError("MHR reference anchors are outside the training split.")
            state["sha256"] = _value_digest(state)
            self.graph.set_extra_state(
                {
                    **self.graph.get_extra_state(),
                    "resource": state,
                }
            )
            self._bind_graph()
            return
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()

            def identity():
                keys, _, images = _content_keys(
                    [row["text"] for row in train_dataset],
                    [row["image_path"] for row in train_dataset],
                )
                _check_image_cache(self.image_encoder, images)
                return {
                    "training_keys": sorted(keys),
                    "graph": self.graph.recipe,
                    "text": _encoder_identity(self.text_encoder),
                    "image": _encoder_identity(self.image_encoder),
                    "batch_size": self.config.get("preprocessing", {}).get("batch_size", 64),
                }

            def build():
                self.graph.prepare(
                    train_dataset,
                    self.text_encoder,
                    self.image_encoder,
                    self.config.get("preprocessing", {}).get("batch_size", 64),
                )
                return self.graph.get_extra_state()

            state = load_prepared_resource(
                self,
                "mhr_reference_graph",
                identity,
                build,
                encoders=(self.text_encoder, self.image_encoder),
            )
            self.graph.set_extra_state(state)
            self._bind_graph()
        finally:
            for module, mode in modes:
                module.training = mode

    def forward(self, batch):
        if not self.graph.ready or not self._graph_valid:
            raise ValueError(
                "MHR reference graph is unprepared or invalidated; prepare or load a complete checkpoint."
            )
        self._check_frozen()
        current = tuple(
            _weight_stamp(encoder) for encoder in (self.text_encoder, self.image_encoder)
        )
        if current != self._encoder_stamps or self._resource_stamp() != self._graph_stamp:
            self._graph_valid = False
            raise ValueError(
                "MHR encoder weights or graph resources changed; prepare the reference graph again."
            )
        texts, paths = batch["text"], batch["image_path"]
        keys, _, images = _content_keys(texts, paths, prepared=hub_inputs(self.config))
        _check_image_cache(self.image_encoder, images)
        text, image = self.text_encoder(texts), self.image_encoder(paths)
        reference_text, reference_image, adjacency, _ = self.graph.device_views(text.device)
        return self.head(
            text,
            image,
            reference_text,
            reference_image,
            adjacency,
            self.graph.query_adjacency(text, image, keys),
        )

    def calculate_loss(self, batch):
        return F.cross_entropy(self(batch), batch["labels"])

    @torch.no_grad()
    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self(batch)
        finally:
            for module, mode in modes:
                module.training = mode

    def get_extra_state(self):
        return {"version": 1, "model": "MHR", "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        self._graph_valid, self._encoder_stamps, self._graph_stamp = False, None, None
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "model", "parameters"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or state["model"] != "MHR"
        ):
            raise ValueError("Invalid MHR model state schema.")
        _validate_json_value(state["parameters"], "MHR state parameters")
        if fingerprint(state["parameters"]) != fingerprint(self.parameters_spec):
            raise ValueError("MHR checkpoint parameters differ from the configured model.")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if assign:
            raise ValueError("MHR requires assign=False to preserve manifold parameter ownership.")
        return super().load_state_dict(state_dict, strict=strict, assign=False)

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0))
        if probe.device.type == "mps":
            raise ValueError(
                "MHR float64 Lorentz geometry requires CPU or CUDA; MPS is unsupported."
            )
        if (
            probe.dtype != torch.float32
            or fn(torch.empty(0, dtype=torch.float64)).dtype != torch.float64
        ):
            raise ValueError(
                "MHR only supports device migration; preserve float32 encoders and float64 geometry."
            )
        valid = self._graph_valid
        self._graph_valid, self._encoder_stamps, self._graph_stamp = False, None, None
        result = super()._apply(fn, recurse)
        self.head.validate_state()
        if valid and self.graph.ready:
            self._bind_graph()
        return result
