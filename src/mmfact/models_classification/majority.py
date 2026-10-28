"""Training-label majority with additive count smoothing (C01)."""

import math
from copy import deepcopy

import torch

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.data.features import fingerprint
from mmfact.models_classification._classical_baselines import (
    _checksum,
    _fit_seed,
    _training_data,
    _validate_identity,
)
from mmfact.registry import register_model

_RECIPE = {"name": "majority", "version": 1, "class_order": [0, 1], "tie": "real"}


@register_model("Majority")
class Majority(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = deepcopy(config["model"]["parameters"])
        if (
            not isinstance(p, dict)
            or set(p) != {"smoothing"}
            or type(p["smoothing"]) not in (int, float)
            or not math.isfinite(p["smoothing"])
            or p["smoothing"] <= 0
        ):
            raise ValueError("Majority requires one finite positive smoothing parameter.")
        self.parameters_spec, self.resource = p, None
        self.register_buffer("_device", torch.empty(0), persistent=False)

    def training_stages(self):
        return (TrainingStage("fit", mode="fit"),)

    def calculate_loss(self, batch):
        raise RuntimeError("Majority uses a single fit stage, not gradient training.")

    def fit_stage(self, stage, train_dataset, val_loader, options):
        _fit_seed(stage, options)
        _, labels, identity = _training_data(train_dataset)
        counts = [labels.count(0), labels.count(1)]
        resource = {
            "ready": True,
            "recipe": deepcopy(_RECIPE),
            "training_sha256": identity,
            "num_samples": len(labels),
            "class_counts": counts,
        }
        resource["sha256"] = _checksum(resource)
        self.resource = resource
        return {
            key: deepcopy(resource[key])
            for key in ("num_samples", "class_counts", "training_sha256")
        }

    @torch.no_grad()
    def predict(self, batch):
        if self.resource is None:
            raise RuntimeError("Majority must be fitted or restored before prediction.")
        counts = self.resource["class_counts"]
        low, high = min(counts), max(counts)
        smoothing = self.parameters_spec["smoothing"]
        ratio = (high - low) / (low + smoothing)
        difference = (
            math.log1p(ratio)
            if math.isfinite(ratio)
            else math.log(high + smoothing) - math.log(low + smoothing)
        )
        # A common shift preserves probabilities and avoids losing the majority
        # when a very large smoothing value makes log(n+s) round to equal values.
        values = [difference if count > low else 0.0 for count in counts]
        return torch.tensor(values, dtype=torch.float64, device=self._device.device).expand(
            len(batch["text"]), 2
        )

    def get_extra_state(self):
        return {
            "version": 1,
            "parameters": deepcopy(self.parameters_spec),
            "resource": deepcopy(self.resource),
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "parameters", "resource"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or fingerprint(state["parameters"]) != fingerprint(self.parameters_spec)
        ):
            raise ValueError("Majority checkpoint parameters or schema differ.")
        resource = deepcopy(state["resource"])
        if resource is not None:
            if not isinstance(resource, dict) or set(resource) != {
                "ready",
                "recipe",
                "training_sha256",
                "num_samples",
                "class_counts",
                "sha256",
            }:
                raise ValueError("Majority checkpoint resource fields are invalid.")
            _validate_identity(resource, _RECIPE)
        self.resource = resource
