"""Independently selected text/image classifiers and validation probability fusion."""

import math
from copy import deepcopy

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.data.preprocessing import configure_preprocessing
from mmfact.evaluation.metrics import classification_metrics
from mmfact.models_classification._neural_baselines import _signature
from mmfact.models_classification.imageonly import ImageOnly
from mmfact.models_classification.textplm import TextPLM
from mmfact.registry import register_model


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@register_model("LateFusion")
class LateFusion(AbstractDetector):
    """Train and restore each peer before fitting one text-probability weight."""

    def __init__(self, config):
        super().__init__(config)
        p = deepcopy(config["model"]["parameters"])
        if not isinstance(p, dict) or set(p) != {"text_peer", "image_peer", "alpha_grid"}:
            raise ValueError("LateFusion requires text_peer, image_peer and alpha_grid.")
        grid = p["alpha_grid"]
        if (
            not isinstance(grid, list)
            or not grid
            or any(not _finite(value) or not 0 <= value <= 1 for value in grid)
            or any(first >= second for first, second in zip(grid, grid[1:]))
        ):
            raise ValueError("alpha_grid must be a strictly increasing finite list in [0,1].")
        for name, peer in (("text_peer", TextPLM), ("image_peer", ImageOnly)):
            if not isinstance(p[name], dict):
                raise ValueError(f"{name} must contain the corresponding baseline parameters.")
            peer._validate(p[name])
        self.parameters_spec, self._parameters_signature = p, _signature(p)
        for name, peer in (("text_peer", TextPLM), ("image_peer", ImageOnly)):
            child = {"model": {"name": peer.__name__, "parameters": p[name]}}
            setattr(self, name, peer(child))
        self.register_buffer("alpha", torch.tensor(float(grid[0]), dtype=torch.float64))
        self._stage, self._ready, self.fit_report = "initial", False, None
        self._pending_ready = False
        self._selected = {"text": False, "image": False}
        self._trained = {"text": False, "image": False}
        self._backward_seen = False
        self.text_peer.register_load_state_dict_post_hook(self._peer_loaded)
        self.image_peer.register_load_state_dict_post_hook(self._peer_loaded)
        self.register_load_state_dict_pre_hook(self._before_load)
        self.register_load_state_dict_post_hook(self._after_load)

    @property
    def text_encoder(self):
        return self.text_peer.text_encoder

    @property
    def image_encoder(self):
        return self.image_peer.image_encoder

    def training_stages(self):
        return (
            TrainingStage("text", module="text_peer"),
            TrainingStage("image", module="image_peer"),
            TrainingStage("fusion_weight", mode="fit"),
        )

    def _apply_stage(self, stage):
        self._stage = stage
        for name in ("text", "image"):
            peer, encoder = getattr(self, f"{name}_peer"), getattr(self, f"{name}_encoder")
            active = stage in ("initial", name)
            peer.requires_grad_(active)
            encoder._set_frozen(not active or self.parameters_spec[f"{name}_peer"][name]["freeze"])
            if not active:
                peer.eval()
            if encoder.preprocessing is not None:
                encoder.preprocessing.reset()

    def set_training_stage(self, stage):
        if stage not in ("text", "image", "fusion_weight", None):
            raise ValueError("Unknown LateFusion training stage.")
        if stage == "text":
            self._selected = {"text": False, "image": False}
            self._trained = {"text": False, "image": False}
            self._ready, self.fit_report = False, None
            self.alpha.fill_(self.parameters_spec["alpha_grid"][0])
        elif stage == "image":
            if self._stage not in ("text", "image") or not self._selected["text"]:
                raise RuntimeError("Restore the selected trained text peer before the image stage.")
            self._selected["image"], self._trained["image"] = False, False
            self._ready, self.fit_report = False, None
            self.alpha.fill_(self.parameters_spec["alpha_grid"][0])
        elif stage == "fusion_weight":
            if self._stage not in ("image", "fusion_weight") or not all(self._selected.values()):
                raise RuntimeError("Restore both selected trained peers before fitting alpha.")
        elif stage is None and not self._ready:
            raise RuntimeError("LateFusion inference requires a fitted validation weight.")
        self._backward_seen = False
        self._apply_stage(stage)

    def train(self, mode=True):
        super().train(mode)
        for name in ("text", "image"):
            if self._stage not in ("initial", name):
                getattr(self, f"{name}_peer").eval()
        return self

    def calculate_loss(self, batch):
        if self._stage not in ("text", "image"):
            raise RuntimeError("Select a LateFusion peer stage before calculating loss.")
        loss = getattr(self, f"{self._stage}_peer").calculate_loss(batch)
        if self.training and loss.requires_grad:
            stage = self._stage

            def backward(gradient):
                if self._stage == stage and torch.isfinite(gradient).all():
                    self._backward_seen = True

            loss.register_hook(backward)
        return loss

    def on_epoch_start(self, epoch, total_epochs):
        self._backward_seen = False

    def on_epoch_end(self, epoch):
        if self._stage not in self._trained or not self._backward_seen:
            raise RuntimeError("LateFusion peer selection requires a completed gradient epoch.")
        self._trained[self._stage] = True

    def _peer_loaded(self, module, incompatible):
        name = "text" if module is self.text_peer else "image"
        if (
            self._stage == name
            and self._trained[name]
            and not incompatible.missing_keys
            and not incompatible.unexpected_keys
        ):
            self._selected[name] = True

    def optimizer_parameter_groups(self):
        if self._stage not in ("text", "image"):
            return {}
        return getattr(self, f"{self._stage}_peer").optimizer_parameter_groups()

    def training_stage_data(self, stage, train_data, val_data, options):
        if stage != self._stage:
            raise ValueError("LateFusion loaders must match the active stage.")
        configured = options.get("parameter_groups", {})
        if set(configured) - {"text", "image", "head"}:
            raise ValueError("Unknown LateFusion optimizer parameter group.")
        active = self.optimizer_parameter_groups()
        options["parameter_groups"] = {
            key: value for key, value in configured.items() if key in active
        }
        loaders = []
        for original, key in ((train_data, "batch_size"), (val_data, "eval_batch_size")):
            if (
                not isinstance(original, DataLoader)
                or original.batch_size is None
                or type(original.batch_sampler) is not BatchSampler
                or type(original.sampler) not in (RandomSampler, SequentialSampler)
                or original.drop_last
            ):
                raise ValueError("LateFusion requires complete ordinary news loaders.")
            if key == "eval_batch_size" and type(original.sampler) is not SequentialSampler:
                raise ValueError("LateFusion validation must be sequential and complete.")
            if stage == "fusion_weight" and original.batch_size == options[key]:
                loaders.append(original)
            else:
                generator = original.generator
                if generator is not None and stage != "fusion_weight":
                    generator = torch.Generator().manual_seed(generator.initial_seed())
                loaders.append(
                    DataLoader(
                        original.dataset,
                        batch_size=options[key],
                        shuffle=isinstance(original.sampler, RandomSampler),
                        collate_fn=original.collate_fn,
                        num_workers=original.num_workers,
                        pin_memory=original.pin_memory,
                        worker_init_fn=original.worker_init_fn,
                        generator=generator,
                        persistent_workers=original.persistent_workers,
                    )
                )
        if stage == "fusion_weight":
            configure_preprocessing(self, {"val": val_data.dataset}, self.config["preprocessing"])
        return tuple(loaders)

    @staticmethod
    def _mix(text, image, alpha):
        probabilities = alpha * text + (1 - alpha) * image
        probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        return (probabilities / probabilities.sum(-1, keepdim=True)).log()

    @torch.no_grad()
    def fit_stage(self, stage, train_dataset, val_loader, options):
        if stage != "fusion_weight" or self._stage != stage or not all(self._selected.values()):
            raise RuntimeError("Alpha fitting requires both selected trained peers.")
        metric, seed = options["selection_metric"], options["seed"]
        if metric not in ("accuracy", "macro_f1") or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError(
                "LateFusion requires an accuracy/macro_f1 metric and an integer run seed."
            )
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("Freeze both peers before fitting alpha.")
        self._ready, self.fit_report = False, None
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            text, image, labels = [], [], []
            for batch in val_loader:
                truth = batch["labels"]
                if (
                    truth.dtype != torch.long
                    or truth.ndim != 1
                    or not ((truth == 0) | (truth == 1)).all()
                ):
                    raise ValueError("Validation labels must be a binary int64 vector.")
                text.append(self.text_peer.predict(batch).float().softmax(-1).cpu())
                image.append(self.image_peer.predict(batch).float().softmax(-1).cpu())
                labels.extend(truth.cpu().tolist())
            if not labels:
                raise ValueError("Alpha fitting requires nonempty validation data.")
            text, image = torch.cat(text), torch.cat(image)
            if (
                text.shape != image.shape
                or text.shape != (len(labels), 2)
                or not torch.isfinite(text).all()
                or not torch.isfinite(image).all()
            ):
                raise ValueError("LateFusion requires finite aligned two-class peer predictions.")
            scores = [
                classification_metrics(labels, self._mix(text, image, value).argmax(-1).tolist())[
                    metric
                ]
                for value in self.parameters_spec["alpha_grid"]
            ]
            best = max(range(len(scores)), key=scores.__getitem__)
            self.alpha.fill_(self.parameters_spec["alpha_grid"][best])
            self.fit_report = {
                "alpha": float(self.alpha),
                "selection_metric": metric,
                "scores": scores,
                "validation_size": len(labels),
                "seed": seed,
            }
            self._ready = True
            return deepcopy(self.fit_report)
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            if self._stage in ("text", "image"):
                return getattr(self, f"{self._stage}_peer").predict(batch)
            if not self._ready:
                raise RuntimeError("LateFusion has not fitted its validation weight.")
            return self._mix(
                self.text_peer.predict(batch).float().softmax(-1),
                self.image_peer.predict(batch).float().softmax(-1),
                float(self.alpha),
            )
        finally:
            for module, mode in modes:
                module.training = mode

    def get_extra_state(self):
        return {
            "version": 1,
            "model": "LateFusion",
            "parameters": deepcopy(self.parameters_spec),
            "stage": self._stage,
            "ready": self._ready,
            "selected": deepcopy(self._selected),
            "fit_report": deepcopy(self.fit_report),
        }

    def set_extra_state(self, state):
        self._ready = False
        if (
            not isinstance(state, dict)
            or set(state)
            != {"version", "model", "parameters", "stage", "ready", "selected", "fit_report"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or state["model"] != "LateFusion"
            or _signature(state["parameters"]) != self._parameters_signature
            or state["stage"] not in ("initial", "text", "image", "fusion_weight", None)
            or type(state["ready"]) is not bool
            or not isinstance(state["selected"], dict)
            or set(state["selected"]) != {"text", "image"}
            or any(type(value) is not bool for value in state["selected"].values())
        ):
            raise ValueError("Invalid LateFusion checkpoint configuration or lifecycle.")
        stage, selected, ready, report = (
            state[key] for key in ("stage", "selected", "ready", "fit_report")
        )
        if (
            (stage == "initial" and any(selected.values()))
            or (stage == "text" and selected["image"])
            or (stage == "image" and not selected["text"])
            or (stage in ("fusion_weight", None) and not all(selected.values()))
            or (stage is None and not ready)
            or (ready and stage not in ("fusion_weight", None))
        ):
            raise ValueError("Invalid LateFusion selected peer state.")
        if ready:
            if (
                not isinstance(report, dict)
                or set(report) != {"alpha", "selection_metric", "scores", "validation_size", "seed"}
                or report["selection_metric"] not in ("accuracy", "macro_f1")
                or type(report["validation_size"]) is not int
                or report["validation_size"] < 1
                or type(report["seed"]) is not int
                or not 0 <= report["seed"] < 2**32
                or not isinstance(report["scores"], list)
                or len(report["scores"]) != len(self.parameters_spec["alpha_grid"])
                or any(not _finite(value) or not 0 <= value <= 100 for value in report["scores"])
                or not _finite(report["alpha"])
            ):
                raise ValueError("Invalid LateFusion validation report.")
            best = max(range(len(report["scores"])), key=report["scores"].__getitem__)
            if report["alpha"] != self.parameters_spec["alpha_grid"][best]:
                raise ValueError("LateFusion alpha does not match first-best validation selection.")
        elif report is not None:
            raise ValueError("Unfitted LateFusion cannot contain a validation report.")
        self._selected, self.fit_report = deepcopy(selected), deepcopy(report)
        self._trained = deepcopy(selected)
        self._pending_ready = ready
        self._apply_stage(stage)

    def _before_load(self, module, state, prefix, *args):
        self._ready = False
        self._pending_ready = False
        alpha = state.get(prefix + "alpha")
        if (
            not isinstance(alpha, torch.Tensor)
            or alpha.shape != ()
            or alpha.dtype != torch.float64
            or not torch.isfinite(alpha)
        ):
            raise ValueError("LateFusion alpha checkpoint must be one finite float64 scalar.")

    def _after_load(self, module, incompatible):
        expected = (
            self.fit_report["alpha"]
            if self._pending_ready
            else self.parameters_spec["alpha_grid"][0]
        )
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
            or float(self.alpha) != expected
        ):
            raise ValueError("LateFusion checkpoint is incomplete or alpha/report disagree.")
        self._apply_stage(self._stage)
        self._ready = self._pending_ready
