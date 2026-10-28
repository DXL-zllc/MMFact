"""Shared supervised training; test data never enters checkpoint selection."""

import json
import math
import re
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.config import _validate_json_value
from mmfact.evaluation.metrics import classification_metrics
from mmfact.evaluation.reporting import GROUP_FIELDS


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    temporary.replace(path)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


class _OptimizerSet:
    """Disjoint optimizers consume gradients from one completed backward pass."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


class Trainer:
    """A trainer instance owns one candidate, one seed, and one output directory."""

    def __init__(
        self, config: Mapping[str, Any], model: AbstractDetector, run_dir: Path, seed: int
    ):
        self.config = config
        self.model = model
        self.run_dir = Path(run_dir)
        self.seed = seed
        self.training_options = config["training"]
        reference = next(model.parameters(), None)
        if reference is None:
            reference = next(model.buffers(), torch.empty(0))
        self.device = reference.device

    def _precision(self):
        precision = self.training_options.get("precision", "full")
        if not isinstance(precision, str) or precision not in {"full", "fp16", "bf16"}:
            raise ValueError("training.precision must be full, fp16, or bf16.")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("FP16 precision requires CUDA.")
        if precision == "bf16":
            supported = self.device.type == "cpu"
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    supported = torch.cuda.is_bf16_supported()
            if not supported:
                raise ValueError("BF16 precision requires a supported CPU or CUDA device.")
        return precision

    def _autocast(self):
        precision = self._precision()
        if precision == "full":
            return nullcontext()
        return torch.autocast(
            self.device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16
        )

    def _optimizer(self):
        options = self.training_options
        trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
        if not trainable:
            raise ValueError("Training requires at least one trainable model parameter.")
        hook = getattr(self.model, "optimizer_parameter_groups", None)
        groups = (
            hook() if hook else {"default": (p for p in self.model.parameters() if p.requires_grad)}
        )
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("optimizer_parameter_groups must return a nonempty mapping.")
        overrides = options.get("parameter_groups", {})
        if set(overrides) - set(groups):
            raise ValueError("training.parameter_groups contains unknown model parameter groups.")
        excluded = set()
        if options.get("weight_decay_exclude_bias_norm", False):
            norm_types = (
                torch.nn.modules.batchnorm._NormBase,
                torch.nn.LayerNorm,
                torch.nn.GroupNorm,
                torch.nn.LocalResponseNorm,
            )
            excluded = {
                id(parameter)
                for module in self.model.modules()
                for name, parameter in module.named_parameters(recurse=False)
                if name == "bias" or isinstance(module, norm_types)
            }
        parameters, seen, settings_by_group = [], set(), []
        for name, values in groups.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Optimizer parameter group names must be nonempty strings.")
            try:
                values = list(values)
            except TypeError as error:
                raise ValueError(f"Optimizer parameter group {name!r} must be iterable.") from error
            if not values:
                raise ValueError(f"Optimizer parameter group {name!r} is empty.")
            for parameter in values:
                identity = id(parameter)
                if identity not in trainable or identity in seen:
                    raise ValueError("Optimizer groups require unique trainable model parameters.")
                seen.add(identity)
            settings = overrides.get(name, {})
            algorithm = settings.get("optimizer", options["optimizer"])
            inherited = (
                options.get("optimizer_parameters", {}) if algorithm == options["optimizer"] else {}
            )
            settings_by_group.append((algorithm, settings.get("optimizer_parameters", inherited)))
            rate = settings.get("learning_rate", options["learning_rate"])
            decay = settings.get("weight_decay", options["weight_decay"])
            for no_decay in (False, True):
                selected = [p for p in values if (id(p) in excluded) == no_decay]
                if selected:
                    parameters.append(
                        {
                            "params": selected,
                            "name": name,
                            "lr": rate,
                            "weight_decay": 0.0 if no_decay else decay,
                        }
                    )
        if seen != trainable:
            raise ValueError("Optimizer groups must cover all trainable model parameters.")
        partitions = []
        for group_name, settings in zip(groups, settings_by_group):
            matching = next((item for item in partitions if item[0] == settings), None)
            if matching is None:
                matching = (settings, [])
                partitions.append(matching)
            matching[1].extend(group for group in parameters if group["name"] == group_name)
        if options.get("precision", "full") != "full" and len(partitions) != 1:
            raise ValueError("AMP precision does not support multiple heterogeneous optimizers.")
        factories = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        if any(name == "adabelief" for name, _ in settings_by_group):
            from adabelief_pytorch import AdaBelief

            factories["adabelief"] = AdaBelief
        if any(name == "riemannian_adam" for name, _ in settings_by_group):
            from geoopt.optim import RiemannianAdam

            factories["riemannian_adam"] = RiemannianAdam
        optimizers = [
            factories[algorithm](
                selected,
                lr=options["learning_rate"],
                weight_decay=options["weight_decay"],
                **settings,
            )
            for (algorithm, settings), selected in partitions
        ]
        return optimizers[0] if len(optimizers) == 1 else _OptimizerSet(optimizers)

    @staticmethod
    def _loss(value: Tensor | Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        terms = value if isinstance(value, Mapping) else {"loss": value}
        if not terms:
            raise ValueError("calculate_loss returned an empty mapping.")
        for name, term in terms.items():
            if not isinstance(name, str) or not isinstance(term, Tensor) or term.ndim != 0:
                raise ValueError("Every loss must be a named scalar torch.Tensor.")
            if not torch.isfinite(term).item():
                raise ValueError(f"Non-finite loss term: {name}.")
        total = sum(terms.values())
        if not torch.isfinite(total).item():
            raise ValueError("Non-finite total loss.")
        return total, {name: float(term.detach().cpu()) for name, term in terms.items()}

    def predict(self, loader: DataLoader) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return self._predict_heads(loader, ())[""]

    def _predict_heads(self, loader, heads):
        self.model.eval()
        rows_by_head = {head: [] for head in heads or ("",)}
        with torch.no_grad():
            for batch in loader:
                inputs = move_to_device(batch, self.device)
                with self._autocast():
                    outputs = (
                        self.model.predict_heads(inputs)
                        if heads
                        else {"": self.model.predict(inputs)}
                    )
                if not isinstance(outputs, Mapping) or set(outputs) != set(rows_by_head):
                    raise ValueError("predict_heads keys must match the declared stage heads.")
                size = len(batch["sample_id"])
                metadata = batch.get("metadata", [{} for _ in range(size)])
                for head, logits in outputs.items():
                    if not isinstance(logits, Tensor) or tuple(logits.shape) != (size, 2):
                        raise ValueError(
                            "predict must return a torch.Tensor of shape [batch_size, 2]."
                        )
                    if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
                        raise ValueError("predict must return finite floating-point logits.")
                    values = logits if logits.dtype == torch.float64 else logits.float()
                    probabilities = values.softmax(dim=-1).cpu()
                    predictions = logits.argmax(dim=-1).cpu().tolist()
                    rows_by_head[head].extend(
                        {
                            "sample_id": sample_id,
                            "label": int(batch["labels"][index]),
                            "prediction": predictions[index],
                            "score_fake": float(probabilities[index, 1]),
                            **{
                                key: metadata[index][key]
                                for key in GROUP_FIELDS
                                if key in metadata[index]
                            },
                        }
                        for index, sample_id in enumerate(batch["sample_id"])
                    )
        return {
            head: (
                classification_metrics(
                    [row["label"] for row in rows],
                    [row["prediction"] for row in rows],
                    [row["score_fake"] for row in rows],
                ),
                rows,
            )
            for head, rows in rows_by_head.items()
        }

    def _phase_batches(self, loader, phases, interval):
        if interval == "epoch":
            for phase in phases:
                self.model.set_training_phase(phase)
                for batch in loader:
                    yield phase, batch
        else:
            for batch in loader:
                for phase in phases:
                    self.model.set_training_phase(phase)
                    yield phase, batch

    def fit(self, train_data: DataLoader, val_data: DataLoader) -> dict[str, Any]:
        """Select the earliest best validation epoch. Test data is not accepted."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model.prepare(train_data.dataset)
        self.model.to(self.device)
        stages = self.model.training_stages()
        if (
            not isinstance(stages, tuple)
            or not stages
            or any(
                not isinstance(stage, TrainingStage)
                or not isinstance(stage.name, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", stage.name) is None
                or (stage.settings is not None and not isinstance(stage.settings, str))
                or (stage.settings is not None and not stage.settings)
                or (stage.module is not None and not isinstance(stage.module, str))
                or (stage.module is not None and not stage.module)
                or not isinstance(stage.heads, tuple)
                or any(
                    not isinstance(head, str)
                    or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", head) is None
                    for head in stage.heads
                )
                or len(set(stage.heads)) != len(stage.heads)
                or (stage.module is not None and bool(stage.heads))
                or not isinstance(stage.mode, str)
                or stage.mode not in {"gradient", "fit"}
                for stage in stages
            )
            or len({stage.name for stage in stages}) != len(stages)
        ):
            raise ValueError("training_stages must return unique, valid TrainingStage entries.")
        if any(stage.mode == "fit" and (stage.module or stage.heads) for stage in stages):
            raise ValueError("Fit stages require full-model checkpoints without module or heads.")
        if all(stage.mode == "fit" for stage in stages) and any(
            name.startswith("training.")
            for name in self.config.get("search", {}).get("parameters", {})
        ):
            raise ValueError("Training search dimensions do not apply to all-fit models.")
        overrides = self.config["training"].get("stage_overrides", {})
        keys = {stage.settings or stage.name for stage in stages}
        if set(overrides) - keys:
            raise ValueError("training.stage_overrides contains unknown stage settings keys.")
        legacy = stages == (TrainingStage("detection"),) and not overrides
        results = []
        for stage in stages:
            is_fit = stage.mode == "fit"
            self.model.set_training_stage(stage.name)
            self.model.zero_grad(set_to_none=True)
            if is_fit and any(parameter.requires_grad for parameter in self.model.parameters()):
                raise ValueError("Fit stages require all torch parameters to be frozen.")
            options = deepcopy(self.config["training"])
            specified = overrides.get(stage.settings or stage.name, {})
            if is_fit and set(specified) - {"selection_metric", "min_delta", "eval_batch_size"}:
                raise ValueError(
                    "Fit stage overrides only support selection metric and eval batch."
                )
            options.update(specified)
            options.setdefault("selection_metric", self.config["evaluation"]["selection_metric"])
            options.setdefault("min_delta", 0.0)
            stage_train, stage_val = self.model.training_stage_data(
                stage.name, train_data, val_data, options
            )
            metric, minimum = options.pop("selection_metric"), options.pop("min_delta")
            if (
                not isinstance(metric, str)
                or metric not in {"loss", "last", "accuracy", "macro_f1"}
                or (
                    type(minimum) not in (float, int)
                    or not math.isfinite(minimum)
                    or minimum < 0
                    or (metric == "last" and minimum != 0)
                )
            ):
                raise ValueError("Stage selection_metric/min_delta are invalid.")
            legacy = legacy and metric == self.config["evaluation"]["selection_metric"]
            if is_fit:
                if metric not in {"accuracy", "macro_f1"} or minimum != 0:
                    raise ValueError("Fit stages require accuracy/macro_f1 and min_delta=0.")
                if (
                    not isinstance(stage_train, DataLoader)
                    or stage_train.dataset is not train_data.dataset
                    or not len(train_data.dataset)
                ):
                    raise ValueError("Fit stages require the original nonempty training dataset.")
                if (
                    not isinstance(stage_val, DataLoader)
                    or not isinstance(val_data, DataLoader)
                    or stage_val.dataset is not val_data.dataset
                    or not len(stage_val.dataset)
                    or stage_val.drop_last
                    or type(stage_val.sampler) is not SequentialSampler
                    or type(stage_val.batch_sampler) is not BatchSampler
                ):
                    raise ValueError("Fit stages require complete sequential validation loaders.")
            elif (
                not isinstance(stage_train, DataLoader)
                or not len(stage_train)
                or (
                    (stage_val is not None or metric != "last")
                    and (not isinstance(stage_val, DataLoader) or not len(stage_val))
                )
            ):
                raise ValueError(
                    "Stage loaders must be nonempty; only last selection permits val=None."
                )
            for key, loader, original in (
                ("batch_size", stage_train, train_data),
                ("eval_batch_size", stage_val, val_data),
            ):
                if key not in specified or loader is None:
                    continue
                if loader.batch_size is None:
                    if loader is original:
                        raise ValueError(
                            f"Stage {stage.name} must build a custom loader for {key} overrides."
                        )
                elif loader.batch_size != specified[key]:
                    raise ValueError(
                        f"Stage {stage.name} loader does not apply {key}={specified[key]}."
                    )
            directory = self.run_dir if legacy else self.run_dir / "stages" / stage.name
            worker = Trainer(self.config, self.model, directory, self.seed)
            results.append(
                {
                    "stage": stage.name,
                    **(
                        worker._fit_once(stage.name, train_data.dataset, stage_val, options, metric)
                        if is_fit
                        else worker._fit(
                            stage_train,
                            stage_val,
                            options,
                            metric,
                            minimum,
                            stage.module,
                            stage.heads,
                        )
                    ),
                }
            )
        self.model.set_training_stage(None)
        if legacy:
            return {key: value for key, value in results[0].items() if key != "stage"}
        # Non-gradient fitting deliberately ignores inherited precision options.
        if stages[-1].mode == "fit":
            self.training_options = {**self.training_options, "precision": "full"}
        metrics, rows = self.predict(val_data)
        metric = self.config["evaluation"]["selection_metric"]
        self._save_checkpoint(results[-1]["best_epoch"], metric, metrics, stages=results)
        write_predictions(self.run_dir / "validation_predictions.jsonl", rows)
        write_json(self.run_dir / "stages.json", results)
        return {
            "seed": self.seed,
            "best_epoch": results[-1]["best_epoch"],
            "validation": metrics,
            "checkpoint": str((self.run_dir / "best.pt").resolve()),
            "stages": results,
        }

    def _fit_once(self, stage, train_dataset, val_data, options, metric):
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Fit stages require all torch parameters to be frozen.")
        parameters = tuple(self.model.parameters())
        self.training_options = {**options, "precision": "full"}
        self.model.eval()
        runtime = {
            **deepcopy(options),
            "selection_metric": metric,
            "min_delta": 0,
            "seed": self.seed,
        }
        with torch.no_grad():
            report = self.model.fit_stage(stage, train_dataset, val_data, runtime)
        if not isinstance(report, dict):
            raise ValueError("fit_stage must return a JSON-compatible report dict.")
        _validate_json_value(report, "fit_report")
        report = json.loads(json.dumps(report, allow_nan=False))
        if tuple(map(id, self.model.parameters())) != tuple(map(id, parameters)) or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise ValueError("fit_stage must not add or unfreeze torch parameters.")
        metrics, rows = self.predict(val_data)
        details = {
            "mode": "fit",
            "fit_calls": 1,
            "fit_report": report,
            "unused_training_options": sorted(
                set(options)
                - {
                    "device",
                    "seeds",
                    "eval_batch_size",
                    "num_workers",
                    "stage_overrides",
                    "selection_metric",
                    "min_delta",
                }
            ),
        }
        self._save_checkpoint(None, metric, metrics, **details)
        write_predictions(self.run_dir / "validation_predictions.jsonl", rows)
        write_json(self.run_dir / "fit.json", details)
        checkpoint = self._read_checkpoint(self.run_dir / "best.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return {
            "seed": self.seed,
            "best_epoch": None,
            "validation": metrics,
            "checkpoint": str((self.run_dir / "best.pt").resolve()),
            **details,
        }

    def _save_checkpoint(self, epoch, metric, metrics, *, module=None, **extra):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_state_dict": (
                    self.model.get_submodule(module) if module else self.model
                ).state_dict(),
                "model_config": dict(self.config["model"]),
                "seed": self.seed,
                "epoch": epoch,
                "selection_metric": metric,
                "validation_metrics": metrics,
                **({"stage_module": module} if module else {}),
                **extra,
            },
            self.run_dir / "best.pt",
        )

    def _validation_loss(self, loader):
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                with self._autocast():
                    loss, _ = self._loss(
                        self.model.validation_loss(move_to_device(batch, self.device))
                    )
                size = self.model.loss_weight(batch)
                if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                    raise ValueError("loss_weight must be a positive integer.")
                total += float(loss.cpu()) * size
                count += size
        if not count:
            raise ValueError("Stage validation data must be nonempty.")
        return total / count

    def _fit(self, train_data, val_data, options, metric, min_delta, module=None, heads=()):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if module:
            selected = {
                id(parameter) for parameter in self.model.get_submodule(module).parameters()
            }
            if any(
                parameter.requires_grad and id(parameter) not in selected
                for parameter in self.model.parameters()
            ):
                raise ValueError("Stage module must own all currently trainable parameters.")
        if heads:
            if module or metric not in {"accuracy", "macro_f1"}:
                raise ValueError("Independent stage heads require accuracy/macro_f1 and no module.")
            owned = set()
            for head in heads:
                parameters = {
                    id(parameter) for parameter in self.model.get_submodule(head).parameters()
                }
                if not parameters or owned & parameters:
                    raise ValueError(
                        "Independent stage heads must own nonempty disjoint parameters."
                    )
                owned.update(parameters)
            if any(p.requires_grad and id(p) not in owned for p in self.model.parameters()):
                raise ValueError("Independent stage heads must cover all trainable parameters.")
        self.training_options = options
        precision = self._precision()
        accumulation = options.get("gradient_accumulation_steps", 1)
        if type(accumulation) is not int or accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        phases = self.model.training_phases()
        if (
            not phases
            or any(not isinstance(phase, str) or not phase for phase in phases)
            or len(set(phases)) != len(phases)
        ):
            raise ValueError("training_phases must contain unique phase names.")
        if accumulation > 1 and len(phases) != 1:
            raise ValueError("Gradient accumulation requires a single training phase.")
        optimizer = self._optimizer()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        scaler = None
        if precision == "fp16":
            scaler = (
                torch.amp.GradScaler("cuda")
                if hasattr(torch.amp, "GradScaler")
                else torch.cuda.amp.GradScaler()
            )
        base_rates = [group["lr"] for group in optimizer.param_groups]
        schedule = options.get("scheduler") or {}
        validation_monitor = None
        if schedule.get("name") == "plateau" and schedule["monitor"].startswith("validation."):
            validation_monitor = schedule["monitor"].removeprefix("validation.")
            if heads or metric == "last" or (metric == "loss" and validation_monitor != "loss"):
                raise ValueError("Validation plateau monitor is unavailable for this stage.")
        plateau = []
        if schedule.get("name") == "plateau":
            for selected in getattr(optimizer, "optimizers", [optimizer]):
                plateau.append(
                    torch.optim.lr_scheduler.ReduceLROnPlateau(
                        selected,
                        mode="max" if validation_monitor in {"accuracy", "macro_f1"} else "min",
                        factor=schedule["factor"],
                        patience=schedule["patience"],
                        min_lr=[
                            schedule["min_lr"] * group["lr"] / options["learning_rate"]
                            for group in selected.param_groups
                        ],
                    )
                )
        interval = self.model.training_phase_interval()
        if not isinstance(interval, str) or interval not in {"epoch", "batch"}:
            raise ValueError("training_phase_interval must be epoch or batch.")
        batches = len(train_data) * len(phases)
        windows = math.ceil(batches / accumulation)
        step, total_steps = 0, options["epochs"] * windows
        if schedule.get("name") == "warmup_cosine" and schedule["warmup_steps"] > total_steps:
            raise ValueError("warmup_cosine warmup_steps must not exceed the stage's total steps.")
        best_score, best_epoch = -math.inf, 0
        best_metrics: dict[str, Any] = {}
        head_results = {}
        history = []
        for epoch in range(1, options["epochs"] + 1):
            self.model.train()
            self.model.on_epoch_start(epoch, options["epochs"])
            totals: dict[str, float] = {}
            counts: dict[str, int] = {}
            phase_steps = dict.fromkeys(phases, 0)
            skipped = {"empty": 0, "overflow": 0}
            window_weight = 0
            for index, (phase, batch) in enumerate(
                self._phase_batches(train_data, phases, interval)
            ):
                if index % accumulation == 0:
                    self.model.zero_grad(set_to_none=True)
                    window_weight = 0
                if index % accumulation == 0 and schedule.get("name") in {
                    "inverse",
                    "step",
                    "linear_warmup",
                    "warmup_cosine",
                    "cosine",
                }:
                    if schedule["name"] == "step":
                        factor = schedule["gamma"] ** (step // schedule["step_size"])
                    elif schedule["name"] == "warmup_cosine":
                        warmup = schedule["warmup_steps"]
                        if step < warmup:
                            start = schedule["warmup_start_lr"] / options["learning_rate"]
                            # A one-step warmup has only its peak endpoint.
                            factor = 1 if warmup == 1 else start + (1 - start) * step / (warmup - 1)
                        else:
                            minimum = schedule["min_lr"] / options["learning_rate"]
                            peak_step = max(0, warmup - 1)
                            progress = (step - peak_step) / max(1, total_steps - 1 - peak_step)
                            factor = (
                                minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
                            )
                    elif schedule["name"] == "cosine":
                        minimum = schedule["min_lr"] / options["learning_rate"]
                        factor = (
                            minimum
                            + (1 - minimum) * (1 + math.cos(math.pi * step / total_steps)) / 2
                        )
                    else:
                        if schedule["name"] == "inverse":
                            factor = (1 + schedule["gamma"] * step / total_steps) ** -schedule[
                                "power"
                            ]
                        else:
                            warmup = int(total_steps * schedule["warmup_ratio"])
                            factor = (
                                step / max(1, warmup)
                                if step < warmup
                                else (total_steps - step) / max(1, total_steps - warmup)
                            )
                    for group, rate in zip(optimizer.param_groups, base_rates):
                        group["lr"] = rate * factor
                with self._autocast():
                    value = self.model.calculate_loss(move_to_device(batch, self.device))
                    if value is not None:
                        loss, terms = self._loss(value)
                if value is not None:
                    size = self.model.loss_weight(batch)
                    if type(size) is not int or size < 1:
                        raise ValueError("loss_weight must be a positive integer.")
                    weighted = loss * size if accumulation > 1 else loss
                    (scaler.scale(weighted) if scaler else weighted).backward()
                    window_weight += size
                    for name, value in terms.items():
                        totals[name] = totals.get(name, 0.0) + value * size
                        counts[name] = counts.get(name, 0) + size
                if (index + 1) % accumulation != 0 and index + 1 != batches:
                    continue
                # Scheduler positions include empty and AMP-overflow windows.
                step += 1
                if not window_weight:
                    skipped["empty"] += 1
                    continue
                if scaler:
                    scaler.unscale_(optimizer)
                if accumulation > 1:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(window_weight)
                finite = precision == "full" or all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in parameters
                )
                if not finite and scaler is None:
                    raise ValueError("Non-finite gradients in BF16 training.")
                if finite and options["gradient_clip_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        parameters, options["gradient_clip_norm"], error_if_nonfinite=True
                    )
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if finite:
                    phase_steps[phase] += 1
                else:
                    skipped["overflow"] += 1
            if not counts:
                raise ValueError("No applicable training batches in this epoch.")
            self.model.on_epoch_end(epoch)
            if heads:
                predictions = self._predict_heads(val_data, heads)
                val_metrics = {head: values[0] for head, values in predictions.items()}
                val_rows = None
            elif metric == "loss":
                val_metrics, val_rows = {"loss": self._validation_loss(val_data)}, None
            elif metric == "last":
                val_metrics, val_rows = {}, None
            else:
                val_metrics, val_rows = self.predict(val_data)
            if validation_monitor == "loss" and "loss" not in val_metrics:
                val_metrics["loss"] = self._validation_loss(val_data)
            improved = False
            if not heads:
                score = (
                    epoch
                    if metric == "last"
                    else -val_metrics[metric]
                    if metric == "loss"
                    else val_metrics[metric]
                )
                improved = (
                    epoch == options["epochs"]
                    if metric == "last"
                    else score > best_score + min_delta
                )
            if improved:
                best_score, best_epoch, best_metrics = score, epoch, val_metrics
                self._save_checkpoint(epoch, metric, val_metrics, module=module)
                if val_rows is not None:
                    write_predictions(self.run_dir / "validation_predictions.jsonl", val_rows)
            selected_heads = {}
            for head in heads:
                metrics, rows = predictions[head]
                previous = head_results.get(head)
                selected_heads[head] = (
                    previous is None or metrics[metric] > previous["validation"][metric] + min_delta
                )
                if selected_heads[head]:
                    directory = self.run_dir / "heads" / head
                    worker = Trainer(self.config, self.model, directory, self.seed)
                    worker._save_checkpoint(epoch, metric, metrics, module=head)
                    write_predictions(directory / "validation_predictions.jsonl", rows)
                    head_results[head] = {
                        "best_epoch": epoch,
                        "validation": metrics,
                        "checkpoint": str((directory / "best.pt").resolve()),
                    }
            if any(selected_heads.values()):
                improved, best_epoch = True, epoch
            record = {
                "epoch": epoch,
                "train_loss": {name: total / counts[name] for name, total in totals.items()},
                "phase_steps": phase_steps,
                "phase_interval": interval,
                "optimizer_steps": sum(phase_steps.values()),
                "scheduled_steps": windows,
                "skipped_steps": skipped,
                "validation": val_metrics,
                "selected": improved,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                **({"selected_heads": selected_heads} if heads else {}),
            }
            if plateau:
                monitor = schedule["monitor"]
                if validation_monitor:
                    monitored = val_metrics[validation_monitor]
                elif monitor in record["train_loss"]:
                    monitored = record["train_loss"][monitor]
                else:
                    raise ValueError(f"Scheduler monitor {monitor!r} is not a training loss term.")
                record["scheduler_monitor"] = {"name": monitor, "value": monitored}
                for scheduler in plateau:
                    scheduler.step(monitored)
            history.append(record)
            write_json(self.run_dir / "history.json", history)
            if heads:
                epochs = {head: item["best_epoch"] for head, item in head_results.items()}
                message = f"epoch={epoch} head_best_epochs={epochs}"
            else:
                selection = (
                    "selection=last"
                    if metric == "last"
                    else f"validation_{metric}={val_metrics[metric]:.6f}"
                )
                message = f"epoch={epoch} {selection} best_epoch={best_epoch}"
            with (self.run_dir / "train.log").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            if metric != "last" and epoch - best_epoch >= options["patience"]:
                break
        if not any(record["optimizer_steps"] for record in history):
            raise ValueError("No optimizer updates completed; all training windows were skipped.")
        if heads:
            for head, result in head_results.items():
                checkpoint = self._read_checkpoint(Path(result["checkpoint"]), module=head)
                self.model.get_submodule(head).load_state_dict(
                    checkpoint["model_state_dict"], strict=True
                )
            best_metrics = {head: item["validation"] for head, item in head_results.items()}
            self._save_checkpoint(epoch, metric, best_metrics, heads=head_results)
            return {
                "seed": self.seed,
                "best_epoch": None,
                "completed_epoch": epoch,
                "validation": best_metrics,
                "heads": head_results,
                "checkpoint": str((self.run_dir / "best.pt").resolve()),
            }
        checkpoint = self._read_checkpoint(self.run_dir / "best.pt", module=module)
        target = self.model.get_submodule(module) if module else self.model
        target.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return {
            "seed": self.seed,
            "best_epoch": best_epoch,
            "validation": best_metrics,
            "checkpoint": str((self.run_dir / "best.pt").resolve()),
        }

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Load framework weights only, checking model name and structural parameters."""
        checkpoint = self._read_checkpoint(path)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return checkpoint

    def _read_checkpoint(self, path, *, module=None):
        # Keep the serialized state on CPU while copying into the live model;
        # mapping both copies to CUDA can double peak VRAM usage.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
            raise ValueError(
                "Unsupported checkpoint: expected an MMFact format_version=1 checkpoint."
            )
        if checkpoint.get("stage_module") != module:
            raise ValueError("Checkpoint stage module does not match the requested model state.")
        saved = checkpoint["model_config"]
        current = self.config["model"]
        if saved["name"].lower() != current["name"].lower():
            raise ValueError("Checkpoint model name does not match the configured model.")
        if saved.get("parameters", {}) != current.get("parameters", {}):
            raise ValueError("Checkpoint model.parameters do not match the configured model.")
        if saved.get("module") != current.get("module"):
            raise ValueError("Checkpoint model.module does not match the configured model.")
        if saved.get("provenance", {}) != current.get("provenance", {}):
            raise ValueError("Checkpoint model.provenance does not match the configured model.")
        return checkpoint
