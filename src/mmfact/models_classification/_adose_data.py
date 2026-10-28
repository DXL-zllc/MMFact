"""ADOSE training-only domains, hidden labels and staged acquisition."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from mmfact.base import TrainingStage
from mmfact.data.collate import collate_news
from mmfact.models_classification._domains import DomainTargets


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _image_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ticket):
        index, domain, supervised, adversarial = ticket
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "text": row["text"],
            "image_path": row["image_path"],
            "label": row["label"] if supervised else -1,
            "metadata": {
                key: value
                for key, value in row.get("metadata", {}).items()
                if key in {"text_id", "image_id", "pair_id"}
            },
            "domain": domain,
            "supervised": supervised,
            "adversarial": adversarial,
        }


def _collate(rows):
    batch = collate_news(rows)
    batch["domain_labels"] = torch.tensor([row["domain"] for row in rows])
    for key in ("supervised", "adversarial"):
        batch[f"{key}_mask"] = torch.tensor([row[key] for row in rows], dtype=torch.bool)
    return batch


class _BalancedBatches(Sampler):
    """Refresh pools per epoch; cover every row and cycle only shorter domains."""

    def __init__(self, pool, batch_size, full):
        self.pool, self.batch_size, self.full = pool, batch_size, full
        self.steps = max(math.ceil(len(group) / batch_size) for group in pool.groups)

    def __len__(self):
        return self.steps

    def __iter__(self):
        pool = self.pool
        generator = torch.Generator().manual_seed(pool.seed + pool.epoch + 10000 * self.full)
        selected = set(pool.selected)
        target = pool.groups[pool.target_domain]
        remaining = [index for index in target if index not in selected]
        groups = [list(group) for group in pool.groups]
        if not self.full:
            # On tiny saturated pools, continue adversarial training on target inputs.
            groups[pool.target_domain] = remaining or list(target)
        streams = []
        for domain, group in enumerate(groups):
            supervised = self.full or domain != pool.target_domain
            streams.append((domain, group, supervised, True))
        if selected and not self.full:
            streams.append((pool.target_domain, sorted(selected), True, False))
        count = self.steps * self.batch_size
        shuffled = []
        for domain, group, supervised, adversarial in streams:
            indices = []
            while len(indices) < count:
                indices.extend(
                    group[index] for index in torch.randperm(len(group), generator=generator)
                )
            shuffled.append([(index, domain, supervised, adversarial) for index in indices[:count]])
        for step in range(self.steps):
            start = step * self.batch_size
            yield [
                ticket for stream in shuffled for ticket in stream[start : start + self.batch_size]
            ]


class TrainingPool(nn.Module):
    def __init__(self, domains, active, seed):
        super().__init__()
        self._domain_options = deepcopy(domains)
        self.domains = DomainTargets(**domains)
        if self.domains.num_domains != 4 or domains.get("max_labels") != 1:
            raise ValueError("ADOSE requires four single-label training domains.")
        self.active = deepcopy(active)
        self.target_domain = active["target_domain"]
        if type(self.target_domain) is not int or not 0 <= self.target_domain < 4:
            raise ValueError("ADOSE target_domain must be an integer in [0,3].")
        rounds = active["epochs"]
        if (
            not isinstance(rounds, list)
            or not rounds
            or any(type(epoch) is not int or epoch < 1 for epoch in rounds)
            or rounds != sorted(set(rounds))
            or type(active["ratio"]) not in (int, float)
            or not math.isfinite(active["ratio"])
            or not 0 < active["ratio"] <= 1
        ):
            raise ValueError("ADOSE acquisition epochs or ratio are invalid.")
        # True (the legacy default) reveals every target training label in a final stage; False
        # keeps the paper's label budget and selects the model on validation in a budgeted stage.
        self.full_supervision = active.get("full_supervision", True)
        if type(self.full_supervision) is not bool:
            raise ValueError("ADOSE active.full_supervision must be a boolean.")
        self.seed, self.epoch, self.full = seed, 0, False
        self.rows, self.groups, self.selected, self.history = [], [], [], []
        self.training_sha256 = None
        self._restored_ids = None
        self.register_load_state_dict_post_hook(self._restore_selection)

    def prepare(self, rows):
        self.rows, self.groups, self.selected, self.history = [], [], [], []
        self.epoch, self.full, self.training_sha256, self._restored_ids = 0, False, None, None
        self.domains = DomainTargets(**self._domain_options)
        rows = sorted(list(rows), key=lambda row: row["sample_id"])
        self.domains.prepare(rows)
        ids = [row["sample_id"] for row in rows]
        targets = self.domains.targets(ids)
        if not targets.sum(-1).eq(1).all():
            raise ValueError("ADOSE training domains must assign exactly one domain per sample.")
        assignments = targets.argmax(-1).tolist()
        groups = [[i for i, domain in enumerate(assignments) if domain == d] for d in range(4)]
        images = {
            row["image_path"]: row.get("metadata", {}).get("image_id")
            or _image_digest(row["image_path"])
            for row in rows
        }
        self.training_sha256 = _digest(
            [
                (row["sample_id"], row["text"], row["label"], images[row["image_path"]])
                for row in rows
            ]
        )
        self.rows, self.groups = rows, groups

    def loader(self, original, batch_size, full):
        if not self.rows:
            raise RuntimeError("Prepare the ADOSE training pool before constructing loaders.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("ADOSE per-domain batch size must be positive.")
        return DataLoader(
            _Rows(self.rows),
            batch_sampler=_BalancedBatches(self, batch_size, full),
            collate_fn=_collate,
            num_workers=original.num_workers,
            pin_memory=original.pin_memory,
            worker_init_fn=original.worker_init_fn,
            generator=torch.Generator().manual_seed(self.seed + 10000 * full),
        )

    def acquire(self, model):
        from mmfact.models_classification._adose_select import select_samples

        remaining = sorted(set(self.groups[self.target_domain]) - set(self.selected))
        count = min(
            math.ceil(len(self.groups[self.target_domain]) * self.active["ratio"]), len(remaining)
        )
        audit = {"epoch": self.epoch, "candidate_count": len(remaining), "selected_ids": []}
        if not remaining:
            self.history.append({**audit, "reason": "target-pool-exhausted"})
            return
        was_training = model.training
        model.eval()
        features = [[], [], []]
        try:
            with torch.no_grad():
                size = self.active["chunk_size"]
                for start in range(0, len(remaining), size):
                    rows = [self.rows[index] for index in remaining[start : start + size]]
                    batch = {
                        "text": [row["text"] for row in rows],
                        "image_path": [row["image_path"] for row in rows],
                    }
                    for parts, values in zip(features, model.selection_inputs(batch), strict=True):
                        parts.append(values.detach().cpu())
                inputs = tuple(torch.cat(parts) for parts in features)
                options = {
                    key: self.active[key]
                    for key in ("multiplier", "variances", "samples_per_variance", "chunk_size")
                }
                selected, selection_audit = select_samples(
                    inputs, model.selection_heads(), count, seed=self.seed + self.epoch, **options
                )
        finally:
            model.train(was_training)
        if (
            not isinstance(selected, torch.Tensor)
            or selected.dtype != torch.long
            or selected.shape != (count,)
            or ((selected < 0) | (selected >= len(remaining))).any()
            or selected.unique().numel() != count
        ):
            raise ValueError("ADOSE selector must return exactly k different candidate indices.")
        indices = [remaining[index] for index in selected.tolist()]
        self.selected.extend(indices)
        audit.update(selection_audit)
        audit["selected_ids"] = [self.rows[index]["sample_id"] for index in indices]
        self.history.append(audit)

    def reveal_all(self):
        self.selected = list(self.groups[self.target_domain])
        self.full = True

    def get_extra_state(self):
        ids = self.domains.get_extra_state()["sample_ids"]
        state = {
            "version": 1,
            "active": deepcopy(self.active),
            "seed": self.seed,
            "epoch": self.epoch,
            "full_supervision": self.full,
            "selected_ids": [ids[index] for index in self.selected],
            "history": deepcopy(self.history),
            "training_sha256": self.training_sha256,
        }
        state["sha256"] = _digest(state)
        return state

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("active") != self.active
            or state.get("sha256") != _digest({k: v for k, v in state.items() if k != "sha256"})
            or type(state.get("seed")) is not int
            or type(state.get("epoch")) is not int
            or state["epoch"] < 0
            or type(state.get("full_supervision")) is not bool
            or not isinstance(state.get("selected_ids"), list)
            or not isinstance(state.get("history"), list)
        ):
            raise ValueError("ADOSE acquisition resource or configuration is invalid.")
        # Child domains load after this module; ID resolution waits until needed.
        self.seed, self.epoch, self.full = state["seed"], state["epoch"], state["full_supervision"]
        self._restored_ids = list(state["selected_ids"])
        self.history = deepcopy(state["history"])
        self.training_sha256 = state["training_sha256"]

    def _restore_selection(self, module, incompatible_keys):
        if self._restored_ids is None:
            return
        ids = self.domains.get_extra_state()["sample_ids"]
        indices = {value: index for index, value in enumerate(ids)}
        if len(set(self._restored_ids)) != len(self._restored_ids) or any(
            value not in indices for value in self._restored_ids
        ):
            raise ValueError("ADOSE selected IDs do not belong to the saved training pool.")
        self.selected = [indices[value] for value in self._restored_ids]
        if ids:
            targets = self.domains.targets(ids)
            if not targets.sum(-1).eq(1).all():
                raise ValueError("ADOSE saved domains must assign exactly one domain per sample.")
            assignments = targets.argmax(-1).tolist()
            self.groups = [
                [i for i, domain in enumerate(assignments) if domain == d] for d in range(4)
            ]
            if not all(self.groups):
                raise ValueError("ADOSE requires all four training domains to be occupied.")
            if any(index not in self.groups[self.target_domain] for index in self.selected):
                raise ValueError("ADOSE selection must belong to the target training domain.")
            acquired = set()
            target = {ids[index] for index in self.groups[self.target_domain]}
            for round_index, record in enumerate(self.history):
                if not isinstance(record, dict) or round_index >= len(self.active["epochs"]):
                    raise ValueError("ADOSE saved acquisition history is invalid.")
                picked = record.get("selected_ids")
                remaining = len(target - acquired)
                budget = min(math.ceil(len(target) * self.active["ratio"]), remaining)
                if (
                    record.get("epoch") != self.active["epochs"][round_index]
                    or record.get("candidate_count") != remaining
                    or not isinstance(picked, list)
                    or len(picked) != budget
                    or len(set(picked)) != budget
                    or not set(picked) <= target - acquired
                ):
                    raise ValueError("ADOSE saved acquisition IDs, rounds or budgets are invalid.")
                acquired.update(picked)
            if (self.full and len(self.history) != len(self.active["epochs"])) or (
                not self.full and self.history and self.epoch < self.history[-1]["epoch"]
            ):
                raise ValueError("ADOSE saved stage and acquisition progress are inconsistent.")
            expected = target if self.full else acquired
            if set(self._restored_ids) != expected:
                raise ValueError("ADOSE selected IDs disagree with the saved acquisition history.")
            if not isinstance(self.training_sha256, str) or len(self.training_sha256) != 64:
                raise ValueError("ADOSE saved training identity is missing.")
        elif self.history or self.selected or self.training_sha256 is not None or self.full:
            raise ValueError("Unprepared ADOSE resources must be empty.")


class _ADOSETraining:
    def _init_training(self, parameters):
        self.pool = TrainingPool(parameters["domains"], parameters["active"], torch.initial_seed())
        self._stage = None
        self._curriculum_epochs = 0

    def prepare(self, train_dataset):
        self.pool.prepare(train_dataset)

    def _final_stage(self):
        """Second stage: full target labels (legacy) or the paper's label budget only."""
        return "full_supervision" if self.pool.full_supervision else "budgeted"

    def training_stages(self):
        return (TrainingStage("curriculum"), TrainingStage(self._final_stage()))

    def set_training_stage(self, stage):
        if stage not in (None, "curriculum", self._final_stage()):
            raise ValueError("Unknown ADOSE training stage.")
        self._stage = stage

    def training_stage_data(self, stage, train_data, val_data, options):
        final = stage != "curriculum"
        if final:
            if len(self.pool.history) != len(self.pool.active["epochs"]):
                raise ValueError("ADOSE curriculum must complete every acquisition round first.")
            if self.pool.full_supervision:
                self.pool.reveal_all()
            elif options["selection_metric"] == "last":
                raise ValueError("ADOSE budgeted stage selects the model on validation.")
        elif (
            options["selection_metric"] != "last"
            or options["epochs"] <= self.pool.active["epochs"][-1]
        ):
            raise ValueError(
                "ADOSE curriculum needs last selection and training after its final round."
            )
        else:
            self._curriculum_epochs = options["epochs"]
        return self.pool.loader(train_data, options["batch_size"], self.pool.full), (
            val_data if final else None
        )

    def on_epoch_start(self, epoch, total_epochs):
        # The budgeted stage continues the curriculum's epoch count, so shuffles never repeat
        # and the saved acquisition progress stays consistent with the epoch.
        offset = self._curriculum_epochs if self._stage == "budgeted" else 0
        self.pool.epoch = offset + epoch

    def on_epoch_end(self, epoch):
        if self._stage == "curriculum" and epoch in self.pool.active["epochs"]:
            self.pool.acquire(self)
