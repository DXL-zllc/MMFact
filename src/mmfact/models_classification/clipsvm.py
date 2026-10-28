"""Frozen paired CLIP vectors, linear SVM, and train-only sigmoid calibration."""

import json
import math
import warnings
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from mmfact import configs
from mmfact.base import AbstractDetector, TrainingStage
from mmfact.data.collate import collate_news
from mmfact.data.features import file_digest, fingerprint
from mmfact.materials.hub import HubMaterials
from mmfact.models_classification._classical_baselines import (
    _checksum,
    _fit_seed,
    _validate_identity,
)
from mmfact.registry import register_model

_RECIPE = {
    "name": "clip_svm",
    "version": 1,
    "input_dim": 1024,
    "fusion": "concat_text_then_image",
    "post_concat_normalization": False,
    "dtype": "float64",
    "logits": "[0,-(a*margin+b)]",
    "norm_atol": 1e-5,
}


def _parameters(p, dataset):
    if set(p) != {"feature_source", "text", "image", "svm", "calibration"}:
        raise ValueError("CLIPSVM requires feature_source/text/image/svm/calibration only.")
    if p["feature_source"] != {
        "model": "CLIPLinear",
        "text_role": "text_encoder",
        "image_role": "image_encoder",
        "mode": "prepared_only",
        "dtype": "float64",
        "fusion": "concat_text_then_image",
        "post_concat_normalization": False,
    }:
        raise ValueError("CLIPSVM requires the fixed paired CLIPLinear unit-vector source.")
    if p["calibration"] != {
        "method": "sigmoid",
        "n_splits": 5,
        "shuffle": True,
        "seed_source": "training_seed",
        "ensemble": False,
        "n_jobs": 1,
    }:
        raise ValueError("CLIPSVM requires five train-only folds and one final sigmoid model.")
    expected_model = (
        "openai/clip-vit-base-patch16"
        if dataset == "gossipcop"
        else "OFA-Sys/chinese-clip-vit-base-patch16"
    )
    for kind in ("text", "image"):
        branch = p[kind]
        if (
            branch.get("freeze") is not True
            or branch.get("normalize") is not True
            or branch.get("output") != "pooled"
            or branch.get("model_name") != expected_model
            or branch.get("augment")
            or branch.get("flip_probability", 0)
        ):
            raise ValueError(f"CLIPSVM has an incompatible frozen {kind} configuration.")
    if p["text"]["max_length"] != (77 if dataset == "gossipcop" else 52):
        raise ValueError("CLIPSVM text length must match the delivered CLIP vectors.")
    if p["image"].get("square_resize") is not False:
        raise ValueError("CLIPSVM uses the delivered CLIP image preprocessing.")
    svm = p["svm"]
    fixed = {"penalty": "l2", "loss": "squared_hinge", "dual": False, "fit_intercept": True}
    if set(svm) != set(fixed) | {
        "C",
        "tol",
        "max_iter",
        "class_weight",
        "intercept_scaling",
    } or any(svm[k] != v for k, v in fixed.items()):
        raise ValueError("CLIPSVM requires L2 squared-hinge primal LinearSVC with an intercept.")
    if svm["class_weight"] not in (None, "balanced"):
        raise ValueError("CLIPSVM class_weight must be None or balanced.")
    for key in ("C", "tol", "intercept_scaling"):
        if type(svm[key]) not in (int, float) or not math.isfinite(svm[key]) or svm[key] <= 0:
            raise ValueError(f"svm.{key} must be finite and positive.")
    if type(svm["max_iter"]) is not int or svm["max_iter"] < 1:
        raise ValueError("svm.max_iter must be a positive integer.")


@register_model("CLIPSVM")
class CLIPSVM(AbstractDetector):
    """CPU-only fit-stage detector; no neural encoder is constructed."""

    modalities = ("text", "image")

    def __init__(self, config):
        super().__init__(config)
        self.parameters_spec = deepcopy(config["model"]["parameters"])
        dataset = config["dataset"]["name"]
        if dataset not in {"weibo17", "weibo21", "gossipcop"}:
            raise ValueError("CLIPSVM supports the three published MMFact datasets.")
        _parameters(self.parameters_spec, dataset)
        options = config["preprocessing"]
        if (
            options.get("layout") != "hub"
            or options.get("mode") != "prepared"
            or not options.get("enabled")
            or config["training"]["device"] != "cpu"
        ):
            raise ValueError("CLIPSVM requires prepared Hub inputs and training.device=cpu.")
        root = Path(options["materials_root"]).expanduser().resolve()
        manifest = json.loads(
            (Path(configs.__file__).parent / "clipsvm_materials.json").read_text("utf-8")
        )
        if config["model"].get("provenance", {}).get("materials_revision") != manifest["revision"]:
            raise ValueError("CLIPSVM material revision differs from its pinned release manifest.")
        identity = {"repo_id": manifest["repo_id"], "revision": manifest["revision"], "files": {}}
        for entry in manifest["files"]:
            relative = entry["path"]
            if not relative.startswith(f"datasets/{dataset}/"):
                continue
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"CLIPSVM material is missing or outside its root: {relative}")
            if path.stat().st_size != entry["bytes"] or file_digest(path) != entry["sha256"]:
                raise ValueError(
                    f"CLIPSVM material bytes disagree with the fixed release: {relative}"
                )
            identity["files"][relative] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        if len(identity["files"]) != 11:
            raise ValueError("CLIPSVM expects eleven pinned input files per dataset.")
        self.material_identity = identity
        store = HubMaterials(root, dataset)
        self.matrices = {}
        for kind, role in (("text", "text_encoder"), ("image", "image_encoder")):
            matrix = store.feature_for("CLIPLinear", role)
            branch = self.parameters_spec[kind]
            output = (
                "projected_cls" if kind == "text" and dataset != "gossipcop" else "projected_pooler"
            )
            expected = {"model_name": branch["model_name"], "normalize": True, "output": output}
            if kind == "text":
                expected["max_length"] = branch["max_length"]
            matrix.require_settings(expected)
            if (
                matrix.input_kind != kind
                or len(matrix.shape) != 2
                or matrix.shape[1] != 512
                or matrix.entry.get("dtype") != "float32"
                or matrix.entry.get("mask")
            ):
                raise ValueError("CLIPSVM needs unmasked, projected global 512-D unit vectors.")
            matrix.rows  # Check complete index coverage and uniqueness before fitting.
            self.matrices[kind] = matrix
        self.resource = None
        self.register_buffer("_device", torch.empty(0), persistent=False)

    def training_stages(self):
        return (TrainingStage("fit", mode="fit"),)

    def calculate_loss(self, batch):
        raise RuntimeError("CLIPSVM uses a single fit stage, not gradient training.")

    def extract_features(self, batch):
        """Read classifier inputs without accessing labels, domains or raw images."""
        if self._device.device.type != "cpu":
            raise ValueError("CLIPSVM prediction and fitting are CPU-only.")
        values = []
        for kind in ("text", "image"):
            value, mask = self.matrices[kind].read_batch(batch, device="cpu", dtype=torch.float64)
            if mask is not None or tuple(value.shape) != (len(batch["sample_id"]), 512):
                raise ValueError("CLIPSVM feature batch must have shape [B,512] per modality.")
            array = value.numpy()
            if not np.allclose(
                np.linalg.norm(array, axis=1), 1.0, rtol=0, atol=_RECIPE["norm_atol"]
            ):
                raise ValueError("CLIPSVM cached inputs must already be unit normalized.")
            values.append(array)
        return np.ascontiguousarray(np.concatenate(values, axis=1), dtype=np.float64)

    @staticmethod
    def _logits(features, resource):
        margin = features @ resource["coef"].numpy().ravel() + resource["intercept"].item()
        z = -(resource["a"].item() * margin + resource["b"].item())
        if not np.isfinite(z).all():
            raise ValueError("CLIPSVM must produce finite calibrated logits.")
        return torch.from_numpy(np.column_stack((np.zeros_like(z), z)))

    def fit_stage(self, stage, train_dataset, val_loader, options):
        seed = _fit_seed(stage, options)
        # The validation loader is deliberately never inspected here.
        start = perf_counter()
        records = [train_dataset[i] for i in range(len(train_dataset))]
        ids = [r["sample_id"] for r in records]
        labels = [r["label"] for r in records]
        if (
            not ids
            or any(not isinstance(i, str) or not i.strip() for i in ids)
            or len(ids) != len(set(ids))
            or any(type(y) is not int or y not in (0, 1) for y in labels)
        ):
            raise ValueError(
                "CLIPSVM training requires unique string IDs and integer binary labels."
            )
        counts = [labels.count(0), labels.count(1)]
        if min(counts) < 5:
            raise ValueError("CLIPSVM needs at least five training samples per class.")
        size = options["batch_size"]
        features = np.concatenate(
            [
                self.extract_features(collate_news(records[i : i + size]))
                for i in range(0, len(records), size)
            ]
        )
        read_seconds = perf_counter() - start
        y = np.asarray(labels, dtype=np.int64)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y))
        assignment = np.full(len(ids), -1, dtype=np.int64)
        for fold, (train, held_out) in enumerate(folds):
            if (
                set(train) & set(held_out)
                or len(train) + len(held_out) != len(ids)
                or np.any(assignment[held_out] != -1)
                or len(np.unique(y[train])) != 2
                or len(np.unique(y[held_out])) != 2
            ):
                raise ValueError("Invalid calibration folds.")
            assignment[held_out] = fold
        if np.any(assignment < 0):
            raise ValueError("Calibration folds do not cover the training set.")
        estimator = CalibratedClassifierCV(
            estimator=LinearSVC(**self.parameters_spec["svm"], random_state=seed),
            method="sigmoid",
            cv=folds,
            ensemble=False,
            n_jobs=1,
        )
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(features, y)
        fit_seconds = perf_counter() - start
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise ValueError(
                "CLIPSVM calibration/SVM fit emitted ConvergenceWarning; candidate failed."
            )
        if estimator.classes_.tolist() != [0, 1] or len(estimator.calibrated_classifiers_) != 1:
            raise ValueError("Unexpected sklearn calibrated classifier structure or class order.")
        calibrated = estimator.calibrated_classifiers_[0]
        if len(calibrated.calibrators) != 1 or calibrated.estimator.classes_.tolist() != [0, 1]:
            raise ValueError("CLIPSVM expects exactly one binary sigmoid calibrator.")
        svm, sigmoid = calibrated.estimator, calibrated.calibrators[0]
        members = [
            {"sample_id": i, "label": int(label), "fold": int(fold)}
            for i, label, fold in zip(ids, y, assignment)
        ]
        resource = {
            "ready": True,
            "recipe": deepcopy(_RECIPE),
            "classes": [0, 1],
            "parameters_sha256": fingerprint(self.parameters_spec),
            "training_sha256": fingerprint(
                [
                    [
                        r["sample_id"],
                        r["label"],
                        r["metadata"]["text_id"],
                        r["metadata"]["image_id"],
                    ]
                    for r in records
                ]
            ),
            "num_samples": len(ids),
            "class_counts": counts,
            "fold_members": members,
            "folds_sha256": fingerprint(members),
            "materials": deepcopy(self.material_identity),
            "coef": torch.tensor(svm.coef_, dtype=torch.float64),
            "intercept": torch.tensor(svm.intercept_, dtype=torch.float64),
            "a": torch.tensor(float(sigmoid.a_), dtype=torch.float64),
            "b": torch.tensor(float(sigmoid.b_), dtype=torch.float64),
            "seed": seed,
            "n_iter": int(np.max(svm.n_iter_)),
            "versions": {
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": str(torch.__version__),
            },
        }
        native = estimator.predict_proba(features)
        reconstructed = self._logits(features, resource).softmax(dim=-1).numpy()
        error = float(np.max(np.abs(native - reconstructed)))
        if error > 1e-12 or not np.array_equal(native.argmax(axis=1), reconstructed.argmax(axis=1)):
            raise ValueError("Calibrator export differs from sklearn predict_proba.")
        resource["sha256"] = _checksum(resource)
        self._validate_resource(resource)
        self.resource = resource
        return {
            "num_samples": len(ids),
            "class_counts": counts,
            "seed": seed,
            "training_sha256": resource["training_sha256"],
            "folds_sha256": resource["folds_sha256"],
            "material_sha256": fingerprint(self.material_identity),
            "converged": True,
            "final_svm_n_iter": resource["n_iter"],
            "fold_svm_n_iter": None,
            "fold_iteration_note": "ensemble=false discards fold estimators",
            "internal_svm_fits": 6,
            "feature_read_seconds": read_seconds,
            "svm_and_calibration_fit_seconds": fit_seconds,
            "probability_export_max_abs_error": error,
            "versions": resource["versions"],
            "other_warnings": [w.category.__name__ for w in caught],
        }

    @torch.no_grad()
    def predict(self, batch):
        if self.resource is None:
            raise RuntimeError("CLIPSVM must be fitted or restored before prediction.")
        return self._logits(self.extract_features(batch), self.resource)

    def get_extra_state(self):
        return {
            "version": 1,
            "parameters": deepcopy(self.parameters_spec),
            "resource": deepcopy(self.resource),
        }

    def _validate_resource(self, resource):
        fields = {
            "ready",
            "recipe",
            "classes",
            "parameters_sha256",
            "training_sha256",
            "num_samples",
            "class_counts",
            "fold_members",
            "folds_sha256",
            "materials",
            "coef",
            "intercept",
            "a",
            "b",
            "seed",
            "n_iter",
            "versions",
            "sha256",
        }
        if not isinstance(resource, dict) or set(resource) != fields:
            raise ValueError("CLIPSVM checkpoint resource schema is invalid.")
        _validate_identity(resource, _RECIPE)
        if (
            resource["classes"] != [0, 1]
            or min(resource["class_counts"]) < 5
            or resource["parameters_sha256"] != fingerprint(self.parameters_spec)
            or resource["materials"] != self.material_identity
        ):
            raise ValueError("CLIPSVM checkpoint class, parameter or material identity differs.")
        for key, shape in (("coef", (1, 1024)), ("intercept", (1,)), ("a", ()), ("b", ())):
            value = resource[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or tuple(value.shape) != shape
                or value.requires_grad
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"CLIPSVM checkpoint {key} is not finite CPU float64 {shape}.")
        members = resource["fold_members"]
        if (
            not isinstance(members, list)
            or len(members) != resource["num_samples"]
            or fingerprint(members) != resource["folds_sha256"]
        ):
            raise ValueError("CLIPSVM checkpoint calibration fold identity is invalid.")
        ids, counts = set(), [0, 0]
        fold_counts = np.zeros((5, 2), dtype=int)
        for row in members:
            if (
                not isinstance(row, dict)
                or set(row) != {"sample_id", "label", "fold"}
                or not isinstance(row["sample_id"], str)
                or not row["sample_id"].strip()
                or row["sample_id"] in ids
                or type(row["label"]) is not int
                or row["label"] not in (0, 1)
                or type(row["fold"]) is not int
                or not 0 <= row["fold"] < 5
            ):
                raise ValueError("CLIPSVM checkpoint fold member is invalid.")
            ids.add(row["sample_id"])
            counts[row["label"]] += 1
            fold_counts[row["fold"], row["label"]] += 1
        if counts != resource["class_counts"] or np.any(fold_counts == 0):
            raise ValueError("CLIPSVM checkpoint fold/class coverage is invalid.")
        _fit_seed("fit", resource)
        if (
            type(resource["n_iter"]) is not int
            or not 1 <= resource["n_iter"] <= self.parameters_spec["svm"]["max_iter"]
        ):
            raise ValueError("CLIPSVM checkpoint final iteration count is invalid.")
        if (
            not isinstance(resource["versions"], dict)
            or set(resource["versions"]) != {"sklearn", "numpy", "scipy", "torch"}
            or any(not isinstance(v, str) or not v for v in resource["versions"].values())
        ):
            raise ValueError("CLIPSVM checkpoint numerical library versions are invalid.")

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "parameters", "resource"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or fingerprint(state["parameters"]) != fingerprint(self.parameters_spec)
        ):
            raise ValueError("CLIPSVM checkpoint parameters or version differ.")
        resource = deepcopy(state["resource"])
        if resource is not None:
            self._validate_resource(resource)
        self.resource = resource
