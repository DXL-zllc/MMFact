"""One experiment entry point for configuration, selection and final evaluation."""

import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from mmfact.config import expand_search, load_config
from mmfact.data import collate_news, load_datasets
from mmfact.data.preprocessing import configure_preprocessing, preprocess
from mmfact.evaluation.metrics import aggregate_metrics
from mmfact.evaluation.reporting import write_breakdown, write_breakdown_summary
from mmfact.registry import get_model_class
from mmfact.trainer import Trainer, write_json, write_predictions
from mmfact.version import __version__


def init_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", requested):
        raise ValueError("training.device must be auto, cpu, cuda[:N], or mps.")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("training.device must be auto, cpu, cuda[:N], or mps.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {device.index} does not exist.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return device


def _loader(dataset, config: Mapping[str, Any], seed: int, *, train: bool = False):
    options = config["training"]
    return DataLoader(
        dataset,
        batch_size=options["batch_size"] if train else options["eval_batch_size"],
        shuffle=train,
        num_workers=options["num_workers"],
        collate_fn=collate_news,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_worker_seed,
    )


def _data_manifest(datasets, *, hash_images: bool = False) -> dict[str, Any]:
    result = {}
    image_digests = {}
    for split, dataset in datasets.items():
        labels = [record["label"] for record in dataset.records]
        # Manifest/record hashes are cheap; reading every image for a byte-level
        # fingerprint is explicit because the real datasets can be very large.
        digest = hashlib.sha256()
        for record in dataset.records:
            image = record["image_path"]
            if hash_images and image not in image_digests:
                image_digest = hashlib.sha256()
                with Path(image).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        image_digest.update(chunk)
                image_digests[image] = image_digest.hexdigest()
            canonical = dict(record)
            canonical["image_path"] = (
                image_digests[image] if hash_images else str(Path(image).relative_to(dataset.root))
            )
            digest.update(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
        result[split] = {
            "path": str(dataset.manifest_path),
            "manifest_sha256": hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest(),
            "content_sha256": digest.hexdigest(),
            "image_bytes_hashed": hash_images,
            "num_samples": len(dataset),
            "num_real": labels.count(0),
            "num_fake": labels.count(1),
        }
    return result


def _environment(device: torch.device) -> dict[str, Any]:
    versions = {}
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "gensim",
        "jieba",
        "scikit-learn",
        "geoopt",
        "peft",
        "accelerate",
        "bitsandbytes",
        "PyYAML",
        "Pillow",
        "numpy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "mmfact_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "packages": versions,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "note": "Seeds are controlled; exact reproducibility across hardware is not guaranteed.",
    }


def quick_start(
    model: str = "EANN",
    dataset: str = "weibo17",
    config_files: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    *,
    dry_run: bool = False,
    mode: str = "train",
    checkpoint: str | Path | None = None,
    input_mode: str | None = None,
    materials_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train, evaluate existing weights, or prepare reusable model inputs.

    dry_run reads and validates data and configuration without instantiating a
    model or creating run artifacts. No datasets or model weights are downloaded.
    """
    if mode not in {"train", "evaluate", "preprocess"}:
        raise ValueError("mode must be train, evaluate, or preprocess.")
    if mode == "evaluate" and checkpoint is None:
        raise ValueError("--mode evaluate requires --checkpoint.")
    if mode != "evaluate" and checkpoint is not None:
        raise ValueError(
            "--checkpoint is only supported with --mode evaluate; resume is not supported."
        )
    config = load_config(model, dataset, config_files, overrides)
    if input_mode is not None:
        config["preprocessing"]["mode"] = input_mode
    from mmfact.config import _deep_merge, _read_yaml, _set_dotted, validate_config
    from mmfact.data.cache_policy import cache_policy
    from mmfact.materials.prepared import load_material_config

    root = materials_root or config["preprocessing"].get("materials_root")
    if root is None and dataset in {"weibo17", "weibo21", "gossipcop"}:
        default = Path("materials").resolve()
        if (default / "datasets" / dataset / "feature_catalog.json").is_file():
            root = default
    if root is not None:
        explicit = {}
        for path in config_files:
            _deep_merge(explicit, _read_yaml(Path(path).expanduser()))
        for override in overrides:
            key, value = override.split("=", 1)
            _set_dotted(explicit, key, yaml.safe_load(value))
        if input_mode is not None:
            explicit.setdefault("preprocessing", {})["mode"] = input_mode
        config = load_material_config(config, root, overrides=explicit)
    validate_config(config)
    with cache_policy(config["preprocessing"].get("mode", "raw")):
        return _run(config, dry_run=dry_run, mode=mode, checkpoint=checkpoint)


def _run(config, *, dry_run, mode, checkpoint):
    """Execute one resolved recipe under its explicit cache consumption policy."""
    if mode == "preprocess" and not config["preprocessing"]["enabled"]:
        raise ValueError("--mode preprocess requires preprocessing.enabled=true.")
    candidates = expand_search(config)
    detector_class = None
    if not dry_run:
        detector_class = get_model_class(config["model"]["name"], config["model"].get("module"))
    datasets = load_datasets(
        config["dataset"], splits=("test",) if mode == "evaluate" else ("train", "val", "test")
    )
    manifest = _data_manifest(datasets, hash_images=config["dataset"]["hash_images"])
    if dry_run:
        return {
            "status": "dry_run",
            "mode": mode,
            "config": config,
            "datasets": manifest,
            "num_candidates": len(candidates),
            "num_seeds": len(config["training"]["seeds"]),
            "note": "Configuration/data validation only; no model was loaded or trained.",
        }

    device = resolve_device(config["training"]["device"])
    if mode == "preprocess":
        init_seed(config["training"]["seeds"][0])
        model_instance = detector_class(config).to(device)
        cache = preprocess(model_instance, datasets, config["preprocessing"])
        return {
            "status": "preprocessed",
            "model": config["model"]["name"],
            "dataset": config["dataset"]["name"],
            "cache": cache,
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output_dir"]) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
    write_json(run_dir / "data_manifest.json", manifest)

    if mode == "evaluate":
        path = Path(checkpoint).resolve()
        # Read the saved seed before constructing the adapter, whose processors
        # may consume RNG state. Only trusted local framework weights are used.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or saved.get("format_version") != 1:
            raise ValueError("Unsupported MMFact checkpoint.")
        seed = saved["seed"]
        init_seed(seed)
        trainer = Trainer(config, detector_class(config).to(device), run_dir, seed)
        trainer.load_checkpoint(path)
        configure_preprocessing(trainer.model, datasets, config["preprocessing"])
        metrics, rows = trainer.predict(_loader(datasets["test"], config, seed))
        write_predictions(run_dir / "test_predictions.jsonl", rows)
        write_json(run_dir / "test_metrics.json", metrics)
        write_breakdown(
            run_dir, "test", rows,
            model=config["model"]["name"], dataset=config["dataset"]["name"], seed=seed,
        )
        write_json(run_dir / "environment.json", _environment(device))
        summary = {
            "status": "completed",
            "mode": mode,
            "run_dir": str(run_dir),
            "checkpoint": str(path),
            "seed": seed,
            "test": metrics,
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "experiment.json", {"mode": mode, "checkpoint": str(path)})
        return summary

    trials = []
    metric = config["evaluation"]["selection_metric"]
    for index, candidate in enumerate(candidates):
        seed_results = []
        trial_dir = run_dir / "trials" / f"trial_{index:03d}"
        trial_dir.mkdir(parents=True)
        (trial_dir / "config.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False), "utf-8")
        for seed in candidate["training"]["seeds"]:
            init_seed(seed)
            model_instance = detector_class(candidate).to(device)
            configure_preprocessing(model_instance, datasets, candidate["preprocessing"])
            trainer = Trainer(candidate, model_instance, trial_dir / f"seed_{seed}", seed)
            seed_results.append(
                trainer.fit(
                    _loader(datasets["train"], candidate, seed, train=True),
                    _loader(datasets["val"], candidate, seed + 1),
                )
            )
            # Release the previous adapter before allocating the next seed's
            # model, rather than requiring two backbones to fit in GPU memory.
            del trainer, model_instance
            gc.collect()
        trials.append(
            {
                "index": index,
                "seeds": seed_results,
                "mean_validation_score": mean(
                    result["validation"][metric] for result in seed_results
                ),
            }
        )
        write_json(
            run_dir / "experiment.json",
            {"mode": mode, "selection_metric": metric, "trials": trials},
        )

    # max returns the first maximum: grid ties retain the earliest candidate.
    selected = max(trials, key=lambda trial: trial["mean_validation_score"])
    selected_config = candidates[selected["index"]]
    test_results = []
    breakdowns = {"validation": [], "test": []}
    for result in selected["seeds"]:
        seed = result["seed"]
        init_seed(seed)
        trainer = Trainer(
            selected_config,
            detector_class(selected_config).to(device),
            Path(result["checkpoint"]).parent,
            seed,
        )
        trainer.load_checkpoint(Path(result["checkpoint"]))
        configure_preprocessing(trainer.model, datasets, selected_config["preprocessing"])
        metrics, rows = trainer.predict(_loader(datasets["test"], selected_config, seed + 2))
        write_predictions(trainer.run_dir / "test_predictions.jsonl", rows)
        write_json(trainer.run_dir / "test_metrics.json", metrics)
        with (trainer.run_dir / "validation_predictions.jsonl").open(encoding="utf-8") as handle:
            validation_rows = [json.loads(line) for line in handle]
        for split, predictions in (("validation", validation_rows), ("test", rows)):
            breakdowns[split].append(write_breakdown(
                trainer.run_dir, split, predictions,
                model=selected_config["model"]["name"],
                dataset=selected_config["dataset"]["name"], seed=seed,
            ))
        test_results.append(metrics)
        del trainer
        gc.collect()
    for split, reports in breakdowns.items():
        write_breakdown_summary(run_dir, split, reports)
    summary = {
        "status": "completed",
        "mode": mode,
        "run_dir": str(run_dir),
        "selection_metric": metric,
        "selected_trial": selected["index"],
        "num_candidates": len(candidates),
        "seeds": selected_config["training"]["seeds"],
        "validation": aggregate_metrics([result["validation"] for result in selected["seeds"]]),
        "test": aggregate_metrics(test_results),
    }
    write_json(
        run_dir / "experiment.json",
        {
            "mode": mode,
            "selection_metric": metric,
            "selected_trial": selected["index"],
            "trials": trials,
        },
    )
    write_json(run_dir / "environment.json", _environment(device))
    write_json(run_dir / "summary.json", summary)
    return summary
