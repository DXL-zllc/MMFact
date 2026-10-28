"""Load canonical JSONL manifests without extracting model features."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def _reject_nonfinite(value: str):
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _binary_label(value: Any, label_map: Mapping[Any, int] | None) -> int:
    # bool is an int subclass, but is not a valid explicit class label.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("label must be an integer or a string")
    if label_map is not None:
        key = value
        if key not in label_map and isinstance(key, int):
            key = str(key)
        if key not in label_map:
            raise ValueError(f"label {value!r} is not present in label_map")
        value = label_map[key]
    if type(value) is not int or value not in (0, 1):
        raise ValueError("label must map to integer 0 (real) or 1 (fake)")
    return value


def _group_key(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("group_id")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("metadata.group_id must be a nonempty string or integer")
    if isinstance(value, str) and not value.strip():
        raise ValueError("metadata.group_id must be a nonempty string or integer")
    # CSV adapters may represent the same identifier as either 7 or "7".
    return str(value)


class NewsDataset(Dataset):
    """A nonempty manifest of canonical image-text examples.

    ``manifest_path`` and every input ``image_path`` are relative to ``root``.
    A manifest may also be passed as an absolute path. Returned image paths are
    absolute, and metadata always includes the original ``raw_label``. Unknown
    top-level fields are retained in metadata, with conflicting keys rejected.
    Empty or invalid rows raise errors; no examples are silently removed.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        *,
        label_map: Mapping[Any, int] | None = None,
        verify_images: bool = False,
        require_images: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"dataset root is not a directory: {self.root}")
        manifest = Path(manifest_path).expanduser()
        self.manifest_path = (
            manifest if manifest.is_absolute() else self.root / manifest
        ).resolve()
        if label_map is not None:
            if not isinstance(label_map, Mapping):
                raise ValueError("label_map must be a mapping")
            for key, value in label_map.items():
                if isinstance(key, bool) or not isinstance(key, (int, str)):
                    raise ValueError("label_map keys must be integer or string labels")
                if type(value) is not int or value not in (0, 1):
                    raise ValueError("label_map values must be integer 0 or 1")
        if not isinstance(verify_images, bool):
            raise ValueError("verify_images must be a boolean")
        if not isinstance(require_images, bool):
            raise ValueError("require_images must be a boolean")
        if verify_images and not require_images:
            raise ValueError("verify_images requires require_images=true")

        self.records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        image_index = None
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                location = f"{self.manifest_path}:{line_number}"
                try:
                    if not line.strip():
                        raise ValueError("empty JSONL row")
                    record = json.loads(line, parse_constant=_reject_nonfinite)
                    if not isinstance(record, dict):
                        raise ValueError("each JSONL row must be an object")
                    if "sample_id" not in record and "id" in record:
                        record["sample_id"] = record.pop("id")
                    if "image_path" not in record and "image_id" in record:
                        if image_index is None:
                            image_index = {}
                            with (self.root / "images/index.jsonl").open(encoding="utf-8") as index:
                                for entry in index:
                                    image = json.loads(entry, parse_constant=_reject_nonfinite)
                                    key = image["image_id"]
                                    if key in image_index:
                                        raise ValueError("duplicate image_id in images/index.jsonl")
                                    image_index[key] = image["file"]
                        if record["image_id"] not in image_index:
                            raise ValueError("image_id is missing from images/index.jsonl")
                        record["image_path"] = image_index[record["image_id"]]
                    for field in ("sample_id", "text", "image_path", "label"):
                        if field not in record:
                            raise ValueError(f"missing required field {field!r}")
                    sample_id = record["sample_id"]
                    if not isinstance(sample_id, str) or not sample_id.strip():
                        raise ValueError("sample_id must be a nonempty string")
                    if sample_id in seen_ids:
                        raise ValueError(f"duplicate sample_id {sample_id!r}")
                    text = record["text"]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("text must be a nonempty string")

                    image_value = record["image_path"]
                    if not isinstance(image_value, str) or not image_value.strip():
                        raise ValueError("image_path must be a nonempty relative path")
                    image_relative = Path(image_value)
                    if image_relative.is_absolute():
                        raise ValueError("image_path must be relative to dataset root")
                    image_path = (self.root / image_relative).resolve()
                    if not image_path.is_relative_to(self.root):
                        raise ValueError("image_path must stay within dataset root")
                    if require_images and not image_path.is_file():
                        raise ValueError(f"image file does not exist: {image_path}")
                    if verify_images:
                        try:
                            with Image.open(image_path) as image:
                                image.load()
                        except (OSError, ValueError, SyntaxError) as exc:
                            raise ValueError(f"cannot decode image: {image_path}") from exc

                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata must be an object")
                    metadata = dict(metadata)
                    for key, value in record.items():
                        if key in {"sample_id", "text", "image_path", "label", "metadata"}:
                            continue
                        if key in metadata and metadata[key] != value:
                            raise ValueError(f"conflicting metadata key {key!r}")
                        metadata[key] = value
                    _group_key(metadata)
                    metadata["raw_label"] = record["label"]
                    label = _binary_label(record["label"], label_map)
                    self.records.append(
                        {
                            "sample_id": sample_id,
                            "text": text,
                            "image_path": str(image_path),
                            "label": label,
                            "metadata": metadata,
                        }
                    )
                    seen_ids.add(sample_id)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{location}: {exc}") from exc
        if not self.records:
            raise ValueError(f"manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def load_datasets(
    dataset_config: Mapping[str, Any], *, splits: tuple[str, ...] = ("train", "val", "test")
) -> dict[str, NewsDataset]:
    """Load selected splits, checking overlaps among the loaded manifests."""

    if not isinstance(dataset_config, Mapping):
        raise ValueError("dataset configuration must be a mapping")
    root = dataset_config.get("root")
    if not isinstance(root, (str, Path)) or not str(root).strip():
        raise ValueError("dataset.root must be a nonempty path")
    manifests = dataset_config.get("splits")
    if not isinstance(manifests, Mapping) or set(manifests) != {"train", "val", "test"}:
        raise ValueError("dataset.splits must contain exactly train, val, and test")
    if (
        not isinstance(splits, tuple)
        or not splits
        or any(not isinstance(split, str) or split not in manifests for split in splits)
        or len(set(splits)) != len(splits)
    ):
        raise ValueError("Selected splits must be a nonempty tuple of unique train/val/test names")

    datasets: dict[str, NewsDataset] = {}
    sample_owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    for split in splits:
        manifest = manifests[split]
        if not isinstance(manifest, (str, Path)) or not str(manifest).strip():
            raise ValueError(f"dataset.splits.{split} must be a nonempty path")
        dataset = NewsDataset(
            manifest_path=manifest,
            root=root,
            label_map=dataset_config.get("label_map"),
            verify_images=dataset_config.get("verify_images", False),
            require_images=dataset_config.get("require_images", True),
        )
        for record in dataset.records:
            sample_id = record["sample_id"]
            owner = sample_owners.get(sample_id)
            if owner is not None and owner != split:
                raise ValueError(f"sample_id {sample_id!r} overlaps splits {owner!r} and {split!r}")
            sample_owners[sample_id] = split
            group_id = _group_key(record["metadata"])
            if group_id is not None:
                owner = group_owners.get(group_id)
                if owner is not None and owner != split:
                    raise ValueError(
                        f"metadata.group_id {group_id!r} overlaps splits {owner!r} and {split!r}"
                    )
                group_owners[group_id] = split
        datasets[split] = dataset
    return datasets
