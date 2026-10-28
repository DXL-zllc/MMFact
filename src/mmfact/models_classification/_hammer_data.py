"""HAMMER caption/token supervision, image geometry and native DGM4 loaders."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073))[:, None, None]
_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711))[:, None, None]
_TYPES = {
    "orig": (0, 0, 0, 0),
    "face_swap": (1, 0, 0, 0),
    "face_attribute": (0, 1, 0, 0),
    "text_swap": (0, 0, 1, 0),
    "text_attribute": (0, 0, 0, 1),
    "face_swap&text_swap": (1, 0, 1, 0),
    "face_swap&text_attribute": (1, 0, 0, 1),
    "face_attribute&text_swap": (0, 1, 1, 0),
    "face_attribute&text_attribute": (0, 1, 0, 1),
}


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _clean(text):
    if not isinstance(text, str):
        raise ValueError("HAMMER text must be a string.")
    text = re.sub(r"""[,.'!?"()*#:;~]""", "", text.lower())
    text = text.replace("-", " ").replace("/", " ").replace("<person>", "person")
    return re.sub(r"\s{2,}", " ", text).rstrip("\n").strip(" ")


def clean_caption(text: str, max_words: int = 50) -> str:
    """Apply the author's English recipe; Chinese characters remain intact."""
    _positive_int(max_words, "max_words")
    return " ".join(_clean(text).split(" ")[:max_words])


def _positions(values, caption):
    if (
        not isinstance(values, (list, tuple))
        or any(type(index) is not int or index < 0 for index in values)
        or len(set(values)) != len(values)
        or any(index >= len(caption.split(" ")) for index in values)
    ):
        raise ValueError("fake_text_pos must contain unique, valid cleaned-caption word indices.")
    return sorted(values)


def encode_text(
    tokenizer,
    texts: Sequence[str],
    *,
    max_words: int,
    max_length: int,
    fake_text_pos: Sequence[Sequence[int]] | None = None,
) -> dict[str, Tensor]:
    """Remove each SEP, right-pad and map manipulated words to all their pieces."""
    _positive_int(max_words, "max_words")
    _positive_int(max_length, "max_length")
    if (
        max_length < 3
        or not isinstance(texts, Sequence)
        or isinstance(texts, (str, bytes))
        or not texts
    ):
        raise ValueError("HAMMER needs nonempty texts and max_length >= 3.")
    captions = [clean_caption(text, max_words) for text in texts]
    if any(not caption.strip() for caption in captions):
        raise ValueError("HAMMER captions must contain at least one word.")
    positions = None
    if fake_text_pos is not None:
        if len(fake_text_pos) != len(texts):
            raise ValueError("fake_text_pos must match the caption batch.")
        positions = [
            _positions(indices, _clean(text)) for indices, text in zip(fake_text_pos, texts)
        ]
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("HAMMER requires a fast tokenizer for word-to-subword supervision.")
    if any(
        type(getattr(tokenizer, key, None)) is not int or getattr(tokenizer, key) < 0
        for key in ("cls_token_id", "sep_token_id", "pad_token_id")
    ):
        raise ValueError("HAMMER tokenizer requires valid CLS, SEP and PAD ids.")
    encoded = tokenizer(
        [caption.split(" ") for caption in captions],
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        is_split_into_words=True,
    )
    ids, targets = [], []
    for i, tokens in enumerate(encoded["input_ids"]):
        if (
            len(tokens) < 3
            or tokens[0] != tokenizer.cls_token_id
            or tokens[-1] != tokenizer.sep_token_id
        ):
            raise ValueError("HAMMER tokenizer must supply CLS, content and final SEP.")
        ids.append(tokens[:-1])
        if positions is not None:
            words = encoded.word_ids(i)[1:-1]
            targets.append(
                [int(word in positions[i]) if word is not None else -100 for word in words]
            )
            if not any(label != -100 for label in targets[-1]):
                raise ValueError("HAMMER needs at least one supervised content token.")
    length = max(map(len, ids))
    result = {
        "input_ids": torch.tensor(
            [row + [tokenizer.pad_token_id] * (length - len(row)) for row in ids], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1] * len(row) + [0] * (length - len(row)) for row in ids], dtype=torch.long
        ),
    }
    if positions is not None:
        result["token_labels"] = torch.tensor(
            [row + [-100] * (length - 1 - len(row)) for row in targets], dtype=torch.long
        )
    return result


def _box(values, width, height):
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 4
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
    ):
        raise ValueError("DGM4 boxes must have four finite numeric xyxy coordinates.")
    x0, y0, x1, y1 = values
    if not any(values):
        return [0.0] * 4
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("DGM4 boxes must be nondegenerate and within the original image.")
    if int(x1 - x0) < 1 or int(y1 - y0) < 1:
        raise ValueError(
            "DGM4 nonzero boxes must retain positive size after source integerization."
        )
    return list(values)


def _photometric(image):
    # Two draws with replacement, each applied with p=.5, magnitude 7/10.
    for operation in np.random.choice(5, size=2):
        if np.random.random() > 0.5:
            continue
        if operation == 1:
            image = ImageOps.autocontrast(image)
        elif operation == 2:
            image = ImageOps.equalize(image)
        elif operation in (3, 4):
            pixels = np.asarray(image)
            factor = 0.1 + 1.8 * 0.7
            if operation == 3:
                pixels = np.clip(pixels.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            elif min(pixels.shape[:2]) >= 3:
                # Integer 3x3 smoothing reproduces the uint8 kernel rounding;
                # the outer border remains unchanged, as in the author's op.
                source = pixels.astype(np.int32)
                smooth = 4 * source[1:-1, 1:-1]
                for dy in range(3):
                    for dx in range(3):
                        smooth = (
                            smooth
                            + source[dy : dy + pixels.shape[0] - 2, dx : dx + pixels.shape[1] - 2]
                        )
                smooth = ((smooth + 6) // 13).astype(np.float32)
                pixels = pixels.copy()
                pixels[1:-1, 1:-1] = np.clip(
                    smooth + factor * (source[1:-1, 1:-1].astype(np.float32) - smooth), 0, 255
                ).astype(np.uint8)
            image = Image.fromarray(pixels)
    return image


def image_inputs(
    paths: Sequence[str | Path], *, image_size: int, training: bool, boxes: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """Square-resize RGB images and synchronously flip original-pixel boxes."""
    _positive_int(image_size, "image_size")
    if type(training) is not bool or isinstance(paths, (str, bytes)) or not paths:
        raise ValueError("HAMMER needs nonempty image paths and a boolean training flag.")
    if boxes is not None and (
        not isinstance(boxes, Tensor)
        or boxes.shape != (len(paths), 4)
        or boxes.dtype == torch.bool
        or not torch.isfinite(boxes).all()
    ):
        raise ValueError("HAMMER boxes must be a finite [B,4] tensor.")
    pixels, normalized = [], []
    for i, path in enumerate(paths):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        flip = training and random.random() < 0.5
        if boxes is not None:
            xyxy = _box(boxes[i].detach().cpu().tolist(), width, height)
            if any(xyxy):
                # Source truncates x,y,w,h separately, before geometric transforms.
                x, y, right, bottom = xyxy
                x, y, w, h = int(x), int(y), int(right - x), int(bottom - y)
                if flip:
                    x = width - x - w
                normalized.append(
                    [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
                )
            else:
                normalized.append([0.0] * 4)
        if flip:
            image = ImageOps.mirror(image)
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        if training:
            image = _photometric(image)
        value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        pixels.append((value - _MEAN) / _STD)
    return torch.stack(pixels), torch.tensor(
        normalized, dtype=torch.float32
    ) if boxes is not None else None


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate DGM4 JSON key: {key}.")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite DGM4 JSON constant: {value}.")


def _read_records(path):
    payload = path.read_bytes()
    text = payload.decode("utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    else:
        rows = [
            json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
            for line in text.splitlines()
        ]
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("DGM4 manifests must be nonempty JSON arrays or JSONL objects.")
    return rows, hashlib.sha256(payload).hexdigest()


class _Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return deepcopy(self.rows[index])


def _collate(rows):
    return {
        "sample_id": [row["sample_id"] for row in rows],
        "text": [row["text"] for row in rows],
        "image_path": [row["image_path"] for row in rows],
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        "box_xyxy": torch.tensor([row["box_xyxy"] for row in rows], dtype=torch.float64),
        "fake_text_pos": [row["fake_text_pos"] for row in rows],
        "manipulation_labels": torch.tensor(
            [row["manipulation_labels"] for row in rows], dtype=torch.float32
        ),
    }


class DGM4Data(nn.Module):
    """Validate explicit auxiliary train/val resources; never train in prepare."""

    def __init__(self, options: Mapping, seed: int):
        super().__init__()
        if not isinstance(options, Mapping) or set(options) != {"root", "train", "val"}:
            raise ValueError("DGM4 data requires exactly root, train and val options.")
        if type(seed) is not int or seed < 0:
            raise ValueError("DGM4 seed must be a nonnegative integer.")
        self.options = deepcopy(dict(options))
        for key, value in self.options.items():
            values = value if key != "root" and isinstance(value, list) else [value]
            if not values or any(not isinstance(path, str) or not path.strip() for path in values):
                raise ValueError(f"DGM4 {key} requires a nonempty path or list of paths.")
        self.seed = seed
        self._rows = {}
        self._provenance = {}

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def prepare(self):
        self._rows, self._provenance = {}, {}
        root = Path(self.options["root"]).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"DGM4 root is not a directory: {root}.")
        datasets, resources, raw_ids, images, inputs, seen_manifests = {}, {}, {}, {}, {}, set()
        for split in ("train", "val"):
            paths = self.options[split]
            paths = paths if isinstance(paths, list) else [paths]
            records, manifest_hashes, news_ids, id_count = {}, [], set(), 0
            occurrences = {}
            inputs[split] = set()
            missing_boxes = missing_positions = 0
            for path in paths:
                manifest = Path(path).expanduser()
                manifest = (manifest if manifest.is_absolute() else root / manifest).resolve()
                if manifest in seen_manifests:
                    raise ValueError("DGM4 train/val manifests must be distinct.")
                seen_manifests.add(manifest)
                manifest_rows, manifest_sha = _read_records(manifest)
                manifest_hashes.append(manifest_sha)
                for row in manifest_rows:
                    required = {"image", "text", "fake_cls", "fake_text_pos"}
                    if (
                        not required <= row.keys()
                        or not isinstance(row["fake_cls"], str)
                        or row["fake_cls"] not in _TYPES
                    ):
                        raise ValueError(
                            "DGM4 records need native fields and a legal manipulation class."
                        )
                    if "id" in row:
                        news_id = row["id"]
                        if type(news_id) not in (int, str) or not str(news_id).strip():
                            raise ValueError("DGM4 original news id must be a string or integer.")
                        news_ids.add(str(news_id))
                        id_count += 1
                    relative = row["image"]
                    if (
                        not isinstance(relative, str)
                        or not relative.strip()
                        or Path(relative).is_absolute()
                        or ".." in Path(relative).parts
                    ):
                        raise ValueError("DGM4 image must be a safe relative path.")
                    image = (root / relative).resolve()
                    if not image.is_relative_to(root) or not image.is_file():
                        raise ValueError("DGM4 image must exist within its explicit root.")
                    if image not in images:
                        payload = image.read_bytes()
                        with Image.open(BytesIO(payload)) as opened:
                            opened.convert("RGB").load()
                            width, height = opened.size
                        images[image] = (hashlib.sha256(payload).hexdigest(), width, height)
                    image_sha, width, height = images[image]
                    caption = _clean(row["text"])
                    if not caption.strip():
                        raise ValueError("DGM4 captions must contain at least one word.")
                    positions = _positions(row["fake_text_pos"], caption)
                    inputs[split].add(_digest([image_sha, caption]))
                    box = row.get("fake_image_box", [])
                    box = _box([0] * 4 if box == [] else box, width, height)
                    manipulation = _TYPES[row["fake_cls"]]
                    if (not any(manipulation[:2]) and any(box)) or (
                        not any(manipulation[2:]) and positions
                    ):
                        raise ValueError(
                            "DGM4 localization annotations contradict manipulation type."
                        )
                    missing_boxes += int(any(manipulation[:2]) and not any(box))
                    missing_positions += int(any(manipulation[2:]) and not positions)
                    identity = {
                        "text": row["text"],
                        "image_sha256": image_sha,
                        "fake_cls": row["fake_cls"],
                        "box_xyxy": box,
                        "fake_text_pos": positions,
                    }
                    content_id = "dgm4:" + _digest(identity)
                    occurrence = occurrences.get(content_id, 0)
                    occurrences[content_id] = occurrence + 1
                    # The released native training manifest repeats pristine
                    # records. Preserve that sampling weight instead of rejecting
                    # or silently deduplicating the author's training examples.
                    sample_id = content_id if not occurrence else f"{content_id}:repeat:{occurrence}"
                    records[sample_id] = {
                        "sample_id": sample_id,
                        "text": row["text"],
                        "image_path": str(image),
                        "label": int(row["fake_cls"] != "orig"),
                        "box_xyxy": box,
                        "fake_text_pos": positions,
                        "manipulation_labels": manipulation,
                    }
            datasets[split] = [records[key] for key in sorted(records)]
            raw_ids[split] = news_ids
            resources[split] = {
                "count": len(records),
                "manifest_sha256": manifest_hashes,
                "records_sha256": _digest(sorted(records)),
                "raw_id_count": len(news_ids),
                "raw_id_available_count": id_count,
                "raw_id_missing_count": len(records) - id_count,
                "image_manipulation_without_box": missing_boxes,
                "text_manipulation_without_positions": missing_positions,
                "unique_content_records": len(occurrences),
                "repeated_record_occurrences": len(records) - len(occurrences),
            }
        if {row["sample_id"] for row in datasets["train"]} & {
            row["sample_id"] for row in datasets["val"]
        }:
            raise ValueError("DGM4 train and val overlap in complete content/annotations.")
        if inputs["train"] & inputs["val"]:
            raise ValueError("DGM4 train and val overlap in image/caption inputs.")
        overlap = sorted(raw_ids["train"] & raw_ids["val"])
        provenance = {
            "version": 2,
            "splits": resources,
            "image_count": len(images),
            "raw_id_overlap": {"count": len(overlap), "sha256": _digest(overlap)},
        }
        provenance["resource_sha256"] = _digest(provenance)
        self._rows, self._provenance = datasets, provenance

    def loader(self, split: str, original_loader: DataLoader, options: Mapping):
        if split not in ("train", "val"):
            raise ValueError("DGM4 only exposes auxiliary train/val loaders.")
        if split not in self._rows:
            raise RuntimeError("Prepare DGM4 auxiliary resources before constructing loaders.")
        if not isinstance(original_loader, DataLoader) or not isinstance(options, Mapping):
            raise ValueError("DGM4 loader requires the original DataLoader and stage options.")
        key = "batch_size" if split == "train" else "eval_batch_size"
        batch_size = options.get(key, original_loader.batch_size)
        _positive_int(batch_size, key)
        seed = original_loader.generator.initial_seed() if original_loader.generator else self.seed
        extras = {}
        if original_loader.num_workers:
            extras.update(
                persistent_workers=original_loader.persistent_workers,
                prefetch_factor=original_loader.prefetch_factor,
                multiprocessing_context=original_loader.multiprocessing_context,
            )
        return DataLoader(
            _Rows(self._rows[split]),
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=False,
            collate_fn=_collate,
            num_workers=original_loader.num_workers,
            pin_memory=original_loader.pin_memory,
            worker_init_fn=original_loader.worker_init_fn,
            generator=torch.Generator().manual_seed(seed),
            **extras,
        )

    def get_extra_state(self):
        return deepcopy({"version": 1, "options": self.options, "provenance": self._provenance})

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "options", "provenance"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or state["options"] != self.options
            or not isinstance(state["provenance"], dict)
        ):
            raise ValueError("DGM4 auxiliary resource state is incompatible.")
        provenance = deepcopy(state["provenance"])
        if provenance:
            expected = provenance.pop("resource_sha256", None)
            split_keys = {
                "count",
                "manifest_sha256",
                "records_sha256",
                "raw_id_count",
                "raw_id_available_count",
                "raw_id_missing_count",
                "image_manipulation_without_box",
                "text_manipulation_without_positions",
            }
            digest_keys = ("records_sha256",)

            def valid_hash(value):
                return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)

            if (
                set(provenance) != {"version", "splits", "image_count", "raw_id_overlap"}
                or type(provenance["version"]) is not int
                or provenance["version"] not in (1, 2)
                or not isinstance(provenance["splits"], dict)
                or set(provenance["splits"]) != {"train", "val"}
                or type(provenance["image_count"]) is not int
                or provenance["image_count"] < 1
            ):
                raise ValueError("DGM4 auxiliary resource provenance is invalid.")
            if provenance["version"] == 2:
                split_keys |= {"unique_content_records", "repeated_record_occurrences"}
            for resource in provenance["splits"].values():
                if (
                    not isinstance(resource, dict)
                    or set(resource) != split_keys
                    or any(not valid_hash(resource[key]) for key in digest_keys)
                    or not isinstance(resource["manifest_sha256"], list)
                    or not resource["manifest_sha256"]
                    or any(not valid_hash(value) for value in resource["manifest_sha256"])
                    or any(
                        type(resource[key]) is not int or resource[key] < 0
                        for key in split_keys - {"manifest_sha256", "records_sha256"}
                    )
                    or resource["count"] < 1
                    or not 0
                    <= resource["raw_id_count"]
                    <= resource["raw_id_available_count"]
                    <= resource["count"]
                    or resource["raw_id_missing_count"] + resource["raw_id_available_count"]
                    != resource["count"]
                    or resource["image_manipulation_without_box"] > resource["count"]
                    or resource["text_manipulation_without_positions"] > resource["count"]
                    or (
                        provenance["version"] == 2
                        and (
                            resource["unique_content_records"] < 1
                            or resource["unique_content_records"]
                            + resource["repeated_record_occurrences"] != resource["count"]
                        )
                    )
                ):
                    raise ValueError("DGM4 auxiliary split provenance is invalid.")
            overlap = provenance["raw_id_overlap"]
            if (
                not isinstance(overlap, dict)
                or set(overlap) != {"count", "sha256"}
                or type(overlap["count"]) is not int
                or overlap["count"] < 0
                or not valid_hash(overlap["sha256"])
                or overlap["count"]
                > min(value["raw_id_count"] for value in provenance["splits"].values())
                or provenance["image_count"]
                > sum(value["count"] for value in provenance["splits"].values())
                or expected != _digest(provenance)
            ):
                raise ValueError("DGM4 auxiliary resource fingerprint is invalid.")
            provenance["resource_sha256"] = expected
        if provenance != self._provenance:
            self._rows = {}
        self._provenance = provenance
