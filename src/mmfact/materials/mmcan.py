"""Read MMCAN's shared frozen inputs without loading or running their backbones."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mmfact.data.cache_policy import PreparedInputError
from mmfact.data.features import fingerprint


def _path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise PreparedInputError("A shared MMCAN feature requires a relative file path.")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise PreparedInputError(f"Shared MMCAN input is missing or outside the dataset: {path}")
    return path


class _Matrix:
    def __init__(self, root, entry, kind, shape, *, mask=False):
        self.entry = entry
        self.values = np.load(_path(root, entry.get("values")), mmap_mode="r", allow_pickle=False)
        if (
            entry.get("input_kind") != kind
            or list(self.values.shape) != entry.get("shape")
            or tuple(self.values.shape[1:]) != shape
            or self.values.dtype != np.dtype("float32")
            or entry.get("dtype") != "float32"
        ):
            raise PreparedInputError(
                f"Shared MMCAN feature has an incompatible shape: {entry.get('name')}"
            )
        field = {"text": "text_id", "image": "image_id", "pair": "pair_id"}[kind]
        self.rows = {}
        for line in _path(root, entry.get("rows")).read_text("utf-8").splitlines():
            row = json.loads(line)
            key, index = row.get(field), row.get("row")
            if (
                not isinstance(key, str)
                or not key
                or key in self.rows
                or type(index) is not int
                or not 0 <= index < len(self.values)
            ):
                raise PreparedInputError("Invalid or duplicate shared MMCAN row index.")
            self.rows[key] = index
        if len(self.rows) != len(self.values) or set(self.rows.values()) != set(
            range(len(self.values))
        ):
            raise PreparedInputError("Shared MMCAN row index must cover every matrix row once.")
        self.mask = None
        if mask:
            self.mask = np.load(_path(root, entry.get("mask")), mmap_mode="r", allow_pickle=False)
            if self.mask.dtype != np.dtype("bool") or self.mask.shape != self.values.shape[:2]:
                raise PreparedInputError("Shared MMCAN BERT mask must be boolean [rows, tokens].")

    def read(self, keys, device, dtype):
        try:
            indices = [self.rows[key] for key in keys]
        except KeyError as error:
            raise PreparedInputError(
                f"Shared MMCAN inputs do not cover {error.args[0]!r}."
            ) from error
        values = torch.from_numpy(np.array(self.values[indices], copy=True)).to(
            device=device, dtype=dtype
        )
        if not torch.isfinite(values).all():
            raise PreparedInputError("Shared MMCAN features contain nonfinite values.")
        mask = None
        if self.mask is not None:
            mask = torch.from_numpy(np.array(self.mask[indices], copy=True)).to(device=device)
            if not mask.any(-1).all():
                raise PreparedInputError("Shared MMCAN BERT rows require at least one valid token.")
            values = values * mask.unsqueeze(-1).to(dtype)
        return values, mask


class MMCANSharedInputs(nn.Module):
    """Catalog-selected BERT tokens, MAE patches and binary ViLT logits."""

    def __init__(self, directory, text_options):
        super().__init__()
        self.register_buffer("_device", torch.empty(0), persistent=False)
        root = Path(directory).expanduser().resolve()
        entries = json.loads((root / "feature_catalog.json").read_text("utf-8"))["features"]

        def select(name):
            matches = [
                entry
                for entry in entries
                if {"method": "MMCAN", "input": name} in entry.get("consumers", [])
            ]
            if len(matches) != 1:
                raise PreparedInputError(f"MMCAN requires exactly one catalog input {name!r}.")
            return matches[0]

        text, image, matching = (
            select(name) for name in ("text_encoder", "image_encoder", "matching_encoder")
        )
        settings = text.get("settings", {})
        if (
            settings.get("model_name") != text_options["model_name"]
            or settings.get("max_length") != text_options["max_length"]
            or settings.get("mask_padding") is not True
            or settings.get("pooling") != "tokens"
        ):
            raise PreparedInputError(
                "MMCAN shared BERT language, length or padding recipe disagrees with configuration."
            )
        if image.get("settings", {}).get("model_name") != "facebook/vit-mae-base":
            raise PreparedInputError(
                "MMCAN shared image tokens require the public MAE-base backbone."
            )
        matching_settings = matching.get("settings", {})
        if (
            matching_settings.get("encoder") != "mmcan-vilt-pretrained-binary-itm"
            or matching_settings.get("output") != "logits"
            or matching_settings.get("normalize") is not False
            or matching_settings.get("max_length") != 40
        ):
            raise PreparedInputError("MMCAN matching inputs must be real binary ViLT ITM logits.")
        self.max_length, self.output_dim, self.sequence_length = (
            text_options["max_length"],
            768,
            196,
        )
        self.text = _Matrix(root, text, "text", (self.max_length, 768), mask=True)
        self.image = _Matrix(root, image, "image", (197, 768))
        self.matching = _Matrix(root, matching, "pair", (2,))
        self.identity = fingerprint(
            [
                {
                    "entry": {
                        key: matrix.entry[key]
                        for key in (
                            "values",
                            "rows",
                            "mask",
                            "shape",
                            "dtype",
                            "settings",
                            "input_kind",
                        )
                        if key in matrix.entry
                    },
                    "rows": matrix.rows,
                }
                for matrix in (self.text, self.image, self.matching)
            ]
        )

    def forward(self, batch):
        text_ids = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in batch["text"]]
        metadata = batch.get("metadata")
        if not isinstance(metadata, list) or len(metadata) != len(text_ids):
            raise PreparedInputError("Shared MMCAN batches require canonical image_id metadata.")
        image_ids, pair_ids = [], []
        for text_id, row in zip(text_ids, metadata, strict=True):
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or len(image_id) != 64:
                raise PreparedInputError("Shared MMCAN batches require an image SHA256 identity.")
            pair_id = hashlib.sha256(f"{text_id}:{image_id}".encode()).hexdigest()
            if row.get("text_id", text_id) != text_id or row.get("pair_id", pair_id) != pair_id:
                raise PreparedInputError("Shared MMCAN content identities disagree with the batch.")
            image_ids.append(image_id)
            pair_ids.append(pair_id)
        device, dtype = self._device.device, self._device.dtype
        text, mask = self.text.read(text_ids, device, dtype)
        image, _ = self.image.read(image_ids, device, dtype)
        matching, _ = self.matching.read(pair_ids, device, dtype)
        return text, image[:, 1:], mask, matching

    def get_extra_state(self):
        return {"format_version": 1, "catalog_identity": self.identity}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise PreparedInputError("MMCAN shared input catalog differs from the checkpoint.")
