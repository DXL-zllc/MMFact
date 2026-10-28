"""Read the shared Hub layout lazily, without running material producers."""

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mmfact.data.cache_policy import PreparedInputError
from mmfact.data.features import fingerprint

_KEYS = {
    "text": ("text_id",),
    "image": ("image_id",),
    "pair": ("pair_id",),
    "caption": ("image_ids",),
    "arguments": ("pair_ids",),
    "description": ("description_id",),
    "pair_text": ("clip_text_id", "pair_ids"),
}


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _path(root, relative):
    if not isinstance(relative, str) or not relative.strip():
        raise PreparedInputError("A material requires a nonempty relative path.")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise PreparedInputError(f"Material path escapes the dataset: {relative}")
    if not path.is_file():
        raise PreparedInputError(f"Prepared material is missing: {path}")
    return path


def _json(path):
    try:
        return json.loads(path.read_text("utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        raise PreparedInputError(f"Invalid material JSON: {path}") from error


def _jsonl(path):
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL rows must be objects.")
                except (TypeError, ValueError) as error:
                    raise PreparedInputError(f"Invalid material row at {path}:{number}") from error
                yield row
    except OSError as error:
        raise PreparedInputError(f"Cannot read prepared material: {path}") from error


def _array(path):
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(values, np.ndarray) or values.dtype.kind not in "biuf":
            raise ValueError("A material matrix must have a numeric or boolean dtype.")
        return values
    except (OSError, TypeError, ValueError, EOFError) as error:
        raise PreparedInputError(f"Invalid prepared matrix: {path}") from error


def _select(entries, *, name=None, method=None, role=None):
    matches = [
        entry
        for entry in entries
        if (name is None or entry.get("name") == name)
        and (
            method is None
            or any(
                consumer.get("method") == method and consumer.get("input") == role
                for consumer in entry.get("consumers", [])
            )
        )
    ]
    if len(matches) != 1:
        target = repr(name) if name is not None else f"{method}.{role}"
        raise PreparedInputError(f"Expected exactly one prepared catalog entry for {target}.")
    return matches[0]


class FeatureMatrix:
    """A catalog matrix whose rows may have multiple content-ID aliases.

    Matrix headers and small row indices are checked on first access. Numeric
    finiteness is checked only for requested rows; opening a large array never
    scans or hashes its payload. Masks are returned unchanged, with true=valid.
    """

    def __init__(self, root, entry):
        self.root = Path(root).expanduser().resolve()
        self.entry = deepcopy(entry)
        self.name = entry.get("name")
        self.input_kind = entry.get("input_kind")
        self.settings = deepcopy(entry.get("settings", {}))
        shape = entry.get("shape")
        if (
            not isinstance(self.name, str)
            or self.input_kind not in _KEYS
            or not isinstance(self.settings, dict)
            or not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise PreparedInputError("Invalid shared feature catalog entry.")
        self.shape = tuple(shape)
        self._values, self._mask, self._rows, self._records = None, None, None, None

    def require_settings(self, expected):
        """Check the settings relevant to a consumer without inventing defaults."""
        for key, value in expected.items():
            if self.settings.get(key) != value:
                raise PreparedInputError(f"Prepared feature {self.name!r} disagrees on {key!r}.")
        return self

    @property
    def values(self):
        if self._values is None:
            values = _array(_path(self.root, self.entry.get("values")))
            if values.shape != self.shape or str(values.dtype) != self.entry.get("dtype"):
                raise PreparedInputError(f"Prepared feature shape/dtype disagrees: {self.name}")
            self._values = values
        return self._values

    @property
    def mask(self):
        if self.entry.get("mask") and self._mask is None:
            mask = _array(_path(self.root, self.entry["mask"]))
            shape = self.values.shape
            valid_shape = (
                mask.ndim >= 2
                and mask.shape[0] == shape[0]
                and (
                    mask.shape[1:] == shape[1 : mask.ndim]
                    or mask.shape[1:] == shape[-mask.ndim : -1]
                )
            )
            if mask.dtype != np.dtype("bool") or not valid_shape:
                raise PreparedInputError(f"Prepared feature has an incompatible mask: {self.name}")
            self._mask = mask
        return self._mask

    @property
    def rows(self):
        if self._rows is None:
            if self.entry.get("keys"):
                keys = _json(_path(self.root, self.entry["keys"]))
                if not isinstance(keys, list):
                    raise PreparedInputError(f"Prepared matrix keys must be a list: {self.name}")
                field = self.entry.get("id_field", _KEYS[self.input_kind][0])
                records = [{"row": index, field: key} for index, key in enumerate(keys)]
            else:
                records = list(_jsonl(_path(self.root, self.entry.get("rows"))))
            aliases, covered = {}, set()
            for record in records:
                index = record.get("row")
                if type(index) is not int or not 0 <= index < self.shape[0] or index in covered:
                    raise PreparedInputError(f"Invalid matrix row index: {self.name}")
                keys = []
                for field in _KEYS[self.input_kind]:
                    value = record.get(field, [])
                    keys.extend(value if isinstance(value, list) else [value])
                if not keys or any(not isinstance(key, str) or not key for key in keys):
                    raise PreparedInputError(
                        f"Missing content identity in matrix rows: {self.name}"
                    )
                if len(set(keys)) != len(keys) or any(key in aliases for key in keys):
                    raise PreparedInputError(
                        f"Duplicate content identity in matrix rows: {self.name}"
                    )
                aliases.update((key, index) for key in keys)
                covered.add(index)
            if covered != set(range(self.shape[0])):
                raise PreparedInputError(f"Matrix rows do not cover every stored row: {self.name}")
            self._rows, self._records = aliases, records
        return self._rows

    @property
    def row_records(self):
        self.rows
        return self._records

    @property
    def identity(self):
        return fingerprint({"entry": self.entry, "rows": self.row_records})

    def read(self, keys, *, device=None, dtype=None):
        indices = self._indices(keys)
        values = torch.from_numpy(np.array(self.values[indices], copy=True))
        if values.is_floating_point() and not torch.isfinite(values).all():
            raise PreparedInputError(f"Prepared feature contains nonfinite values: {self.name}")
        return values.to(device=device, dtype=dtype), self._read_mask(indices, device)

    def _indices(self, keys):
        try:
            return [self.rows[key] for key in keys]
        except (KeyError, TypeError) as error:
            raise PreparedInputError(
                f"Prepared feature {self.name!r} does not cover {error.args[0]!r}."
            ) from error

    def _read_mask(self, indices, device):
        mask = self.mask
        if mask is not None:
            mask = torch.from_numpy(np.array(mask[indices], copy=True)).to(device=device)
        return mask

    def read_mask(self, keys, *, device=None):
        return self._read_mask(self._indices(keys), device)

    def read_batch(self, batch, *, device=None, dtype=None):
        return self.read(batch_keys(batch, self.input_kind), device=device, dtype=dtype)


def batch_keys(batch, kind):
    """Resolve canonical IDs from batch metadata without opening original images."""
    metadata = batch.get("metadata")
    texts = batch.get("text", [])
    if not isinstance(metadata, list) or len(metadata) != len(texts):
        raise PreparedInputError("Prepared batches require one metadata record per text.")
    field = {
        "text": "text_id",
        "image": "image_id",
        "pair": "pair_id",
        "caption": "image_id",
        "arguments": "pair_id",
        "pair_text": "pair_id",
        "description": "description_id",
    }.get(kind)
    if field is None:
        raise PreparedInputError(f"Unsupported prepared input kind: {kind}")
    keys = []
    for text, row in zip(texts, metadata, strict=True):
        if not isinstance(row, Mapping):
            raise PreparedInputError("Prepared metadata records must be mappings.")
        text_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if row.get("text_id", text_id) != text_id:
            raise PreparedInputError("Prepared text identity disagrees with batch content.")
        image_id = row.get("image_id")
        pair_id = None
        if isinstance(image_id, str) and image_id:
            pair_id = hashlib.sha256(f"{text_id}:{image_id}".encode()).hexdigest()
            if row.get("pair_id", pair_id) != pair_id:
                raise PreparedInputError("Prepared pair identity disagrees with batch content.")
        key = {"text_id": text_id, "pair_id": pair_id}.get(field, row.get(field))
        if not isinstance(key, str) or not key:
            raise PreparedInputError(f"Prepared batch is missing {field}.")
        keys.append(key)
    return keys


class JsonRecords(Mapping):
    """An on-demand JSONL index for annotations and method-specific resources."""

    def __init__(self, path, key=None):
        self.path, self.key, self._records = Path(path), key, None

    @property
    def records(self):
        if self._records is None:
            result = {}
            for row in _jsonl(self.path):
                if self.key is None:
                    self.key = next(
                        (
                            key
                            for key in ("pair_id", "image_id", "text_id", "description_id", "id")
                            if key in row
                        ),
                        None,
                    )
                key = row.get(self.key)
                if not isinstance(key, str) or not key or key in result:
                    raise PreparedInputError(f"Invalid or duplicate {self.key!r} in {self.path}")
                result[key] = row
            self._records = result
        return self._records

    def __getitem__(self, key):
        try:
            return self.records[key]
        except KeyError as error:
            raise PreparedInputError(
                f"Prepared records {self.path} do not cover {key!r}."
            ) from error

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def read(self, keys):
        return [self[key] for key in keys]


class TrainingResource:
    """Named catalog files; arrays stay memory mapped until their rows are used."""

    def __init__(self, root, entry):
        self.root, self.entry = Path(root), deepcopy(entry)
        self.files = deepcopy(entry.get("files", {}))
        if not isinstance(self.files, dict):
            raise PreparedInputError("Training resource files must be a mapping.")
        self._arrays = {}

    def path(self, name):
        if name not in self.files:
            raise PreparedInputError(f"Training resource has no file {name!r}.")
        return _path(self.root, self.files[name])

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = _array(self.path(name))
        return self._arrays[name]

    def json(self, name):
        return _json(self.path(name))

    def jsonl(self, name):
        return list(_jsonl(self.path(name)))

    @property
    def settings(self):
        directory = self.entry.get("directory")
        return _json(_path(self.root, f"{directory}/settings.json"))


class HubMaterials:
    """Select shared features by the method/branch consumer catalog."""

    def __init__(self, root, dataset=None):
        root = Path(root).expanduser().resolve()
        self.root = (root / "datasets" / dataset).resolve() if dataset is not None else root
        if not self.root.is_relative_to(root):
            raise PreparedInputError("Dataset name must stay within the material root.")
        catalog = _json(_path(self.root, "feature_catalog.json"))
        self.entries = catalog.get("features") if isinstance(catalog, dict) else None
        if not isinstance(self.entries, list) or not all(isinstance(e, dict) for e in self.entries):
            raise PreparedInputError("Invalid shared feature catalog.")
        names = [entry.get("name") for entry in self.entries]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
            names
        ):
            raise PreparedInputError("Shared feature names must be nonempty and unique.")
        self._features, self._annotations, self._training = {}, {}, {}
        self._training_entries = None

    def feature(self, name):
        if name not in self._features:
            self._features[name] = FeatureMatrix(self.root, _select(self.entries, name=name))
        return self._features[name]

    def feature_for(self, method, role):
        entry = _select(self.entries, method=method, role=role)
        consumer = next(
            consumer
            for consumer in entry["consumers"]
            if consumer.get("method") == method and consumer.get("input") == role
        )
        overrides = consumer.get("settings_overrides")
        if not overrides:
            return self.feature(entry["name"])
        if not isinstance(overrides, dict):
            raise PreparedInputError("Consumer setting overrides must be a mapping.")
        key = (entry["name"], method, role)
        if key not in self._features:
            entry = deepcopy(entry)
            entry["settings"].update(deepcopy(overrides))
            self._features[key] = FeatureMatrix(self.root, entry)
        return self._features[key]

    def features_for(self, method):
        roles = [
            consumer["input"]
            for entry in self.entries
            for consumer in entry.get("consumers", [])
            if consumer.get("method") == method
        ]
        return {role: _select(self.entries, method=method, role=role) for role in roles}

    def annotations(self, name, *, key=None):
        identity = (name, key)
        if identity not in self._annotations:
            relative = name if name.startswith("annotations/") else f"annotations/{name}"
            if not relative.endswith(".jsonl"):
                relative += ".jsonl"
            self._annotations[identity] = JsonRecords(_path(self.root, relative), key)
        return self._annotations[identity]

    def records(self, relative, *, key=None):
        return JsonRecords(_path(self.root, relative), key)

    @property
    def annotation_schema(self):
        return _json(_path(self.root, "annotations/schema.json"))

    def training(self, name):
        entry = _select(self.training_catalog, name=name)
        if name not in self._training:
            self._training[name] = TrainingResource(self.root, entry)
        return self._training[name]

    def training_for(self, method, role):
        if self._training_entries is None:
            self.training_catalog
        entry = _select(self._training_entries, method=method, role=role)
        return self.training(entry["name"])

    @property
    def training_catalog(self):
        if self._training_entries is None:
            catalog = _json(_path(self.root, "training/catalog.json"))
            entries = catalog.get("resources")
            if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                raise PreparedInputError("Invalid training resource catalog.")
            self._training_entries = entries
        return self._training_entries


class HubEncoder(nn.Module):
    """A frozen encoder interface backed by existing matrix rows only."""

    freeze, cache_features, variable_inputs = True, False, False
    packages = ()

    def __init__(self, matrix, *, tokenizer=None, bindings=None):
        super().__init__()
        self.matrix, self.preprocessing = matrix, None
        self.input_kind, self.encoding_spec = matrix.input_kind, deepcopy(matrix.settings)
        self.model_name = matrix.settings.get("model_name")
        self.revision = matrix.settings.get("revision", "main")
        self.register_buffer("_device", torch.empty(0), persistent=False)
        self._tokenizer, self._bindings = tokenizer, {}
        shape = matrix.shape[1:]
        spatial = matrix.input_kind == "image" and len(shape) == 3
        self.output_dim = shape[0] if spatial else shape[-1]
        self.max_length = matrix.settings.get("max_length", shape[0] if len(shape) > 1 else 1)
        self.sequence_length = shape[0] if len(shape) > 1 and not spatial else 1
        if bindings:
            self.bind(list(bindings), list(bindings.values()))
        if matrix.input_kind == "description":
            for row in matrix.row_records:
                if isinstance(row.get("text"), str):
                    self.bind([row["text"]], [row["description_id"]])

    @property
    def device(self):
        return self._device.device

    def bind(self, inputs, keys):
        for value, key in zip(inputs, keys, strict=True):
            if (
                not isinstance(value, str)
                or not isinstance(key, str)
                or key not in self.matrix.rows
            ):
                raise PreparedInputError("Encoder bindings require existing string content IDs.")
            if value in self._bindings and self._bindings[value] != key:
                if self.matrix.rows[self._bindings[value]] != self.matrix.rows[key]:
                    raise PreparedInputError(
                        "The same encoder input maps to different stored rows."
                    )
            self._bindings[value] = key
        return self

    def _keys(self, inputs):
        keys = []
        for value in inputs:
            if not isinstance(value, str):
                raise PreparedInputError("Prepared encoder inputs must be strings.")
            if value in self._bindings:
                key = self._bindings[value]
            elif self.input_kind == "image":
                key = Path(value).stem
            elif self.input_kind in {"text", "description", "pair_text"}:
                key = hashlib.sha256(value.encode("utf-8")).hexdigest()
            elif value in self.matrix.rows:
                key = value
            else:
                raise PreparedInputError(
                    f"Prepared {self.input_kind} inputs need explicit ID bindings."
                )
            keys.append(key)
        return keys

    def forward(self, inputs):
        dtype = self._device.dtype if np.dtype(self.matrix.entry["dtype"]).kind == "f" else None
        return self.matrix.read(self._keys(inputs), device=self.device, dtype=dtype)[0]

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                serialized = self.encoding_spec.get("tokenizer")
                if serialized is not None:
                    from tokenizers import Tokenizer
                    from transformers import PreTrainedTokenizerFast

                    tokenizer = Tokenizer.from_str(json.dumps(serialized))
                    self._tokenizer = PreTrainedTokenizerFast(
                        tokenizer_object=tokenizer,
                        **self.encoding_spec.get("special_tokens", {}),
                    )
                else:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, revision=self.revision, local_files_only=True
                    )
                for setting in ("padding_side", "truncation_side"):
                    setattr(self._tokenizer, setting, self.encoding_spec.get(setting, "right"))
            except (ImportError, OSError, ValueError) as error:
                raise PreparedInputError(
                    "This branch requires its matching local tokenizer; no model weights are needed."
                ) from error
        return self._tokenizer

    def token_mask(self, inputs):
        keys = self._keys(inputs)
        if self.matrix.entry.get("mask"):
            return self.matrix.read_mask(keys, device=self.device)
        if self.input_kind not in {"text", "description", "pair_text", "caption", "arguments"}:
            raise PreparedInputError("This prepared image feature does not have a token mask.")
        return self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["attention_mask"].to(device=self.device, dtype=torch.bool)

    def inputs_from_rows(self, rows):
        field = "image_path" if self.input_kind == "image" else "text"
        return [row[field] for row in rows]

    def get_extra_state(self):
        return {
            "format_version": 1,
            "catalog_identity": self.matrix.identity,
            "bindings": dict(self._bindings),
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("catalog_identity") != self.matrix.identity
            or not isinstance(state.get("bindings"), dict)
        ):
            raise PreparedInputError("Prepared encoder catalog differs from the checkpoint.")
        self.bind(list(state["bindings"]), list(state["bindings"].values()))
