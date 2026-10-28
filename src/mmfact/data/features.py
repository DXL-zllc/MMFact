"""Content-addressed, disk-backed features shared by compatible encoders."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from mmfact.data.cache_policy import input_mode, prepared_only, require_prepared

_SCHEMA = 1


def compatible_spec(value):
    """Ignore only PyTorch wheel backend tags, retaining all algorithm versions.

    A frozen tensor does not depend on which CUDA wheel reads it. The original
    producer identity remains in metadata and in the content-addressed path.
    """
    if isinstance(value, dict):
        result = {key: compatible_spec(child) for key, child in value.items()}
        versions = result.get("versions")
        if isinstance(versions, dict):
            for package in ("torch", "torchvision"):
                if isinstance(versions.get(package), str):
                    versions[package] = re.sub(r"\+(?:cu[0-9]+|cpu)$", "", versions[package])
        return result
    if isinstance(value, (list, tuple)):
        return [compatible_spec(child) for child in value]
    return value


def fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_digest(value):
    """Hash nested tensors and primitives without serializing tensor copies."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            digest.update(fingerprint(["tensor", str(item.dtype), list(item.shape)]).encode())
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        elif isinstance(item, dict):
            digest.update(fingerprint(["dict", len(item)]).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(fingerprint([type(item).__name__, len(item)]).encode())
            for child in item:
                update(child)
        else:
            digest.update(fingerprint([type(item).__name__, item]).encode())

    update(value)
    return digest.hexdigest()


def state_digest(module):
    """Hash actual parameters and buffers without copying the whole network."""
    return _value_digest(module.state_dict())


def _corrupt(path, error):
    return ValueError(f"Invalid feature cache at {path}; delete it and rebuild. {error}")


class FeatureCache:
    """One .npy array per encoder specification and ordered set of input keys."""

    def __init__(self, root, spec, keys, *, mode=None):
        self.mode = input_mode(mode)
        self.keys = list(keys)
        if not all(isinstance(key, str) for key in self.keys):
            raise ValueError("Feature keys must be strings.")
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("Feature keys must be nonempty and unique.")
        self.identity = json.loads(json.dumps({"schema": _SCHEMA, "spec": spec, "keys": self.keys}))
        self.path = Path(root) / "features" / fingerprint(self.identity)
        self.indices = {key: index for index, key in enumerate(self.keys)}
        self.hit, self.shape, self._array = False, None, None
        if self.path.exists():
            self._load()
            self.hit = True
        elif prepared_only(self.mode):
            self._find_covering_cache()

    def _find_covering_cache(self):
        """Reuse a complete immutable cache for a larger input collection."""
        needed, candidates = set(self.keys), []
        for directory in self.path.parent.glob("*"):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                metadata = json.loads((directory / "metadata.json").read_text("utf-8"))
                identity = metadata["identity"]
                keys = identity["keys"]
                if (
                    identity["schema"] == _SCHEMA
                    and compatible_spec(identity["spec"]) == compatible_spec(self.identity["spec"])
                    and isinstance(keys, list)
                    and all(isinstance(key, str) for key in keys)
                    and len(set(keys)) == len(keys)
                    and needed.issubset(keys)
                    and directory.name == fingerprint(identity)
                ):
                    candidates.append((len(keys), directory.name, directory, identity))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if not candidates:
            return
        _, _, self.path, self.identity = min(candidates)
        self.keys = self.identity["keys"]
        self.indices = {key: index for index, key in enumerate(self.keys)}
        self._load()
        self.hit = True

    def _load(self):
        try:
            metadata = json.loads((self.path / "metadata.json").read_text("utf-8"))
            if metadata["identity"] != self.identity:
                raise ValueError("Specification or input keys do not match.")
            array_path = self.path / "features.npy"
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if (
                array.ndim < 2
                or array.shape[0] != len(self.keys)
                or list(array.shape) != metadata["shape"]
                or str(array.dtype) != metadata["dtype"]
                or file_digest(array_path) != metadata["sha256"]
            ):
                raise ValueError("Array shape, dtype or checksum does not match.")
            self._array, self.shape = array, tuple(array.shape)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            raise _corrupt(self.path, error) from error

    def build(self, batches_callable):
        """Stream batches into an unpublished array, then atomically publish it."""
        if self._array is not None:
            return self
        if self.path.exists():
            self._load()
            self.hit = True
            return self
        if prepared_only(self.mode):
            self._find_covering_cache()
            if self._array is not None:
                return self
        require_prepared(self.path, "feature cache", mode=self.mode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=self.path.parent))
        array, offset = None, 0
        try:
            for batch in batches_callable():
                if not isinstance(batch, torch.Tensor) or batch.ndim < 2:
                    raise ValueError("Feature batches must be tensors with shape [B, ...].")
                batch = batch.detach().cpu().contiguous()
                if not torch.isfinite(batch).all():
                    raise ValueError("Features must be finite.")
                values = batch.numpy()
                if array is None:
                    array = np.lib.format.open_memmap(
                        temporary / "features.npy",
                        mode="w+",
                        dtype=values.dtype,
                        shape=(len(self.keys), *values.shape[1:]),
                    )
                stop = offset + len(values)
                if (
                    stop > len(self.keys)
                    or values.shape[1:] != array.shape[1:]
                    or values.dtype != array.dtype
                ):
                    raise ValueError("Inconsistent feature batch shape, dtype or row count.")
                array[offset:stop] = values
                offset = stop
            if array is None or offset != len(self.keys):
                raise ValueError("Feature row count must equal the number of input keys.")
            array.flush()
            metadata = {
                "identity": self.identity,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": file_digest(temporary / "features.npy"),
            }
            del array
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False), "utf-8"
            )
            try:
                temporary.rename(self.path)
            except OSError:
                if not self.path.exists():
                    raise
                self.hit = True  # Another writer completed the same cache first.
            self._load()
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return self

    def read(self, keys):
        if self._array is None:
            raise ValueError("Build the feature cache before reading it.")
        try:
            rows = [self.indices[key] for key in keys]
        except KeyError as error:
            raise KeyError(f"Input key {error.args[0]!r} is absent from {self.path}.") from error
        return torch.from_numpy(np.array(self._array[rows], copy=True))


def resource_path(root, spec):
    """Path for a small fitted resource, such as training-only Word2Vec weights."""
    return Path(root) / "resources" / f"{fingerprint({'schema': _SCHEMA, 'spec': spec})}.pt"


def load_resource(root, spec, build, *, mode=None):
    """Read or atomically create a tensor/primitive dictionary with weights_only."""
    mode = input_mode(mode)
    path = resource_path(root, spec)
    identity = json.loads(json.dumps({"schema": _SCHEMA, "spec": spec}))
    if not path.exists():
        require_prepared(path, "fitted resource", mode=mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".building-", dir=path.parent)
        os.close(descriptor)
        try:
            data = build()
            if not isinstance(data, dict):
                raise ValueError("Cached resources must be dictionaries.")
            with open(temporary, "wb") as stream:
                torch.save(
                    {"identity": identity, "data": data, "sha256": _value_digest(data)}, stream
                )
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    try:
        resource = torch.load(path, map_location="cpu", weights_only=True)
        stored = resource["identity"]
        if (
            path.stem != fingerprint(stored)
            or stored != identity
            or not isinstance(resource["data"], dict)
        ):
            raise ValueError("Resource specification or payload does not match.")
        if resource["sha256"] != _value_digest(resource["data"]):
            raise ValueError("Resource payload checksum does not match.")
        return resource["data"]
    except Exception as error:
        raise _corrupt(path, error) from error
