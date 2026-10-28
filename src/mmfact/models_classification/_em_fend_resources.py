"""Typed EM-FEND evidence with original confidences and content-bound recovery."""

import math
from pathlib import Path

from torch import nn

from mmfact.models_classification._generated import GeneratedTextStore, pair_identity

TYPES = ("person", "location", "context")


def _sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


class EntityResources(GeneratedTextStore):
    """Use the shared atomic manifest/checkpoint contract with a typed payload."""

    def __init__(self, language, manifest, producer_revision):
        nn.Module.__init__(self)
        if (
            language not in {"en", "zh"}
            or not isinstance(producer_revision, str)
            or not producer_revision.strip()
        ):
            raise ValueError("EM-FEND resources require en/zh and an explicit producer revision.")
        self.protocol = {
            "task": "em-fend",
            "language": language,
            "producer_revision": producer_revision,
            "contract": "typed-entities-provider-confidence-v1",
        }
        self.manifest = Path(manifest).expanduser()
        self.records, self.loaded = {}, False

    def validate_record(self, record):
        if not isinstance(record, dict) or set(record) != {
            "pair_sha256",
            "text_sha256",
            "image_sha256",
            "ocr",
            "textual",
            "visual",
            "excluded",
            "provenance",
        }:
            raise ValueError("EM-FEND resource fields do not match its typed contract.")
        for field in ("pair_sha256", "text_sha256", "image_sha256"):
            if not _sha(record[field]):
                raise ValueError("EM-FEND content identities must be SHA256 digests.")
        import hashlib

        expected = hashlib.sha256(
            (record["text_sha256"] + record["image_sha256"]).encode("ascii")
        ).hexdigest()
        if record["pair_sha256"] != expected or not isinstance(record["ocr"], str):
            raise ValueError("EM-FEND pair identity or OCR is invalid.")
        textual = record["textual"]
        if not isinstance(textual, dict) or set(textual) != set(TYPES):
            raise ValueError("EM-FEND textual entities must include all three types.")
        for entities in textual.values():
            if (
                not isinstance(entities, list)
                or any(not isinstance(name, str) or not name.strip() for name in entities)
                or len(set(entities)) != len(entities)
            ):
                raise ValueError("EM-FEND textual entity lists must contain unique nonempty names.")
        visual, excluded = record["visual"], record["excluded"]
        if not isinstance(visual, list) or not isinstance(excluded, list):
            raise ValueError("EM-FEND visual entities and exclusions must be lists.")
        seen = set()
        for item in visual:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "type", "confidence"}
                or item["type"] not in TYPES
                or not isinstance(item["name"], str)
                or not item["name"].strip()
            ):
                raise ValueError("EM-FEND visual entity type/name is invalid.")
            score = item["confidence"]
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("EM-FEND confidence must be a finite provider score in [0,1].")
            key = (item["type"], item["name"])
            if key in seen:
                raise ValueError("EM-FEND visual entities must be unique by type and name.")
            seen.add(key)
        for item in excluded:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "type", "reason"}
                or item["type"] not in TYPES
                or not isinstance(item["name"], str)
                or not item["name"].strip()
                or item["reason"] != "missing_confidence"
            ):
                raise ValueError(
                    "EM-FEND excluded entities require an explicit missing-confidence reason."
                )
        provenance = record["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "provider",
            "revision",
            "responses_sha256",
            "mapping_sha256",
            "text_model",
            "time_utc",
        }:
            raise ValueError("EM-FEND resource provenance is incomplete.")
        if (
            any(
                not isinstance(provenance[name], str) or not provenance[name].strip()
                for name in ("provider", "revision", "text_model", "time_utc")
            )
            or provenance["revision"] != self.protocol["producer_revision"]
            or not _sha(provenance["responses_sha256"])
            or not _sha(provenance["mapping_sha256"])
        ):
            raise ValueError("EM-FEND resource provenance differs from its producer protocol.")
        from copy import deepcopy

        return deepcopy(record)

    def lookup(self, text, image_path):
        if not self.loaded:
            self.load()
        key = pair_identity(text, image_path)["pair_sha256"]
        if key not in self.records and self.manifest.is_file():
            self.load()
        if key not in self.records:
            raise ValueError(
                "No EM-FEND entity/OCR evidence matches this text and image; prepare resources first."
            )
        from copy import deepcopy

        return deepcopy(self.records[key])
