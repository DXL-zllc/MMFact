"""NSLM's fixed, reviewed reverse-image entities; no retrieval during training."""

import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from torch import nn

from mmfact.data.features import file_digest, fingerprint


def evidence_protocol():
    return {
        "provider": "Google Cloud Vision",
        "api": "v1/images:annotate",
        "feature": "WEB_DETECTION",
        "max_results": 10,
        "selection": "reviewed-web-entity-descriptions-only-v1",
        "temporal_scope": "current-snapshot",
    }


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _timestamp(value):
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def validate_record(record):
    required = {"image_sha256", "retrieved_at", "response_sha256", "entities", "sources", "review"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("NSLM evidence record has an incompatible schema.")
    review, entities = record["review"], record["entities"]
    if (
        not _sha(record["image_sha256"])
        or not _sha(record["response_sha256"])
        or not _timestamp(record["retrieved_at"])
        or not isinstance(review, dict)
        or set(review) != {"reviewer", "reviewed_at", "note", "temporal_scope"}
        or not all(isinstance(review[k], str) and review[k].strip() for k in ("reviewer", "note"))
        or not _timestamp(review["reviewed_at"])
        or review["temporal_scope"] != "current-snapshot"
        or not isinstance(entities, list)
        or not isinstance(record["sources"], list)
        or any(
            not isinstance(url, str) or not url.startswith(("https://", "http://"))
            for url in record["sources"]
        )
    ):
        raise ValueError(
            "NSLM evidence needs actual image/response hashes, times and source review."
        )
    identifiers = set()
    for entity in entities:
        if (
            not isinstance(entity, dict)
            or set(entity) != {"entity_id", "description", "score"}
            or not all(
                isinstance(entity[k], str) and entity[k].strip()
                for k in ("entity_id", "description")
            )
            or entity["entity_id"] in identifiers
            or type(entity["score"]) not in (int, float)
            or not math.isfinite(entity["score"])
            or entity["score"] < 0
        ):
            raise ValueError("NSLM reviewed entities require unique IDs, descriptions and scores.")
        # This rejects explicit verdicts; it is NOT proof of historical availability.
        if re.search(
            r"\b(?:fake|false|hoax|debunk\w*|fact.?check\w*|misinformation)\b|辟谣|谣言|虚假|不实|核查",
            entity["description"],
            re.I,
        ):
            raise ValueError("NSLM context cannot contain explicit verification/verdict entities.")
        identifiers.add(entity["entity_id"])
    return deepcopy(record)


class ReverseImageEvidence(nn.Module):
    def __init__(self, manifest):
        super().__init__()
        self.manifest = Path(manifest) if manifest else None
        self.records, self.loaded = {}, False

    def load(self):
        if self.manifest is None:
            raise ValueError("NSLM needs a reviewed reverse-image evidence manifest.")
        parsed = {}
        with self.manifest.open(encoding="utf-8") as stream:
            if json.loads(next(stream, "null")) != {"schema": 1, "generator": evidence_protocol()}:
                raise ValueError("NSLM evidence manifest has an incompatible retrieval protocol.")
            for line in stream:
                if not line.strip():
                    continue
                record = validate_record(json.loads(line))
                key = record["image_sha256"]
                if key in parsed or (key in self.records and self.records[key] != record):
                    raise ValueError("NSLM evidence records duplicate or conflict.")
                parsed[key] = record
        self.records.update(parsed)
        self.loaded = True

    def contexts(self, paths):
        result = []
        for path in paths:
            key = file_digest(path)
            if not self.loaded or (
                key not in self.records and self.manifest and self.manifest.is_file()
            ):
                self.load()
            if key not in self.records:
                raise ValueError(
                    "Missing NSLM image evidence record; run the resource preparation script."
                )
            result.append(
                " ".join(entity["description"] for entity in self.records[key]["entities"])
            )
        return result

    def get_extra_state(self):
        return {
            "version": 1,
            "protocol": evidence_protocol(),
            "records": deepcopy(self.records),
            "sha256": fingerprint(self.records),
            "loaded": self.loaded,
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("protocol") != evidence_protocol()
            or not isinstance(state.get("records"), dict)
            or type(state.get("loaded")) is not bool
        ):
            raise ValueError("NSLM evidence checkpoint has an incompatible protocol.")
        records = {key: validate_record(value) for key, value in state["records"].items()}
        if any(key != value["image_sha256"] for key, value in records.items()) or fingerprint(
            records
        ) != state.get("sha256"):
            raise ValueError("NSLM evidence checkpoint content hash is invalid.")
        self.records, self.loaded = records, state["loaded"]
