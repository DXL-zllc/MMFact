"""Fixed MMRGV rationales, with auditable training-only demonstrations."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from torch import nn

from mmfact.models_classification._entities import resource_digest
from mmfact.models_classification._generated import pair_identity

VIEWS = ("td", "itc", "id")


def rationale_protocol(language, model, revision, max_tokens=512):
    if language not in {"en", "zh"}:
        raise ValueError("MMRGV resources require en/zh.")
    if any(not isinstance(value, str) or not value.strip() for value in (model, revision)):
        raise ValueError("MMRGV generation requires explicit model and revision identities.")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("MMRGV max_tokens must be a positive integer.")
    return {
        "task": "mmrgv",
        "version": 1,
        "language": language,
        "model": model,
        "revision": revision,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_tokens": max_tokens,
        "demonstrations": "four-training-only-two-per-class-exclude-text-and-image-v1",
        "prompts": {
            "td": "Analyze only the news text: source, event date, logical coherence, writing "
            "style and emotional cues. Explain your reasoning about its authenticity.",
            "itc": "Compare the news text and image: people, place, time and event details. "
            "Explain whether they describe a consistent event and what that implies.",
            "id": "Analyze only the image: provenance cues, watermarks, splicing, tampering "
            "and visual anomalies. Explain your reasoning about its authenticity.",
            "caption": "Describe only what is visibly present in the image. Return a concise "
            "caption; do not judge the authenticity of any news claim.",
        },
    }


def _json(line):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("MMRGV JSON contains duplicate keys.")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"MMRGV JSON contains non-finite {value}.")

    return json.loads(line, object_pairs_hook=pairs, parse_constant=invalid)


def _identity(record):
    fields = {"pair_sha256", "text_sha256", "image_sha256"}
    if not isinstance(record, dict) or not fields <= record.keys():
        raise ValueError("MMRGV record lacks a content identity.")
    if any(
        not isinstance(record[key], str)
        or len(record[key]) != 64
        or any(char not in "0123456789abcdef" for char in record[key])
        for key in fields
    ):
        raise ValueError("MMRGV content identities must be SHA256 digests.")
    expected = hashlib.sha256(
        (record["text_sha256"] + record["image_sha256"]).encode("ascii")
    ).hexdigest()
    if record["pair_sha256"] != expected:
        raise ValueError("MMRGV pair identity is inconsistent.")
    return fields


def validate_rationales(values):
    if not isinstance(values, dict) or set(values) != set(VIEWS):
        raise ValueError("MMRGV requires exactly TD, ITC and ID rationales.")
    for value in values.values():
        if (
            not isinstance(value, dict)
            or set(value) != {"text", "judgment"}
            or not isinstance(value["text"], str)
            or not value["text"].strip()
            or type(value["judgment"]) is not int
            or value["judgment"] not in {0, 1, 2}
        ):
            raise ValueError("MMRGV rationales need nonempty text and judgment 0/1/2.")
    return deepcopy(values)


def validate_demonstrations(demos):
    if not isinstance(demos, dict) or not demos:
        raise ValueError("MMRGV requires a training demonstration pool.")
    for key, value in demos.items():
        fields = _identity(value)
        if (
            key != value["pair_sha256"]
            or set(value) != fields | {"label", "rationales"}
            or type(value["label"]) is not int
            or value["label"] not in {0, 1}
        ):
            raise ValueError("MMRGV demonstration has invalid fields or training label.")
        validate_rationales(value["rationales"])
        if any(
            rationale["judgment"] != value["label"] for rationale in value["rationales"].values()
        ):
            raise ValueError("MMRGV demonstrations must explain their known training label.")
    return deepcopy(demos)


def choose_demonstrations(identity, demos):
    selected = []
    for label in (0, 1):
        candidates = [
            key
            for key, demo in sorted(demos.items())
            if demo["label"] == label
            and demo["text_sha256"] != identity["text_sha256"]
            and demo["image_sha256"] != identity["image_sha256"]
        ]
        if len(candidates) < 2:
            raise ValueError("MMRGV needs two disjoint training demonstrations per class.")
        selected.extend(candidates[:2])
    return selected


class RationaleStore(nn.Module):
    """Resources are restored with weights; no inference service or labels are needed."""

    def __init__(self, language, manifest, model, revision, max_tokens=512):
        super().__init__()
        if not isinstance(manifest, (str, Path)) or not str(manifest).strip():
            raise ValueError("MMRGV requires a nonempty resource manifest path.")
        self.protocol = rationale_protocol(language, model, revision, max_tokens)
        self.manifest = Path(manifest).expanduser()
        self.demonstrations, self.records, self.loaded = {}, {}, False

    def validate_record(self, record, demos=None):
        demos = self.demonstrations if demos is None else demos
        fields = _identity(record)
        if (
            set(record) != fields | {"caption", "rationales", "demo_keys"}
            or not isinstance(record["caption"], str)
            or not record["caption"].strip()
        ):
            raise ValueError("MMRGV resource has invalid fields or an empty caption.")
        validate_rationales(record["rationales"])
        if record["demo_keys"] != choose_demonstrations(record, demos):
            raise ValueError("MMRGV demonstrations overlap the query or differ from the protocol.")
        return deepcopy(record)

    def load(self):
        with self.manifest.open(encoding="utf-8") as stream:
            header = _json(next(stream, "null"))
            if (
                not isinstance(header, dict)
                or set(header) != {"schema", "protocol", "demonstrations"}
                or type(header["schema"]) is not int
                or header["schema"] != 1
                or header["protocol"] != self.protocol
            ):
                raise ValueError("MMRGV resource protocol mismatch.")
            demos = validate_demonstrations(header["demonstrations"])
            if self.loaded and demos != self.demonstrations:
                raise ValueError("MMRGV demonstration pool conflicts with restored resources.")
            records = {}
            for line in stream:
                if not line.strip():
                    continue
                record = self.validate_record(_json(line), demos)
                key = record["pair_sha256"]
                if key in records or (key in self.records and self.records[key] != record):
                    raise ValueError("MMRGV duplicate or conflicting resource pair.")
                records[key] = record
        self.demonstrations = demos
        self.records.update(records)
        self.loaded = True
        return self

    def lookup(self, text, image_path):
        identity = pair_identity(text, image_path)
        if not self.loaded:
            self.load()
        key = identity["pair_sha256"]
        if key not in self.records and self.manifest.is_file():
            self.load()
        if key not in self.records:
            raise ValueError(
                "Missing MMRGV rationale; download the required materials with --download-materials."
            )
        return deepcopy(self.records[key])

    def get_extra_state(self):
        payload = {
            "protocol": deepcopy(self.protocol),
            "demonstrations": deepcopy(self.demonstrations),
            "records": deepcopy(self.records),
            "loaded": self.loaded,
        }
        return {"version": 1, **payload, "sha256": resource_digest(payload)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {"version", "protocol", "demonstrations", "records", "loaded", "sha256"}
            or state["version"] != 1
            or state["protocol"] != self.protocol
            or type(state["loaded"]) is not bool
            or not isinstance(state["records"], dict)
        ):
            raise ValueError("Invalid MMRGV resource checkpoint.")
        payload = {key: state[key] for key in ("protocol", "demonstrations", "records", "loaded")}
        if resource_digest(payload) != state["sha256"]:
            raise ValueError("MMRGV resource checkpoint digest mismatch.")
        demos = validate_demonstrations(state["demonstrations"]) if state["loaded"] else {}
        if not state["loaded"] and (state["records"] or state["demonstrations"]):
            raise ValueError("Unloaded MMRGV resources cannot contain evidence.")
        records = {}
        for key, value in state["records"].items():
            record = self.validate_record(value, demos)
            if key != record["pair_sha256"]:
                raise ValueError("MMRGV resource checkpoint key mismatch.")
            records[key] = record
        self.demonstrations, self.records, self.loaded = demos, records, state["loaded"]
