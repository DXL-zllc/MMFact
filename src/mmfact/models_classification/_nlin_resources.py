"""NLIN's fixed image descriptions and two independent Wikidata knowledge streams."""

import hashlib
import re
from copy import deepcopy
from pathlib import Path

from torch import nn

from mmfact.models_classification._entities import _json, resource_digest
from mmfact.models_classification._generated import pair_identity


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def resource_protocol(language, tools):
    if language not in {"en", "zh"} or not isinstance(tools, dict):
        raise ValueError("NLIN resources require en/zh and explicit tools.")
    expected = {"caption", "dense", "ocr", "linker", "knowledge", "translation"}
    if set(tools) != expected:
        raise ValueError("NLIN requires all six tool identities.")
    names = {
        "caption": "BLIP",
        "dense": "VinVL",
        "ocr": "EasyOCR",
        "linker": "TagMe",
        "knowledge": "Wikidata",
        "translation": "MarianMT" if language == "zh" else "identity",
    }
    for key, name in names.items():
        identity = tools[key]
        if (
            not isinstance(identity, dict)
            or not {"name", "revision", "settings"} <= identity.keys()
            or set(identity) - {"name", "revision", "settings", "weights"}
            or identity["name"] != name
            or not _text(identity["revision"])
            or not isinstance(identity["settings"], dict)
        ):
            raise ValueError(f"NLIN {key} needs its actual name, revision and settings.")
    # Reject non-JSON / non-finite settings before recording their identity.
    resource_digest(tools)
    return {
        "task": "nlin-resource-adapted",
        "version": 1,
        "language": language,
        "tools": deepcopy(tools),
        "linking_inputs": ["news_english", "caption-plus-ocr_english-no-dense"],
        "knowledge": "entity-label-colon-English-Wikidata-description-at-recorded-revision",
        "prediction_inputs": "raw-text-and-image-content-sha256-no-labels",
    }


def empty_reason(entities):
    if not entities:
        return "no_linked_entities"
    if not any(entity["description"] for entity in entities):
        return "no_wikidata_description"
    return None


def make_record(text, path, payload):
    record = {**pair_identity(text, path), **deepcopy(payload)}
    return {**record, "record_sha256": resource_digest(record)}


def validate_record(record, language):
    fields = {
        "pair_sha256",
        "text_sha256",
        "image_sha256",
        "news_english",
        "caption",
        "dense",
        "ocr",
        "ocr_english",
        "text_entities",
        "visual_entities",
        "empty_reasons",
        "audit",
        "record_sha256",
    }
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("NLIN resource record has invalid fields.")
    if any(not _sha(record[key]) for key in ("pair_sha256", "text_sha256", "image_sha256")):
        raise ValueError("NLIN content identities must be SHA256 digests.")
    expected_pair = hashlib.sha256(
        (record["text_sha256"] + record["image_sha256"]).encode("ascii")
    ).hexdigest()
    if record["pair_sha256"] != expected_pair:
        raise ValueError("NLIN resource pair identity is inconsistent.")
    if (
        not _text(record["news_english"])
        or not _text(record["caption"])
        or not isinstance(record["ocr"], list)
        or any(not _text(item) for item in record["ocr"])
        or not isinstance(record["ocr_english"], list)
        or len(record["ocr"]) != len(record["ocr_english"])
        or any(not _text(item) for item in record["ocr_english"])
        or (language == "en" and record["ocr"] != record["ocr_english"])
        or not isinstance(record["dense"], list)
    ):
        raise ValueError("NLIN caption, dense labels, OCR or translations are invalid.")
    for item in record["dense"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"object", "attributes"}
            or not _text(item["object"])
            or not isinstance(item["attributes"], list)
            or any(not _text(value) for value in item["attributes"])
        ):
            raise ValueError("NLIN VinVL dense labels need objects and attributes.")
    for side in ("text", "visual"):
        entities = record[f"{side}_entities"]
        if not isinstance(entities, list):
            raise ValueError("NLIN requires both entity lists.")
        seen = set()
        for item in entities:
            if (
                not isinstance(item, dict)
                or set(item) != {"title", "qid", "label", "description", "revision", "reason"}
                or not _text(item["title"])
                or item["title"] in seen
                or not isinstance(item["description"], str)
                or not isinstance(item["label"], str)
            ):
                raise ValueError("NLIN entity record is invalid or duplicated.")
            seen.add(item["title"])
            if item["qid"] is None:
                if any(item[key] is not None for key in ("revision",)) or (
                    item["description"] or item["label"] or item["reason"] != "no_qid"
                ):
                    raise ValueError("NLIN unlinked entities cannot claim knowledge.")
            elif (
                not isinstance(item["qid"], str)
                or re.fullmatch(r"Q[1-9][0-9]*", item["qid"]) is None
                or type(item["revision"]) is not int
                or item["revision"] < 1
                or not _text(item["label"])
                or item["reason"] != (None if item["description"].strip() else "no_description")
            ):
                raise ValueError("NLIN Wikidata knowledge needs QID and revision evidence.")
    reasons = {
        "dense": None if record["dense"] else "no_objects",
        "ocr": None if record["ocr"] else "no_text",
        "text": empty_reason(record["text_entities"]),
        "visual": empty_reason(record["visual_entities"]),
    }
    if record["empty_reasons"] != reasons:
        raise ValueError("NLIN missing-resource reasons disagree with their contents.")
    stages = {"caption", "dense", "ocr", "translation", "text_knowledge", "visual_knowledge"}
    if not isinstance(record["audit"], dict) or set(record["audit"]) != stages:
        raise ValueError("NLIN requires all actual tool-response audit records.")
    for response in record["audit"].values():
        if (
            not isinstance(response, dict)
            or set(response) != {"input_sha256", "result", "raw", "completed_at"}
            or not _sha(response["input_sha256"])
            or not _text(response["completed_at"])
            or response["raw"] is None
        ):
            raise ValueError("NLIN tools require actual responses, input digests and timestamps.")
    audit = record["audit"]
    expected = {
        "caption": record["image_sha256"],
        "dense": record["image_sha256"],
        "ocr": record["image_sha256"],
        "translation": resource_digest(
            {"news_sha256": record["text_sha256"], "ocr": record["ocr"]}
        ),
        "text_knowledge": hashlib.sha256(record["news_english"].encode()).hexdigest(),
        "visual_knowledge": hashlib.sha256(
            (record["caption"] + "\n" + "\n".join(record["ocr_english"])).encode()
        ).hexdigest(),
    }
    results = {
        "caption": record["caption"],
        "dense": record["dense"],
        "ocr": record["ocr"],
        "translation": {
            "news_english": record["news_english"],
            "ocr_english": record["ocr_english"],
        },
        "text_knowledge": record["text_entities"],
        "visual_knowledge": record["visual_entities"],
    }
    if any(
        audit[key]["input_sha256"] != expected[key] or audit[key]["result"] != results[key]
        for key in stages
    ):
        raise ValueError("NLIN tool responses do not match the declared input or derived resource.")
    content = {key: value for key, value in record.items() if key != "record_sha256"}
    if record["record_sha256"] != resource_digest(content):
        raise ValueError("NLIN resource record digest mismatch.")
    return deepcopy(record)


class NLINResources(nn.Module):
    """Lazily load all valid records; restore evidence without external files or services."""

    def __init__(self, language, manifest, tools):
        super().__init__()
        self.protocol = resource_protocol(language, tools)
        if not isinstance(manifest, (str, Path)) or not str(manifest).strip():
            raise ValueError("NLIN requires a nonempty resource manifest path.")
        self.manifest = Path(manifest).expanduser()
        self.records, self.loaded = {}, False

    def load(self):
        with self.manifest.open(encoding="utf-8") as stream:
            header = _json(next(stream, "null"))
            if header != {"schema": 1, "protocol": self.protocol}:
                raise ValueError("NLIN resource protocol mismatch.")
            records = {}
            for line in stream:
                if not line.strip():
                    continue
                record = validate_record(_json(line), self.protocol["language"])
                key = record["pair_sha256"]
                if key in records or (key in self.records and self.records[key] != record):
                    raise ValueError("NLIN duplicate or conflicting resource pair.")
                records[key] = record
        self.records.update(records)
        self.loaded = True
        return self

    def lookup(self, text, path):
        if not self.loaded:
            self.load()
        identity = pair_identity(text, path)
        key = identity["pair_sha256"]
        if key not in self.records and self.manifest.is_file():
            self.load()
        if key not in self.records:
            raise ValueError(
                "Missing NLIN resource; download the required materials with --download-materials."
            )
        record = self.records[key]
        if self.protocol["language"] == "en" and record["news_english"] != text:
            raise ValueError("NLIN English resources must preserve the original news text.")
        return deepcopy(record)

    def get_extra_state(self):
        payload = {
            "protocol": deepcopy(self.protocol),
            "records": deepcopy(self.records),
            "loaded": self.loaded,
        }
        return {"version": 1, **payload, "sha256": resource_digest(payload)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "protocol", "records", "loaded", "sha256"}
            or state["version"] != 1
            or state["protocol"] != self.protocol
            or type(state["loaded"]) is not bool
            or not isinstance(state["records"], dict)
            or (not state["loaded"] and state["records"])
        ):
            raise ValueError("Invalid NLIN resource checkpoint.")
        payload = {key: state[key] for key in ("protocol", "records", "loaded")}
        if resource_digest(payload) != state["sha256"]:
            raise ValueError("NLIN resource checkpoint digest mismatch.")
        records = {}
        for key, value in state["records"].items():
            record = validate_record(value, self.protocol["language"])
            if key != record["pair_sha256"]:
                raise ValueError("NLIN checkpoint resource key mismatch.")
            records[key] = record
        self.records, self.loaded = records, state["loaded"]
