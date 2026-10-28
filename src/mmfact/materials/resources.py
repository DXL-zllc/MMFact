"""Adapt the clean Hub annotations to model inputs without production audit headers."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path

from torch import nn

from mmfact.data.cache_policy import PreparedInputError
from mmfact.data.features import fingerprint
from mmfact.materials.encoders import hub_inputs, material_store

_TASKS = {"aar", "mmfn", "mmrgv", "em_fend", "eric_fnd", "nlin", "nslm", "event_radar"}
_ROLES = ("subject", "object", "location", "predicate")
_TYPES = ("person", "location", "context")


def _sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _identity(text, path):
    if not isinstance(text, str) or not text.strip():
        raise PreparedInputError("Hub resources require nonempty original news text.")
    image_id = Path(path).stem
    if not _sha(image_id):
        raise PreparedInputError("Hub image filenames must contain the canonical image_id.")
    text_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
    pair_id = hashlib.sha256(f"{text_id}:{image_id}".encode("ascii")).hexdigest()
    return {"text_id": text_id, "image_id": image_id, "pair_id": pair_id}


def _string(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise PreparedInputError(f"Hub resources require a valid {name} string.")
    return value


def _basic(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_basic(item) for item in value)
    return isinstance(value, dict) and all(
        isinstance(key, str) and _basic(item) for key, item in value.items()
    )


def material_resources(config, task, native_factory, /, *args, **options):
    """Keep strict native stores for raw inputs; Hub inputs use the clean shared tables."""
    if not hub_inputs(config):
        return native_factory(*args, **options)
    supplied_task = options.pop("task", task)
    if supplied_task.replace("-", "_") != task.replace("-", "_"):
        raise PreparedInputError("Resource factory task arguments disagree.")
    return HubResources(config, task, **options)


class HubResources(nn.Module):
    """Join by content IDs, and checkpoint only records actually used by the model."""

    def __init__(self, config, task, **options):
        super().__init__()
        task = task.lower().replace("-", "_")
        if task not in _TASKS:
            raise PreparedInputError(f"Unsupported Hub resource task: {task!r}.")
        self.task = task
        self.language = options.get(
            "language", "en" if config["dataset"]["name"] == "gossipcop" else "zh"
        )
        self.max_entities = options.get("max_entities", 10)
        self.max_objects = options.get("max_objects", 36)
        if self.language not in {"en", "zh"} or any(
            type(value) is not int or value < 1 for value in (self.max_entities, self.max_objects)
        ):
            raise PreparedInputError("Hub resources require en/zh and positive resource limits.")
        self.protocol = {
            "layout": "hub",
            "version": 1,
            "task": task,
            "dataset": config["dataset"]["name"],
            "language": self.language,
            "max_entities": self.max_entities,
            "max_objects": self.max_objects,
        }
        self._config = config
        self._tables, self.records = {}, {}
        self.loaded = False

    def _table(self, relative, key):
        identity = (relative, key)
        if identity not in self._tables:
            self._tables[identity] = material_store(self._config).records(relative, key=key)
        return self._tables[identity]

    def _row(self, relative, key, identity):
        return self._table(relative, key)[identity]

    def _image(self, image_id):
        return self._row("annotations/images.jsonl", "image_id", image_id)

    def _text(self, text_id):
        return self._row("annotations/texts.jsonl", "text_id", text_id)

    def _pair(self, relative, identity):
        record = self._row(relative, "pair_id", identity["pair_id"])
        if any(record.get(key) != value for key, value in identity.items()):
            raise PreparedInputError("Hub pair annotation does not match its content IDs.")
        return record

    def _entities(self, relative, key, identity):
        link = self._row(relative, key, identity)
        return [
            self._row("annotations/encyclopedia.jsonl", "entity_id", entity_id)
            for entity_id in link["entity_ids"]
        ]

    def _assemble(self, text, identity):
        tid, iid = identity["text_id"], identity["image_id"]
        if self.task in {"aar", "mmrgv"}:
            relative = {
                "aar": "methods/aar/arguments.jsonl",
                "mmrgv": "methods/mmrgv/rationales.jsonl",
            }[self.task]
            source = self._pair(relative, identity)
            fields = ("genuine", "fake") if self.task == "aar" else ("caption", "rationales")
            return {**identity, **{key: deepcopy(source[key]) for key in fields}}
        if self.task == "mmfn":
            source = self._pair("annotations/pairs.jsonl", identity)
            return {**identity, "ocr": self._image(iid)["ocr"], "clip_text": source["clip_text"]}
        if self.task == "em_fend":
            source = self._image(iid)
            return {
                **identity,
                "ocr": source["ocr"],
                "textual": deepcopy(self._text(tid)["entities"]),
                "visual": [
                    {"name": item["name"], "type": item["type"], "confidence": item["weight"]}
                    for item in source["visual_entities"]
                ],
            }
        if self.task == "eric_fnd":
            selected = self._row("methods/eric_fnd/entities.jsonl", "text_id", tid)["entities"]
            descriptions = []
            for item in selected[: self.max_entities]:
                entity = self._row("annotations/encyclopedia.jsonl", "entity_id", item["entity_id"])
                if entity["description"].strip():
                    value = f"{entity['label']}: {entity['description']}"
                    if hashlib.sha256(value.encode("utf-8")).hexdigest() != item["description_id"]:
                        raise PreparedInputError("ERIC description_id does not match its text.")
                    descriptions.append(value)
            return {
                **identity,
                "news_english": self._text(tid)["text_en"],
                "descriptions": descriptions,
            }
        if self.task == "nlin":
            image = self._image(iid)
            return {
                **identity,
                "news_english": self._text(tid)["text_en"],
                "caption": image["caption_en"],
                "dense": deepcopy(image["objects"]),
                "ocr_english": image["ocr_en"].splitlines(),
                "text_entities": deepcopy(
                    self._entities("annotations/entities/text_links.jsonl", "text_id", tid)
                ),
                "visual_entities": deepcopy(
                    self._entities("annotations/entities/image_links.jsonl", "image_id", iid)
                ),
            }
        if self.task == "event_radar":
            roles = self._row("annotations/event_roles.jsonl", "text_id", tid)
            return {
                **identity,
                "roles": deepcopy(roles["roles"]),
                "emotion": deepcopy(
                    self._row("annotations/emotions.jsonl", "text_id", tid)["values"]
                ),
                **self._objects(iid),
            }
        raise PreparedInputError("NSLM image contexts are read with contexts(paths).")

    def _objects(self, image_id):
        source = self._row("annotations/objects.jsonl", "image_id", image_id)
        return {
            "image_id": image_id,
            "image_size": [source["width"], source["height"]],
            "objects": deepcopy(source["objects"][: self.max_objects]),
        }

    def _validate(self, record, key):
        if not isinstance(record, dict) or not _basic(record):
            raise PreparedInputError("Hub resource records must contain finite basic values.")
        if key.startswith(("context:", "objects:")):
            if not _sha(record.get("image_id")) or key.split(":", 1)[1] != record["image_id"]:
                raise PreparedInputError("Hub image resource identity is invalid.")
            if key.startswith("context:"):
                _string(record.get("context"), "context", empty=True)
            else:
                self._validate_objects(record)
            return
        if not all(_sha(record.get(name)) for name in ("text_id", "image_id", "pair_id")):
            raise PreparedInputError("Hub resource content IDs must be SHA256 digests.")
        expected = hashlib.sha256(f"{record['text_id']}:{record['image_id']}".encode()).hexdigest()
        if record["pair_id"] != expected or key != expected:
            raise PreparedInputError("Hub resource pair identity is inconsistent.")
        if self.task == "aar":
            for name in ("genuine", "fake"):
                _string(record.get(name), name)
        elif self.task == "mmfn":
            _string(record.get("ocr"), "ocr", empty=True)
            _string(record.get("clip_text"), "clip_text")
        elif self.task == "mmrgv":
            from mmfact.models_classification._mmrgv_resources import (
                validate_rationales,
            )

            _string(record.get("caption"), "caption")
            validate_rationales(record.get("rationales"))
        elif self.task == "em_fend":
            _string(record.get("ocr"), "ocr", empty=True)
            if not isinstance(record.get("textual"), dict) or set(record["textual"]) != set(_TYPES):
                raise PreparedInputError("EM-FEND requires three typed textual entity lists.")
            for values in record["textual"].values():
                if not isinstance(values, list):
                    raise PreparedInputError("EM-FEND textual entities must be lists.")
                for name in values:
                    _string(name, "entity")
            if not isinstance(record.get("visual"), list):
                raise PreparedInputError("EM-FEND visual entities must be a list.")
            for item in record["visual"]:
                _string(item.get("name"), "visual entity")
                score = item.get("confidence")
                if (
                    item.get("type") not in _TYPES
                    or type(score) not in (int, float)
                    or not 0 <= score <= 1
                ):
                    raise PreparedInputError(
                        "EM-FEND requires actual finite entity weights in [0,1]."
                    )
        elif self.task in {"eric_fnd", "nlin"}:
            _string(record.get("news_english"), "English news")
            if self.task == "eric_fnd":
                if (
                    not isinstance(record.get("descriptions"), list)
                    or len(record["descriptions"]) > self.max_entities
                ):
                    raise PreparedInputError("ERIC description list exceeds its configured limit.")
                for description in record["descriptions"]:
                    _string(description, "entity description")
            else:
                _string(record.get("caption"), "English caption")
                if not all(
                    isinstance(record.get(field), list)
                    for field in ("dense", "ocr_english", "text_entities", "visual_entities")
                ):
                    raise PreparedInputError("NLIN requires dense, OCR and linked entity lists.")
                for field in ("text_entities", "visual_entities"):
                    for entity in record[field]:
                        _string(entity.get("label"), "entity label", empty=True)
                        _string(entity.get("description"), "entity description", empty=True)
        elif self.task == "event_radar":
            roles = record.get("roles")
            if not isinstance(roles, dict) or set(roles) != set(_ROLES):
                raise PreparedInputError("Event-Radar requires four role slots.")
            for span in roles.values():
                if span is not None and (
                    not isinstance(span, dict)
                    or not isinstance(span.get("text"), str)
                    or type(span.get("start")) is not int
                    or type(span.get("end")) is not int
                    or span["start"] < 0
                    or span["end"] - span["start"] != len(span["text"])
                ):
                    raise PreparedInputError(
                        "Event role spans must be Unicode half-open intervals."
                    )
            emotion = record.get("emotion")
            if (
                not isinstance(emotion, list)
                or len(emotion) != (54 if self.language == "en" else 55)
                or any(type(v) not in (int, float) for v in emotion)
            ):
                raise PreparedInputError(
                    "Event emotion vector has the wrong language-specific layout."
                )
            self._validate_objects(record)

    def _validate_objects(self, record):
        size, objects = record.get("image_size"), record.get("objects")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(v) is not int or v < 1 for v in size)
        ):
            raise PreparedInputError("Object resources require positive [width,height].")
        if not isinstance(objects, list) or len(objects) > self.max_objects:
            raise PreparedInputError("Object count exceeds the configured limit.")
        for item in objects:
            box = item.get("box")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= size[0]
                or not 0 <= box[1] < box[3] <= size[1]
            ):
                raise PreparedInputError("Object coordinates do not fit the original image size.")
            if (
                type(item.get("label")) is not int
                or item["label"] < 1
                or type(item.get("score")) not in (int, float)
                or not 0 <= item["score"] <= 1
            ):
                raise PreparedInputError("Object labels and scores are invalid.")

    def _remember(self, key, build):
        if key not in self.records:
            record = build()
            self._validate(record, key)
            self.records[key] = deepcopy(record)
        self.loaded = True
        return deepcopy(self.records[key])

    def lookup(self, text, path):
        identity = _identity(text, path)
        record = self._remember(identity["pair_id"], lambda: self._assemble(text, identity))
        if self.task == "event_radar" and any(
            span is not None and text[span["start"] : span["end"]] != span["text"]
            for span in record["roles"].values()
        ):
            raise PreparedInputError("Event role offsets do not match the original text.")
        return record

    def contexts(self, paths):
        if self.task != "nslm":
            raise PreparedInputError("Only NSLM resources expose image retrieval contexts.")
        result = []
        for path in paths:
            image_id = Path(path).stem
            if not _sha(image_id):
                raise PreparedInputError("Hub contexts require canonical image filenames.")
            record = self._remember(
                f"context:{image_id}",
                lambda: {
                    "image_id": image_id,
                    "context": self._row("methods/nslm/contexts.jsonl", "image_id", image_id)[
                        "context"
                    ],
                },
            )
            result.append(record["context"])
        return result

    def image_objects(self, path):
        if self.task != "event_radar":
            raise PreparedInputError("Only Event-Radar resources expose image objects.")
        image_id = Path(path).stem
        return self._remember(f"objects:{image_id}", lambda: self._objects(image_id))

    def get_extra_state(self):
        payload = {
            "protocol": deepcopy(self.protocol),
            "records": deepcopy(self.records),
            "loaded": self.loaded,
        }
        return {"version": 1, **payload, "sha256": fingerprint(payload)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or not _basic(state)
            or set(state) != {"version", "protocol", "records", "loaded", "sha256"}
            or state["version"] != 1
            or state["protocol"] != self.protocol
            or type(state["loaded"]) is not bool
            or not isinstance(state["records"], dict)
        ):
            raise PreparedInputError("Invalid clean Hub resource checkpoint.")
        payload = {key: state[key] for key in ("protocol", "records", "loaded")}
        if fingerprint(payload) != state["sha256"] or (not state["loaded"] and state["records"]):
            raise PreparedInputError("Hub resource checkpoint checksum does not match.")
        for key, record in state["records"].items():
            self._validate(record, key)
        self.records, self.loaded = deepcopy(state["records"]), state["loaded"]
        self._tables.clear()
