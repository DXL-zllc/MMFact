"""ERIC's English news/entity view of existing, content-bound NLIN snapshots."""

import hashlib
import math
import re
from copy import deepcopy
from pathlib import Path

from torch import nn

from mmfact.models_classification._entities import _json, resource_digest, text_identity
from mmfact.models_classification._generated import pair_identity


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _tools(tools, language):
    names = {"translation", "linker", "knowledge"}
    if not isinstance(tools, dict) or not names <= tools.keys():
        raise ValueError("ERIC NLIN resources require translation, linker and knowledge tools.")
    selected = {name: deepcopy(tools[name]) for name in sorted(names)}
    for name, identity in selected.items():
        if (
            not isinstance(identity, dict)
            or not {"name", "revision", "settings"} <= identity.keys()
            or set(identity) - {"name", "revision", "settings", "weights"}
            or not _text(identity["revision"])
            or not isinstance(identity["settings"], dict)
        ):
            raise ValueError("ERIC NLIN tools require explicit names, revisions and settings.")
    linker, knowledge, translation = (
        selected[key] for key in ("linker", "knowledge", "translation")
    )
    if linker["name"] != "TagMe" or linker["settings"] != {
        "language": "en",
        "rho_threshold": 0.3,
        "long_text": 3,
    }:
        raise ValueError("ERIC needs NLIN's English TagMe rho>0.3/long_text=3 protocol.")
    if knowledge["name"] != "Wikidata" or knowledge["settings"] != {
        "language": "en",
        "wikipedia_language": "en",
    }:
        raise ValueError("ERIC needs English Wikidata descriptions resolved from English titles.")
    allowed = {"identity"} if language == "en" else {"MarianMT", "OpenAIChatCompletions"}
    if translation["name"] not in allowed:
        raise ValueError("ERIC NLIN translation tool does not match the source language.")
    if language == "en" and (translation["revision"] != "1" or translation["settings"]):
        raise ValueError("English NLIN news must use the identity translation protocol.")
    if translation["name"] == "OpenAIChatCompletions":
        settings = translation["settings"]
        if (
            set(settings)
            != {
                "base_url",
                "prompt",
                "temperature",
                "top_p",
                "seed",
                "max_tokens",
                "chunk_characters",
            }
            or not _text(settings["base_url"])
            or not settings["base_url"].startswith(
                ("https://", "http://localhost:", "http://127.0.0.1:")
            )
            or not _text(settings["prompt"])
            or type(settings["seed"]) is not int
            or any(
                type(settings[key]) is not int or settings[key] < 1
                for key in ("max_tokens", "chunk_characters")
            )
            or type(settings["temperature"]) not in (int, float)
            or not 0 <= settings["temperature"] <= 2
            or type(settings["top_p"]) not in (int, float)
            or not 0 < settings["top_p"] <= 1
        ):
            raise ValueError("ERIC needs the complete pinned NLIN API translation settings.")
    resource_digest(selected)
    return selected


def _source_tools(protocol, language):
    if not isinstance(protocol, dict):
        raise ValueError("ERIC expected an NLIN resource protocol.")
    expected = {
        "task": "nlin-resource-adapted",
        "version": 1,
        "language": language,
        "tools": protocol.get("tools"),
        "linking_inputs": ["news_english", "caption-plus-ocr_english-no-dense"],
        "knowledge": "entity-label-colon-English-Wikidata-description-at-recorded-revision",
        "prediction_inputs": "raw-text-and-image-content-sha256-no-labels",
    }
    if protocol != expected:
        raise ValueError("ERIC NLIN resource protocol mismatch.")
    return _tools(protocol["tools"], language)


def _offsets(text, item):
    start, end, mention = (item.get(key) for key in ("start", "end", "spot"))
    if type(start) is not int or type(end) is not int or not 0 <= start < end or not _text(mention):
        raise ValueError("ERIC TagMe annotation needs valid mention offsets.")
    positions = set()
    if end <= len(text) and text[start:end] == mention:
        positions.add((start, end))
    encoded = text.encode("utf-16-le")
    if end * 2 <= len(encoded):
        try:
            prefix = encoded[: start * 2].decode("utf-16-le")
            spot = encoded[start * 2 : end * 2].decode("utf-16-le")
            if spot == mention:
                positions.add((len(prefix), len(prefix) + len(spot)))
        except UnicodeDecodeError:
            pass
    if len(positions) != 1:
        raise ValueError("ERIC TagMe offsets are ambiguous or do not match the English news.")
    return positions.pop()


def _candidates(news, raw):
    tagme = raw.get("tagme") if isinstance(raw, dict) else None
    if (
        not isinstance(tagme, dict)
        or "error" in tagme
        or not isinstance(tagme.get("annotations"), list)
        or tagme.get("lang", "en") != "en"
    ):
        raise ValueError("ERIC requires the actual English TagMe annotation response.")
    candidates, titles = [], []
    for item in tagme["annotations"]:
        if not isinstance(item, dict) or not _text(item.get("title")):
            raise ValueError("ERIC TagMe annotation is missing its title.")
        score = item.get("rho")
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("ERIC TagMe confidence must be finite and in [0,1].")
        if score <= 0.3:
            continue
        start, end = _offsets(news, item)
        candidates.append(
            {
                "title": item["title"],
                "score": score,
                "start": start,
                "end": end,
                "mention": item["spot"],
            }
        )
        if item["title"] not in titles:
            titles.append(item["title"])
    return candidates, titles


def _entities(entities, raw, titles):
    if not isinstance(entities, list) or not isinstance(raw.get("entities"), list):
        raise ValueError("ERIC needs NLIN textual entities and their real API responses.")
    if len(entities) != len(titles) or len(raw["entities"]) != len(titles):
        raise ValueError("ERIC entity count disagrees with the accepted TagMe titles.")
    for title, item, response in zip(titles, entities, raw["entities"], strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"title", "qid", "label", "description", "revision", "reason"}
            or item["title"] != title
            or not isinstance(response, dict)
            or set(response) != {"title", "wikipedia", "wikidata"}
            or response["title"] != title
        ):
            raise ValueError("ERIC entity evidence does not match the TagMe title order.")
        wiki = response["wikipedia"]
        pages = wiki.get("query", {}).get("pages") if isinstance(wiki, dict) else None
        if (
            not isinstance(pages, list)
            or len(pages) != 1
            or not isinstance(pages[0], dict)
            or "error" in wiki
        ):
            raise ValueError("ERIC entity needs its real Wikipedia title-to-QID response.")
        qid = pages[0].get("pageprops", {}).get("wikibase_item")
        expected = {
            "title": title,
            "qid": qid,
            "label": "",
            "description": "",
            "revision": None,
            "reason": "no_qid",
        }
        wikidata = response["wikidata"]
        if qid is None:
            if wikidata is not None:
                raise ValueError("ERIC unlinked entity cannot claim a Wikidata response.")
        else:
            if (
                not isinstance(qid, str)
                or not re.fullmatch(r"Q[1-9][0-9]*", qid)
                or not isinstance(wikidata, dict)
                or "error" in wikidata
            ):
                raise ValueError("ERIC Wikidata entity requires a valid QID and response.")
            value = wikidata.get("entities", {}).get(qid)
            if (
                not isinstance(value, dict)
                or "missing" in value
                or type(value.get("lastrevid")) is not int
                or value["lastrevid"] < 1
            ):
                raise ValueError("ERIC Wikidata entity requires its actual positive lastrevid.")
            label = value.get("labels", {}).get("en", {}).get("value", title)
            description = value.get("descriptions", {}).get("en", {}).get("value", "")
            if not _text(label) or not isinstance(description, str):
                raise ValueError("ERIC Wikidata label/description must be English text fields.")
            expected.update(
                label=label,
                description=description,
                revision=value["lastrevid"],
                reason=None if description.strip() else "no_description",
            )
        if item != expected:
            raise ValueError("ERIC entity differs from its saved Wikidata response.")


def _validate_projection(record, language):
    required = {
        "pair_sha256",
        "text_sha256",
        "image_sha256",
        "news_english",
        "text_entities",
        "ocr",
        "ocr_english",
        "empty_reasons",
        "audit",
        "source_record_sha256",
        "record_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Invalid ERIC textual resource projection.")
    if any(
        not _sha(record[key])
        for key in ("pair_sha256", "text_sha256", "image_sha256", "source_record_sha256")
    ):
        raise ValueError("ERIC source identities must be SHA256 digests.")
    pair = hashlib.sha256(
        (record["text_sha256"] + record["image_sha256"]).encode("ascii")
    ).hexdigest()
    if record["pair_sha256"] != pair:
        raise ValueError("ERIC source pair identity is inconsistent.")
    if (
        not _text(record["news_english"])
        or not isinstance(record["ocr"], list)
        or not isinstance(record["ocr_english"], list)
        or len(record["ocr"]) != len(record["ocr_english"])
        or any(not _text(item) for key in ("ocr", "ocr_english") for item in record[key])
    ):
        raise ValueError("ERIC NLIN translation inputs/results are invalid.")
    if language == "en" and (
        text_identity(record["news_english"]) != record["text_sha256"]
        or record["ocr"] != record["ocr_english"]
    ):
        raise ValueError("ERIC English NLIN resources must preserve the original news/OCR text.")
    audit = record["audit"]
    if not isinstance(audit, dict) or set(audit) != {"translation", "text_knowledge"}:
        raise ValueError("ERIC needs only translation and textual knowledge audit stages.")
    expected = {
        "translation": (
            resource_digest({"news_sha256": record["text_sha256"], "ocr": record["ocr"]}),
            {"news_english": record["news_english"], "ocr_english": record["ocr_english"]},
        ),
        "text_knowledge": (text_identity(record["news_english"]), record["text_entities"]),
    }
    for name, (identity, result) in expected.items():
        response = audit[name]
        if (
            not isinstance(response, dict)
            or set(response) != {"input_sha256", "result", "raw", "completed_at"}
            or response["input_sha256"] != identity
            or response["result"] != result
            or not _text(response["completed_at"])
            or not isinstance(response["raw"], dict)
        ):
            raise ValueError("ERIC NLIN audit differs from its inputs or derived resource.")
    raw = audit["text_knowledge"]["raw"]
    candidates, titles = _candidates(record["news_english"], raw)
    _entities(record["text_entities"], raw, titles)
    reason = (
        None
        if any(item["description"] for item in record["text_entities"])
        else "no_wikidata_description"
        if titles
        else "no_linked_entities"
    )
    if record["empty_reasons"] != {"text": reason}:
        raise ValueError("ERIC NLIN empty reason differs from the textual entities.")
    if record["record_sha256"] != resource_digest(
        {key: value for key, value in record.items() if key != "record_sha256"}
    ):
        raise ValueError("ERIC textual projection digest mismatch.")
    return candidates


class ERICNLINResources(nn.Module):
    """Read NLIN text evidence without requiring its visual generation stages or services."""

    def __init__(self, language, manifest, max_entities=10, tools=None):
        super().__init__()
        if language not in {"en", "zh"} or type(max_entities) is not int or max_entities < 1:
            raise ValueError("ERIC NLIN resources require en/zh and a positive max_entities.")
        if not isinstance(manifest, (str, Path)) or not str(manifest).strip():
            raise ValueError("ERIC NLIN resources need a nonempty manifest path.")
        self.manifest, self.max_entities = Path(manifest).expanduser(), max_entities
        self.protocol = {
            "task": "eric-nlin-english-wikidata",
            "version": 1,
            "language": language,
            "max_entities": max_entities,
            "selection": "score-desc-start-end-title-unique-qid-or-title-top-k-before-description-filter",
            "description": "English-label-colon-English-Wikidata-description",
            "input": "raw-text-and-image-content-sha256-no-labels",
        }
        self.expected_tools = None if tools is None else _tools(tools, language)
        self.tools, self.records, self.loaded = None, {}, False

    def load(self):
        with self.manifest.open(encoding="utf-8") as stream:
            header = _json(next(stream, "null"))
            if (
                not isinstance(header, dict)
                or set(header) != {"schema", "protocol"}
                or header["schema"] != 1
            ):
                raise ValueError("ERIC expects an existing NLIN schema-1 manifest.")
            tools = _source_tools(header["protocol"], self.protocol["language"])
            if (self.expected_tools is not None and tools != self.expected_tools) or (
                self.tools is not None and tools != self.tools
            ):
                raise ValueError("ERIC NLIN source tool identity mismatch.")
            records = {}
            for line in stream:
                if not line.strip():
                    continue
                source = _json(line)
                if not isinstance(source, dict) or source.get("record_sha256") != resource_digest(
                    {key: value for key, value in source.items() if key != "record_sha256"}
                ):
                    raise ValueError("ERIC NLIN source record digest mismatch.")
                fields = {
                    "pair_sha256",
                    "text_sha256",
                    "image_sha256",
                    "news_english",
                    "text_entities",
                    "ocr",
                    "ocr_english",
                    "empty_reasons",
                    "audit",
                }
                if (
                    not fields <= source.keys()
                    or not isinstance(source["audit"], dict)
                    or not {"translation", "text_knowledge"} <= source["audit"].keys()
                    or not isinstance(source["empty_reasons"], dict)
                    or "text" not in source["empty_reasons"]
                ):
                    raise ValueError("ERIC NLIN source lacks its textual evidence.")
                record = {key: deepcopy(source[key]) for key in fields}
                record["audit"] = {
                    key: record["audit"][key] for key in ("translation", "text_knowledge")
                }
                record["empty_reasons"] = {"text": record["empty_reasons"]["text"]}
                record["source_record_sha256"] = source["record_sha256"]
                record["record_sha256"] = resource_digest(record)
                _validate_projection(record, self.protocol["language"])
                key = record["pair_sha256"]
                if key in records or (key in self.records and self.records[key] != record):
                    raise ValueError("ERIC NLIN duplicate or conflicting resource pair.")
                records[key] = record
        self.records.update(records)
        self.tools, self.loaded = tools, True
        return self

    def lookup(self, text, path):
        if not self.loaded:
            self.load()
        identity = pair_identity(text, path)
        key = identity["pair_sha256"]
        if key not in self.records and self.manifest.is_file():
            self.load()
        if key not in self.records:
            raise ValueError("Missing ERIC NLIN resource for this original news/image pair.")
        record = self.records[key]
        candidates, _ = _candidates(
            record["news_english"], record["audit"]["text_knowledge"]["raw"]
        )
        entities = {item["title"]: item for item in record["text_entities"]}
        candidates.sort(
            key=lambda item: (-item["score"], item["start"], item["end"], item["title"])
        )
        selected = {}
        for candidate in candidates:
            item = entities[candidate["title"]]
            entity_key = ("qid", item["qid"]) if item["qid"] else ("title", item["title"])
            selected.setdefault(entity_key, {**item, **candidate})
        values = list(selected.values())[: self.max_entities]
        descriptions = [
            f"{item['label']}: {item['description']}"
            for item in values
            if item["description"].strip()
        ]
        return {
            **identity,
            "news_english": record["news_english"],
            "descriptions": descriptions,
            "selected_entities": deepcopy(values),
            "source_record_sha256": record["source_record_sha256"],
            "empty_reason": None
            if descriptions
            else "no_descriptions"
            if values
            else "no_candidates",
        }

    def get_extra_state(self):
        payload = {
            "protocol": deepcopy(self.protocol),
            "tools": deepcopy(self.tools),
            "records": deepcopy(self.records),
            "loaded": self.loaded,
        }
        return {"version": 1, **payload, "sha256": resource_digest(payload)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "protocol", "tools", "records", "loaded", "sha256"}
            or state["version"] != 1
            or state["protocol"] != self.protocol
            or type(state["loaded"]) is not bool
            or not isinstance(state["records"], dict)
            or (not state["loaded"] and (state["records"] or state["tools"] is not None))
        ):
            raise ValueError("Invalid ERIC NLIN resource checkpoint.")
        payload = {key: state[key] for key in ("protocol", "tools", "records", "loaded")}
        if resource_digest(payload) != state["sha256"]:
            raise ValueError("ERIC NLIN checkpoint digest mismatch.")
        tools = _tools(state["tools"], self.protocol["language"]) if state["loaded"] else None
        if self.expected_tools is not None and tools is not None and tools != self.expected_tools:
            raise ValueError("ERIC NLIN checkpoint tool identity mismatch.")
        for key, record in state["records"].items():
            _validate_projection(record, self.protocol["language"])
            if key != record["pair_sha256"]:
                raise ValueError("ERIC NLIN checkpoint resource key mismatch.")
        self.tools, self.records, self.loaded = tools, deepcopy(state["records"]), state["loaded"]
