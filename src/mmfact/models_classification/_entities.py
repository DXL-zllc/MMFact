"""Content-bound ERIC entity descriptions; inference never contacts external services."""

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

from torch import nn


def resource_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def text_identity(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Entity resources require nonempty news text.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entity_protocol(language, max_entities=10):
    if (
        not isinstance(language, str)
        or language not in {"en", "zh"}
        or type(max_entities) is not int
        or max_entities < 1
    ):
        raise ValueError("Entity resources require en/zh and a positive integer max_entities.")
    return {
        "task": "eric-fnd",
        "language": language,
        "max_entities": max_entities,
        "text_input": "raw-utf8-no-cleaning",
        "extractor": (
            {"name": "jieba-textrank", "allow_pos": ["ns", "n", "nr"], "span": 5}
            if language == "zh"
            else {
                "name": "tagme",
                "endpoint": "https://tagme.d4science.org/tagme/tag",
                "lang": "en",
                "long_text": 3,
                "rho_strictly_above": 0.3,
            }
        ),
        "selection": "score-desc-start-end-title-unique-top-k",
        "offsets": "verified-unique-codepoint-or-utf16-to-codepoint",
        "wikipedia": {
            "endpoint": f"https://{language}.wikipedia.org/w/api.php",
            "formatversion": 2,
            "titles": "exact-with-normalization-and-redirects-no-search",
            "converttitles": language == "zh",
            "revision": "current-at-generation-then-parse-oldid",
            "section": 0,
            "paragraph": "first-nonempty-p-outside-tables-and-metadata-v1",
            "sentence": "punctuation-boundary-abbreviation-exceptions-v1",
            "conversion": "opencc-t2s" if language == "zh" else "identity",
            "empty": "keep-news-with-reason",
        },
    }


def first_sentence(paragraph):
    """Fixed boundary rule, with decimals, initials and common abbreviations protected."""
    text = " ".join(paragraph.split())
    abbreviations = {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "sr",
        "jr",
        "vs",
        "e.g",
        "i.e",
        "etc",
        "no",
        "fig",
    }
    for match in re.finditer(r"[.!?。！？]", text):
        index, mark = match.start(), match.group()
        if mark == ".":
            if (
                index + 1 < len(text)
                and not text[index + 1].isspace()
                and text[index + 1] not in "\"'”’）)"
            ):
                continue
            token = re.search(r"([\w.]+)$", text[:index])
            token = token.group(1) if token else ""
            if token.lower() in abbreviations or re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", token):
                continue
        end = index + 1
        while end < len(text) and text[end] in "\"'”’）)":
            end += 1
        return text[:end]
    return text


def entity_record(text, candidates):
    record = {
        "text_sha256": text_identity(text),
        "candidates": deepcopy(candidates),
        "empty_reason": (
            "no_candidates"
            if not candidates
            else "no_descriptions"
            if not any(item["wikipedia"]["description"] for item in candidates)
            else None
        ),
    }
    return {**record, "record_sha256": resource_digest(record)}


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _positive(value):
    return type(value) is int and value > 0


def _json(line):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Entity resource JSON contains duplicate keys.")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Entity resource JSON contains non-finite {value}.")

    return json.loads(line, object_pairs_hook=unique, parse_constant=reject)


class EntityDescriptionStore(nn.Module):
    """Atomically validate fixed manifests and embed their evidence in safe checkpoints."""

    def __init__(self, language, manifest, max_entities=10):
        super().__init__()
        self.protocol = entity_protocol(language, max_entities)
        if not isinstance(manifest, (str, Path)) or (
            isinstance(manifest, str) and not manifest.strip()
        ):
            raise ValueError("Entity resources require a nonempty manifest path.")
        self.manifest = Path(manifest).expanduser()
        self.max_entities = max_entities
        self.records, self.tools, self.loaded = {}, None, False

    def validate_tools(self, tools):
        if not isinstance(tools, dict) or set(tools) != {"extractor", "converter"}:
            raise ValueError("Entity resources have invalid tool identities.")
        expected = (
            (("jieba-textrank", "0.42.1"), ("opencc-t2s", "0.1.7"))
            if self.protocol["language"] == "zh"
            else (("tagme", "unversioned-remote-service"), ("identity", "1"))
        )
        for key, (name, version) in zip(("extractor", "converter"), expected, strict=True):
            identity = tools[key]
            if (
                not isinstance(identity, dict)
                or set(identity) != {"name", "version", "resources_sha256"}
                or identity["name"] != name
                or identity["version"] != version
                or (
                    not _sha(identity["resources_sha256"])
                    if self.protocol["language"] == "zh"
                    else identity["resources_sha256"] is not None
                )
            ):
                raise ValueError(
                    "Entity resources have incompatible extractor/converter identities."
                )
        return deepcopy(tools)

    def header(self, tools):
        tools = self.validate_tools(tools)
        return {
            "schema": 1,
            "generator": deepcopy(self.protocol),
            "generator_sha256": resource_digest(self.protocol),
            "tools": tools,
            "tools_sha256": resource_digest(tools),
        }

    def validate_record(self, record):
        if (
            not isinstance(record, dict)
            or set(record) != {"text_sha256", "candidates", "empty_reason", "record_sha256"}
            or not _sha(record["text_sha256"])
            or not isinstance(record["candidates"], list)
            or len(record["candidates"]) > self.protocol["max_entities"]
        ):
            raise ValueError("Entity resource record has invalid fields or text identity.")
        titles = set()
        for candidate in record["candidates"]:
            if (
                not isinstance(candidate, dict)
                or set(candidate) != {"mention", "title", "score", "start", "end", "wikipedia"}
                or not all(_nonempty(candidate[key]) for key in ("mention", "title"))
                or type(candidate["score"]) not in (int, float)
                or not math.isfinite(candidate["score"])
                or candidate["score"] < 0
                or type(candidate["start"]) is not int
                or type(candidate["end"]) is not int
                or not 0 <= candidate["start"] < candidate["end"]
                or candidate["title"] in titles
            ):
                raise ValueError("Entity resource candidate is invalid or duplicated.")
            titles.add(candidate["title"])
            if self.protocol["language"] == "en" and candidate["score"] <= 0.3:
                raise ValueError("TagMe entity confidence must be strictly above 0.3.")
            self._validate_wikipedia(candidate["wikipedia"], candidate["title"])
        expected_reason = (
            "no_candidates"
            if not record["candidates"]
            else "no_descriptions"
            if not any(item["wikipedia"]["description"] for item in record["candidates"])
            else None
        )
        content = {key: value for key, value in record.items() if key != "record_sha256"}
        if record["empty_reason"] != expected_reason or record["record_sha256"] != resource_digest(
            content
        ):
            raise ValueError("Entity resource record disagrees with its reason or digest.")
        return deepcopy(record)

    def _validate_wikipedia(self, page, title):
        fields = {
            "status",
            "requested_title",
            "resolved_title",
            "page_id",
            "revision_id",
            "revision_timestamp",
            "url",
            "normalizations",
            "redirects",
            "lead_html_sha256",
            "lead_paragraph",
            "description",
            "reason",
        }
        if (
            not isinstance(page, dict)
            or set(page) != fields
            or page["requested_title"] != title
            or page["status"]
            not in {"available", "missing", "invalid", "non_article", "disambiguation", "no_lead"}
            or not isinstance(page["lead_paragraph"], str)
            or not isinstance(page["description"], str)
            or not _nonempty(page["resolved_title"])
        ):
            raise ValueError("Entity resource Wikipedia evidence is invalid.")
        for key in ("normalizations", "redirects"):
            if not isinstance(page[key], list) or any(
                not isinstance(item, dict)
                or not {"from", "to"} <= item.keys()
                or set(item) - {"from", "to", "tofragment"}
                or not all(_nonempty(value) for value in item.values())
                for item in page[key]
            ):
                raise ValueError("Entity resource title mappings are invalid.")
        absent = page["status"] in {"missing", "invalid"}
        if absent:
            if any(
                page[key] is not None
                for key in (
                    "page_id",
                    "revision_id",
                    "revision_timestamp",
                    "url",
                    "lead_html_sha256",
                )
            ):
                raise ValueError("Missing entities cannot claim a Wikipedia revision.")
        else:
            language = self.protocol["language"]
            if (
                not _positive(page["page_id"])
                or not _positive(page["revision_id"])
                or not _nonempty(page["revision_timestamp"])
                or page["url"]
                != f"https://{language}.wikipedia.org/w/index.php?oldid={page['revision_id']}"
            ):
                raise ValueError("Entity resource Wikipedia revision is inconsistent.")
        if page["status"] in {"available", "no_lead"}:
            if not _sha(page["lead_html_sha256"]):
                raise ValueError("Parsed Wikipedia leads require their HTML digest.")
        elif page["lead_html_sha256"] is not None:
            raise ValueError("Unparsed Wikipedia pages cannot claim a lead digest.")
        if page["status"] == "available":
            if (
                not _nonempty(page["description"])
                or page["description"] != first_sentence(page["lead_paragraph"])
                or not _nonempty(page["lead_paragraph"])
                or page["reason"] is not None
            ):
                raise ValueError("Available entities require a lead description.")
        elif page["description"] or page["lead_paragraph"] or page["reason"] != page["status"]:
            raise ValueError("Empty entity descriptions require an explicit matching reason.")

    def load(self):
        parsed = {}
        with self.manifest.open(encoding="utf-8") as stream:
            header = _json(next(stream, "null"))
            tools = self.validate_tools(header.get("tools") if isinstance(header, dict) else None)
            if header != self.header(tools) or (self.tools is not None and tools != self.tools):
                raise ValueError("Entity manifest has an incompatible protocol or tool identity.")
            for line in stream:
                if not line.strip():
                    continue
                record = self.validate_record(_json(line))
                key = record["text_sha256"]
                if key in parsed or (key in self.records and self.records[key] != record):
                    raise ValueError(
                        "Entity manifest contains duplicate or conflicting text resources."
                    )
                parsed[key] = record
        self.records.update(parsed)
        self.tools, self.loaded = tools, True
        return self

    def lookup(self, text):
        key = text_identity(text)
        if not self.loaded:
            self.load()
        if key not in self.records and self.manifest.is_file():
            self.load()
        if key not in self.records:
            raise ValueError(
                "No entity resource matches this text; download the required materials with --download-materials."
            )
        if any(
            not 0 <= item["start"] < item["end"] <= len(text)
            or text[item["start"] : item["end"]] != item["mention"]
            for item in self.records[key]["candidates"]
        ):
            raise ValueError("Entity resource mention offsets do not match the original text.")
        return [
            item["wikipedia"]["description"]
            for item in self.records[key]["candidates"]
            if item["wikipedia"]["description"]
        ]

    def get_extra_state(self):
        return {
            "format_version": 1,
            "protocol": deepcopy(self.protocol),
            "tools": deepcopy(self.tools),
            "records": deepcopy(self.records),
            "records_sha256": resource_digest(self.records),
            "loaded": self.loaded,
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {"format_version", "protocol", "tools", "records", "records_sha256", "loaded"}
            or state["format_version"] != 1
            or state["protocol"] != self.protocol
            or not isinstance(state["records"], dict)
            or type(state["loaded"]) is not bool
        ):
            raise ValueError("Entity checkpoint has an incompatible protocol.")
        tools = self.validate_tools(state["tools"]) if state["loaded"] else None
        records = {}
        for key, value in state["records"].items():
            record = self.validate_record(value)
            if key != record["text_sha256"]:
                raise ValueError("Entity checkpoint has inconsistent text identities.")
            records[key] = record
        if (
            state["records_sha256"] != resource_digest(records)
            or (not state["loaded"] and (records or state["tools"] is not None))
            or (self.tools is not None and tools != self.tools)
        ):
            raise ValueError(
                "Entity checkpoint disagrees with its resource digest or tool identity."
            )
        self.records, self.tools, self.loaded = records, tools, state["loaded"]
