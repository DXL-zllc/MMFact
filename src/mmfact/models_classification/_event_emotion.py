"""Publisher-only Dual Emotion blocks with explicit local resource provenance."""

import csv
import hashlib
import io
import json
import math
import re
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from mmfact.data.features import fingerprint

_EN_CATEGORIES = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
_ZH_CATEGORIES = (
    "angry",
    "disgusting",
    "fearful",
    "happy",
    "neutral",
    "optimistic",
    "pessimistic",
    "sad",
)
_NRC_CATEGORIES = tuple(sorted((*_EN_CATEGORIES, "negative", "positive")))
_DALIAN_CATEGORIES = (
    "NA",
    "NB",
    "NC",
    "ND",
    "NE",
    "NG",
    "NH",
    "NI",
    "NJ",
    "NK",
    "NL",
    "NN",
    "PA",
    "PB",
    "PC",
    "PD",
    "PE",
    "PF",
    "PG",
    "PH",
    "PK",
)
_EMOTICON_CATEGORIES = ("anger", "happiness", "none", "sadness", "surprise")
_SMILES = (":-)", ":)", ":o)", ":],", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", ":っ)")
_FROWNS = (">:[", ":-(", ":(", ":-c", ":c", ":-<", ":っC", ":<", ":-[", ":[", ":{")
_EMOJI_RANGES = ((0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF), (0x1F1E0, 0x1F1FF))
_URL_BODY = r"(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
_EN_TOKENS = r"(?:[A-Z]\.)+|\d+(?:\.\d+)?%?|\w+(?:[-']\w+)*|\.\.\.|(?:[.,;\"'?():-_`!])"
_COMMON_FILES = {
    "negation",
    "degree",
    "positive",
    "negative",
    "pronoun_first",
    "pronoun_second",
    "pronoun_third",
}


def emotion_categories(language):
    if language not in {"en", "zh"}:
        raise ValueError("Publisher emotion language must be en or zh.")
    return _EN_CATEGORIES if language == "en" else _ZH_CATEGORIES


def emotion_layout(language):
    """Exact feature order, including source-specific Dalian29 and auxiliary statistics."""
    categories = emotion_categories(language)
    symbols = tuple(
        f"punctuation.{name}" for name in ("exclamation", "question", "comma", "period", "ellipsis")
    )
    words = tuple(f"words.{name}" for name in ("positive", "negative", "degree", "negation"))
    pronouns = tuple(f"pronouns.{name}" for name in ("first", "second", "third"))
    if language == "en":
        return (
            *(f"category.label.{name}" for name in categories),
            *(f"category.intensity.{name}" for name in categories),
            *(f"nrc.association.{name}" for name in _NRC_CATEGORIES),
            *(f"nrc.intensity.{name}" for name in categories),
            *(f"vader.{name}" for name in ("pos", "neg", "neu", "compound")),
            "emoticon.smile",
            "emoticon.frown",
            "emoticon.emoji",
            *symbols,
            *words,
            *pronouns,
            "uppercase",
        )
    return (
        *(f"category.intensity.{name}" for name in categories),
        *(f"dalian.count.{name}" for name in _DALIAN_CATEGORIES),
        *(f"dalian.intensity.{index}" for index in range(4)),
        *(f"dalian.polarity.{index}" for index in range(4)),
        "boson",
        *(f"emoticon.{name}" for name in _EMOTICON_CATEGORIES),
        *symbols,
        *words,
        *pronouns,
    )


def _resource_names(language):
    return _COMMON_FILES | (
        {"nrc_association", "nrc_intensity", "vader_lexicon"}
        if language == "en"
        else {"dalian", "boson", "emoticons", "jieba_dictionary"}
    )


def _provenance(language, resources, tokenization):
    result = {
        "schema": 1,
        "recipe": "event-radar-dualemotion-publisher-adapted-corrected-v1",
        "language": language,
        "feature_names": list(emotion_layout(language)),
        "resources": resources,
        "tokenization": tokenization,
        "category": {
            "model": "gpt-4o-2024-08-06",
            "temperature": 0,
            "seed": 42,
            "order": list(emotion_categories(language)),
            "meaning": "independent-publisher-emotion-intensities-in-[0,1]",
            "english_binary_threshold": 0.5,
        },
        "context": {
            "left_window": 4 if language == "en" else 2,
            "matching": "substring-on-space-joined-tokens",
            "reduce": "signed-degree-weighted-sum",
        },
        "english_emoticons": {
            "smiles": list(_SMILES),
            "frowns": list(_FROWNS),
            "emoji_ranges": [list(pair) for pair in _EMOJI_RANGES],
        },
    }
    result["sha256"] = fingerprint(result)
    return result


def validate_emotion_provenance(provenance, language):
    """Check a fixed-feature recipe without reading dictionaries or importing tokenizers."""
    emotion_categories(language)

    def basic(value):
        if type(value) in (str, int, bool) or value is None:
            return True
        if type(value) is float:
            return math.isfinite(value)
        if type(value) is list:
            return all(basic(item) for item in value)
        return type(value) is dict and all(
            type(key) is str and basic(item) for key, item in value.items()
        )

    if (
        not isinstance(provenance, dict)
        or not basic(provenance)
        or type(provenance.get("schema")) is not int
    ):
        raise ValueError("Invalid publisher emotion provenance schema.")
    resources, tokenization = provenance.get("resources"), provenance.get("tokenization")
    if not isinstance(resources, dict) or set(resources) != _resource_names(language):
        raise ValueError("Publisher emotion provenance has an incomplete resource layout.")
    for name, descriptors in resources.items():
        if name in {"positive", "negative"}:
            if not isinstance(descriptors, list) or not descriptors:
                raise ValueError("Publisher emotion word-list provenance is incomplete.")
        else:
            descriptors = [descriptors]
        for descriptor in descriptors:
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"sha256", "source", "license"}
                or any(type(value) is not str or not value.strip() for value in descriptor.values())
                or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
            ):
                raise ValueError("Publisher emotion resource identity is invalid.")
    if not isinstance(tokenization, dict):
        raise ValueError("Publisher emotion tokenization provenance is missing.")
    if language == "en":
        nltk_version = tokenization.get("nltk")
        match = (
            re.fullmatch(r"3\.(\d+)(?:\.[0-9a-z]+)*(?:[.+-][0-9a-z]+)*", nltk_version)
            if isinstance(nltk_version, str)
            else None
        )
        expected_tokens = {
            "algorithm": "case-preserving-source-regexp-url-mention-removal",
            "nltk": nltk_version,
        }
        if match is None or int(match[1]) < 9 or tokenization != expected_tokens:
            raise ValueError("Publisher emotion requires recorded NLTK >=3.9,<4 tokenization.")
    elif (
        tokenization
        != {"algorithm": "jieba", "version": "0.42.1", "HMM": True, "keep_whitespace_tokens": True}
        or type(tokenization.get("HMM")) is not bool
        or type(tokenization.get("keep_whitespace_tokens")) is not bool
    ):
        raise ValueError("Publisher emotion requires jieba0.42.1 HMM tokenization.")
    try:
        expected = _provenance(language, resources, tokenization)
        # Fingerprint comparison also distinguishes booleans from numeric constants.
        if fingerprint(provenance) != fingerprint(expected):
            raise ValueError("Publisher emotion recipe, language, layout or checksum differs.")
    except (TypeError, OverflowError) as error:
        raise ValueError("Publisher emotion provenance must contain basic finite data.") from error
    return deepcopy(provenance)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate emotion resource JSON key: {key}.")
        result[key] = value
    return result


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be finite numeric data.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data.")
    return result


def _word_list(text, name):
    words = [line.strip() for line in text.splitlines() if line.strip()]
    if not words:
        raise ValueError(f"Empty emotion resource: {name}.")
    return words


def _scores(text, name, *, phrases=True):
    values = {}
    for line in _word_list(text, name):
        pieces = line.rsplit(maxsplit=1) if phrases else line.split()
        if len(pieces) != 2:
            raise ValueError(f"Malformed {name} word/score row.")
        word, value = pieces[0], _number(pieces[1], name)
        if word in values and values[word] != value:
            raise ValueError(f"Conflicting {name} entry: {word}.")
        values[word] = value
    return values


def _nrc(text, categories, intensity=False):
    values, seen, entries = {}, set(), {}
    for row in csv.reader(io.StringIO(text), delimiter="\t"):
        if not row or not any(row):
            continue
        if row == ["word", "emotion", "emotion-intensity-score"] and intensity:
            continue
        if len(row) != 3 or not row[0] or row[1] not in categories:
            raise ValueError("Malformed NRC word/category/value row.")
        word, category = row[:2]
        value = _number(row[2], "NRC score")
        if not 0 <= value <= 1 or (not intensity and value not in {0, 1}):
            raise ValueError("NRC scores must be [0,1]; association must be binary.")
        if (word, category) in entries and entries[word, category] != value:
            raise ValueError("Conflicting NRC entry.")
        entries[word, category] = value
        values.setdefault(word, [0.0] * len(categories))[categories.index(category)] = value
        seen.add(category)
    if seen != set(categories):
        raise ValueError("NRC resource must cover its complete category layout.")
    return values


def _dalian(text):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["word", "emotion", "intensity", "polarity"]:
        raise ValueError("Dalian CSV requires word,emotion,intensity,polarity columns.")
    values, counts, seen = {}, {}, set()
    for row in reader:
        word, category = row["word"], row["emotion"]
        if not word or category not in _DALIAN_CATEGORIES or None in row.values() or None in row:
            raise ValueError("Malformed Dalian row/category.")
        offset = counts.get(word, 0)
        if offset == 4:
            raise ValueError("Dalian resource has more than four entries for one word.")
        vector = values.setdefault(word, [0.0] * 29)
        vector[_DALIAN_CATEGORIES.index(category)] += 1
        vector[21 + offset] = _number(row["intensity"], "Dalian intensity")
        vector[25 + offset] = _number(row["polarity"], "Dalian polarity")
        counts[word] = offset + 1
        seen.add(category)
    if seen != set(_DALIAN_CATEGORIES):
        raise ValueError("Dalian resource must cover all 21 category codes.")
    return values


def _emoticons(text):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["emoticon", "label"]:
        raise ValueError("Emoticon CSV requires emoticon,label columns.")
    result = {}
    for row in reader:
        word, label = row["emoticon"], row["label"]
        if not word or label not in _EMOTICON_CATEGORIES or None in row.values() or None in row:
            raise ValueError("Malformed Chinese emoticon/category row.")
        if word in result:
            raise ValueError("Duplicate Chinese emoticon entry.")
        result[word] = label
    if set(result.values()) != set(_EMOTICON_CATEGORIES):
        raise ValueError("Emoticon resource must cover all five categories.")
    return result


class PublisherEmotion:
    """Generate EN54/ZH55 from original text and exact eight GPT category intensities.

    Local JSON descriptors bind authorized raw dictionaries by SHA256. This producer
    has no network calls, fitting, labels, IDs, comments, image or answer inputs.
    """

    def __init__(self, language, resources):
        self.language = language
        self.categories = emotion_categories(language)
        self.feature_names = emotion_layout(language)
        self.output_dim = len(self.feature_names)
        path = Path(resources)
        spec = json.loads(path.read_text("utf-8"), object_pairs_hook=_json_object)
        if (
            not isinstance(spec, dict)
            or set(spec) != {"schema", "language", "files"}
            or type(spec["schema"]) is not int
            or spec["schema"] != 1
            or spec["language"] != language
            or not isinstance(spec["files"], dict)
        ):
            raise ValueError("Emotion resource JSON requires schema=1, language and files.")
        required = _resource_names(language)
        if set(spec["files"]) != required:
            raise ValueError(f"Emotion files must be exactly: {sorted(required)}.")
        texts, identities = {}, {}
        for name, descriptors in spec["files"].items():
            is_list = name in {"positive", "negative"}
            if is_list and (not isinstance(descriptors, list) or not descriptors):
                raise ValueError(f"Emotion {name} must be a nonempty descriptor list.")
            descriptors = descriptors if is_list else [descriptors]
            contents, hashes = [], []
            for descriptor in descriptors:
                if (
                    not isinstance(descriptor, dict)
                    or set(descriptor) != {"path", "sha256", "source", "license"}
                    or any(
                        not isinstance(value, str) or not value.strip()
                        for value in descriptor.values()
                    )
                    or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
                ):
                    raise ValueError(f"Invalid local emotion descriptor: {name}.")
                resource_path = path.parent / descriptor["path"]
                content = resource_path.read_bytes()
                if hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
                    raise ValueError(f"Emotion resource SHA256 mismatch: {name}.")
                contents.append(content.decode("utf-8-sig"))
                hashes.append({key: descriptor[key] for key in ("sha256", "source", "license")})
            texts[name] = contents if is_list else contents[0]
            identities[name] = hashes if is_list else hashes[0]
        self.negation = _word_list(texts["negation"], "negation")
        self.degree = _scores(texts["degree"], "degree")
        self.positive = set().union(*(_word_list(text, "positive") for text in texts["positive"]))
        self.negative = set().union(*(_word_list(text, "negative") for text in texts["negative"]))
        self.pronouns = [
            set(_word_list(texts[f"pronoun_{person}"], f"pronoun_{person}"))
            for person in ("first", "second", "third")
        ]
        if language == "en":
            self.association = _nrc(texts["nrc_association"], _NRC_CATEGORIES)
            self.intensity = _nrc(texts["nrc_intensity"], _EN_CATEGORIES, intensity=True)
            self._prepare_vader(texts["vader_lexicon"])
            tokenization = {
                "algorithm": "case-preserving-source-regexp-url-mention-removal",
                "nltk": version("nltk"),
            }
        else:
            self.dalian = _dalian(texts["dalian"])
            self.boson = _scores(texts["boson"], "Boson", phrases=False)
            self.emoticons = _emoticons(texts["emoticons"])
            self._prepare_jieba(texts["jieba_dictionary"])
            tokenization = {
                "algorithm": "jieba",
                "version": version("jieba"),
                "HMM": True,
                "keep_whitespace_tokens": True,
            }
        self.provenance = _provenance(language, identities, tokenization)

    def _prepare_vader(self, text):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

        lexicon = {}
        for row in csv.reader(io.StringIO(text), delimiter="\t"):
            if not row or not any(row):
                continue
            if len(row) < 2 or not row[0]:
                raise ValueError("Malformed VADER lexicon row.")
            value = _number(row[1], "VADER valence")
            if row[0] in lexicon and lexicon[row[0]] != value:
                raise ValueError("Conflicting VADER lexicon entry.")
            lexicon[row[0]] = value
        if not lexicon:
            raise ValueError("Empty VADER lexicon.")
        # Initialize NLTK from the exact already-hashed bytes, with no data download/cache.
        self.vader = SentimentIntensityAnalyzer.__new__(SentimentIntensityAnalyzer)
        self.vader.lexicon_file = text
        self.vader.lexicon = lexicon
        self.vader.constants = VaderConstants()

    def _prepare_jieba(self, text):
        import jieba

        if version("jieba") != "0.42.1":
            raise ValueError("Event publisher emotion requires jieba==0.42.1.")
        for line in _word_list(text, "jieba_dictionary"):
            pieces = line.split(" ")
            if (
                len(pieces) not in {2, 3}
                or not pieces[0]
                or not pieces[1].isdigit()
                or int(pieces[1]) < 1
            ):
                raise ValueError("Malformed jieba dictionary word/frequency row.")
        self.segmenter = jieba.Tokenizer()
        self.segmenter.FREQ, self.segmenter.total = self.segmenter.gen_pfdict(
            io.BytesIO(text.encode("utf-8"))
        )
        self.segmenter.initialized = True

    def _tokenize(self, text):
        if self.language == "zh":
            return self.segmenter.lcut(text, HMM=True)
        cleaned = re.sub(r"https?://" + _URL_BODY, "", text)
        cleaned = re.sub("@" + _URL_BODY, "", cleaned).replace("\r", "").replace("\t", "")
        return re.findall(_EN_TOKENS, cleaned)

    def _context_weight(self, tokens, index):
        window = 4 if self.language == "en" else 2
        context = " ".join(tokens[max(0, index - window) : index])
        count = sum(word in context for word in self.negation)
        degree = math.prod(value for word, value in self.degree.items() if word in context)
        return (-1) ** count * degree

    def _weighted_sum(self, tokens, lexicon, size):
        values = [0.0] * size
        for index, word in enumerate(tokens):
            if word in lexicon:
                weight = self._context_weight(tokens, index)
                vector = lexicon[word]
                for column, value in enumerate(vector):
                    values[column] += weight * value
        return values

    def _word_statistics(self, tokens):
        if not tokens:
            return [0.0] * 7
        unique = set(tokens)
        count = len(tokens)
        return [
            len(unique & self.positive) / count,
            len(unique & self.negative) / count,
            sum(value for word, value in self.degree.items() if word in unique),
            sum(tokens.count(word) for word in self.negation) / count,
            *(sum(tokens.count(word) for word in words) / count for words in self.pronouns),
        ]

    def extract(self, text, category_scores):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Publisher emotion requires nonempty text.")
        if not isinstance(category_scores, dict) or set(category_scores) != set(self.categories):
            raise ValueError("Publisher emotion requires the exact eight category scores.")
        scores = []
        for name in self.categories:
            value = category_scores[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("Category intensities must be finite numbers in [0,1].")
            value = _number(value, "Category intensity")
            if not 0 <= value <= 1:
                raise ValueError("Category intensities must be in [0,1].")
            scores.append(value)
        tokens = self._tokenize(text)
        punctuation = [
            (text.count(first) + text.count(second)) / len(text)
            for first, second in (
                ("!", "！"),
                ("?", "？"),
                (",", "，"),
                (".", "。"),
                ("..", "。。"),
            )
        ]
        if self.language == "en":
            sentiment = self.vader.polarity_scores(text)
            emoticons = [
                sum(text.count(word) for word in _SMILES) / len(text),
                sum(text.count(word) for word in _FROWNS) / len(text),
                sum(any(start <= ord(char) <= end for start, end in _EMOJI_RANGES) for char in text)
                / len(text),
            ]
            values = [
                *(float(value >= 0.5) for value in scores),
                *scores,
                *self._weighted_sum(tokens, self.association, 10),
                *self._weighted_sum(tokens, self.intensity, 8),
                *(sentiment[key] for key in ("pos", "neg", "neu", "compound")),
                *emoticons,
                *punctuation,
                *self._word_statistics(tokens),
                sum(char.isupper() for char in text) / len(text),
            ]
        else:
            emoticons = [
                sum(text.count(word) for word, label in self.emoticons.items() if label == name)
                / len(tokens)
                if tokens
                else 0.0
                for name in _EMOTICON_CATEGORIES
            ]
            boson = sum(
                self._context_weight(tokens, index) * self.boson[word]
                for index, word in enumerate(tokens)
                if word in self.boson
            )
            values = [
                *scores,
                *self._weighted_sum(tokens, self.dalian, 29),
                boson,
                *emoticons,
                *punctuation,
                *self._word_statistics(tokens),
            ]
        if len(values) != self.output_dim or not all(math.isfinite(value) for value in values):
            raise ValueError("Publisher emotion produced invalid features.")
        return [float(value) for value in values]
