"""SAFE (PAKDD 2020), with explicit caption and training-only TokenSIF adaptation."""

import math
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.data.features import file_digest, fingerprint, load_resource
from mmfact.materials.encoders import hub_inputs, material_store
from mmfact.materials.hub import HubEncoder
from mmfact.materials.training import hub_training_resource
from mmfact.models_classification._encoders import _PretrainedEncoder
from mmfact.registry import register_model

_CAPTION_MODEL = "gpt-4o-2024-08-06"
_CAPTION_PARAMETERS = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 128,
    "image_detail": "low",
    "image_encoding": "rgb_png",
}
_CAPTION_PROMPTS = {
    "en": (
        "Describe only the visible content of this image in one short factual sentence in English. "
        "Mention salient objects, people, actions, and visible text only when clearly legible. "
        "Do not assess authenticity, infer whether news is true or false, or use information "
        "outside the image."
    ),
    "zh": (
        "仅用一句简短的中文句子描述图片中可见的内容。只提及清晰可见的主要物体、人物、动作，"
        "以及能够辨认的图片文字。不要判断真实性、推断新闻真假或使用图片以外的信息。"
    ),
}


class _CaptionStore(nn.Module):
    """Read image-only descriptions; preserve their provenance in checkpoints."""

    def __init__(self, path, language):
        super().__init__()
        self.path, self.language = path, language
        self.rows = {}
        self.generator = {
            "model": _CAPTION_MODEL,
            "language": language,
            "prompt": _CAPTION_PROMPTS[language],
            "parameters": dict(_CAPTION_PARAMETERS),
        }

    def _validate(self, rows, generator, allow_empty=False):
        if (
            generator != self.generator
            or not isinstance(rows, dict)
            or (not rows and not allow_empty)
        ):
            raise ValueError("SAFE requires a nonempty caption manifest with the fixed generator.")
        if any(
            not isinstance(key, str)
            or re.fullmatch(r"[0-9a-f]{64}", key) is None
            or not isinstance(value, str)
            or not value.strip()
            for key, value in rows.items()
        ):
            raise ValueError("SAFE captions must map lowercase image SHA256 to nonempty text.")

    def load(self, allow_empty=False):
        if self.rows:
            return
        import json

        rows = {}
        with Path(self.path).expanduser().open(encoding="utf-8") as stream:
            header = json.loads(next(stream))
            if type(header.get("schema")) is not int or header["schema"] != 1:
                raise ValueError("Unsupported SAFE caption schema; expected 1.")
            for line in stream:
                row = json.loads(line)
                key = row["image_sha256"]
                if key in rows:
                    raise ValueError(f"Duplicate SAFE caption image SHA256: {key}")
                rows[key] = row["caption"]
        self._validate(rows, header.get("generator"), allow_empty=allow_empty)
        self.rows = rows

    def forward(self, paths):
        self.load()
        try:
            return [self.rows[file_digest(path)] for path in paths]
        except KeyError as error:
            raise ValueError(
                f"SAFE caption is missing for image SHA256 {error.args[0]}."
            ) from error

    def get_extra_state(self):
        return {"generator": self.generator, "rows": self.rows}

    def set_extra_state(self, state):
        if state["rows"]:
            self._validate(state["rows"], state["generator"])
        elif state["generator"] != self.generator:
            raise ValueError("SAFE checkpoint caption generator does not match configuration.")
        self.rows = dict(state["rows"])


class _Tokenizer:
    def __init__(self, language, tokenization):
        if (language, tokenization) not in {("en", "author"), ("en", "regex"), ("zh", "jieba")}:
            raise ValueError("SAFE tokenization must be en/author, en/regex, or zh/jieba.")
        self.language, self.tokenization = language, tokenization
        self.resources = None

    def _english_resources(self):
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        try:
            stops = set(stopwords.words("english"))
            wordnet.ensure_loaded()
        except LookupError as error:
            raise ValueError(
                "SAFE English cleaning needs local NLTK data: "
                "python -m nltk.downloader stopwords wordnet"
            ) from error
        archive = getattr(wordnet.root, "zipfile", None)
        wordnet_digest = (
            file_digest(archive.filename)
            if archive is not None
            else fingerprint(
                {name: file_digest(Path(str(wordnet.root)) / name) for name in wordnet.fileids()}
            )
        )
        self.resources = (stops, PorterStemmer(), WordNetLemmatizer())
        self.resource_identity = {"stopwords": sorted(stops), "wordnet_sha256": wordnet_digest}

    @property
    def spec(self):
        if self.tokenization == "author" and self.resources is None:
            self._english_resources()
        if self.language == "zh":
            self._chinese_resources()
        return {
            "language": self.language,
            "tokenization": self.tokenization,
            "version": 1,
            "package_version": version("jieba" if self.language == "zh" else "nltk"),
            "resources": getattr(self, "resource_identity", None),
        }

    def _chinese_resources(self):
        import jieba

        if self.resources is None:
            self.resources = jieba.Tokenizer()
        with self.resources.get_dict_file() as dictionary:
            self.resource_identity = {"dictionary_sha256": file_digest(dictionary.name)}

    def __call__(self, text):
        text = re.sub(r"[\W\d_]+", " ", text.lower(), flags=re.UNICODE)
        if self.language == "zh":
            if self.resources is None:
                self._chinese_resources()
            return [token for token in self.resources.lcut(text) if token.strip()]
        tokens = text.split()
        if self.tokenization == "author":
            if self.resources is None:
                self._english_resources()
            stops, stemmer, lemmatizer = self.resources
            tokens = [
                lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens if word not in stops
            ]
        return tokens


class _SIFState(nn.Module):
    """Shared frozen vectors, training frequencies and sentence-derived PCs."""

    def __init__(self, dimension, pc_count):
        super().__init__()
        self.dimension, self.pc_count = dimension, pc_count
        self.register_buffer("vectors", torch.zeros(2, dimension))
        self.register_buffer("weights", torch.tensor([0.0, 1.0]))
        self.register_buffer("components", torch.zeros(pc_count, dimension))
        self.vocabulary, self.identity = {"<pad>": 0, "<unk>": 1}, None

    def get_extra_state(self):
        return {"vocabulary": self.vocabulary, "identity": self.identity}

    def set_extra_state(self, state):
        vocabulary = state["vocabulary"]
        if (
            vocabulary.get("<pad>") != 0
            or vocabulary.get("<unk>") != 1
            or set(vocabulary.values()) != set(range(len(vocabulary)))
            or len(vocabulary) != len(self.vectors)
            or self.vectors.shape != (len(vocabulary), self.dimension)
            or self.weights.shape != (len(vocabulary),)
            or not all(torch.isfinite(value).all() for value in self.buffers())
        ):
            raise ValueError("Invalid SAFE TokenSIF checkpoint vocabulary or vectors.")
        self.vocabulary, self.identity = dict(vocabulary), state["identity"]

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        for name in ("vectors", "weights"):
            value = state.get(prefix + name)
            if value is not None:
                setattr(self, name, torch.empty_like(value, device=self.vectors.device))
        super()._load_from_state_dict(state, prefix, *args, **kwargs)

    def token_vectors(self, sequences, max_length):
        if self.identity is None:
            raise RuntimeError("Prepare SAFE on training data or restore a checkpoint first.")
        ids = torch.zeros(len(sequences), max_length, dtype=torch.long, device=self.vectors.device)
        for index, words in enumerate(sequences):
            values = [self.vocabulary.get(word, 1) for word in words[:max_length]]
            ids[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=ids.device)
        values = F.embedding(ids, self.vectors) * self.weights[ids, None]
        return values - (values @ self.components.T) @ self.components


class _SIFEncoder(_PretrainedEncoder):
    packages = ("torch", "nltk", "jieba")

    def __init__(self, state, tokenizer, max_length, selector):
        super().__init__()
        self.network, self.tokenizer = state, tokenizer
        self.max_length, self.selector = max_length, selector
        self._set_frozen(True)

    def inputs_from_rows(self, rows):
        return self.selector(rows)

    @property
    def encoding_spec(self):
        return {
            "encoder": "safe_token_sif",
            "max_length": self.max_length,
            "tokenization": self.tokenizer.spec,
            "training_resource": self.network.identity,
            "output": "weighted_tokens_sentence_pc_v1",
        }

    def _encode(self, inputs):
        if self.network.identity is not None and (
            self.tokenizer.spec != self.network.identity["tokenization"]
        ):
            raise ValueError("SAFE tokenizer resources differ from its fitted/checkpoint state.")
        return self.network.token_vectors(
            [self.tokenizer(text) for text in inputs], self.max_length
        )


class _HubSIFEncoder(HubEncoder):
    def inputs_from_rows(self, rows):
        if self.input_kind == "caption":
            return [
                row.get("metadata", {}).get("image_id") or Path(row["image_path"]).stem
                for row in rows
            ]
        return [row["text"] for row in rows]


class _TextCNN(nn.Module):
    def __init__(self, input_dim, filters, kernels, output_dim, dropout):
        super().__init__()
        self.convolutions = nn.ModuleList(nn.Conv1d(input_dim, filters, size) for size in kernels)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(filters * len(kernels), output_dim)
        for layer in self.convolutions:
            bound = math.sqrt(6 / (input_dim * layer.kernel_size[0] * (1 + filters)))
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens):
        sequence = tokens.transpose(1, 2)
        pooled = [F.relu(layer(sequence)).amax(-1) for layer in self.convolutions]
        return self.projection(self.dropout(torch.cat(pooled, -1)))


@register_model("SAFE")
class SAFE(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        text = p["text"]
        dimension, pc_count = text["dimension"], text["pc_count"]
        lengths, kernels = p["lengths"], p["kernel_sizes"]
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    dimension,
                    text["max_vocab_size"],
                    p["shared_dim"],
                    p["filter_num"],
                    p["output_dim"],
                    *lengths.values(),
                    *kernels,
                )
            )
            or text["max_vocab_size"] < 3
            or type(pc_count) is not int
            or not 0 <= pc_count < dimension
            or set(lengths) != {"headline", "body", "caption"}
            or not kernels
            or min(lengths.values()) < max(kernels)
        ):
            raise ValueError("SAFE sizes, vocabulary, PC count and CNN kernels are incompatible.")
        scalar = (text["sif_a"], p["epsilon"], p["dropout"], *p["loss_weights"])
        if (
            not all(math.isfinite(value) for value in scalar)
            or text["sif_a"] <= 0
            or p["epsilon"] <= 0
            or p["epsilon"] >= 0.5
            or not 0 <= p["dropout"] < 1
            or len(p["loss_weights"]) != 2
            or min(p["loss_weights"]) <= 0
        ):
            raise ValueError("SAFE requires finite positive SIF/loss settings and valid dropout.")
        if hub_inputs(config):
            resource = hub_training_resource(
                config,
                "safe_token_sif_v1",
                expected={"dimension": dimension, "pc_count": pc_count, "sif_a": text["sif_a"]},
            )
            if resource.settings.get("tokenization", {}).get("language") != text["language"]:
                raise ValueError("SAFE fitted TokenSIF language disagrees with configuration.")
            store = material_store(config)
            encoders = {}
            for name, length in lengths.items():
                matrix = store.feature_for("SAFE", f"encoders.{name}")
                if matrix.shape[1:] != (length, dimension):
                    raise ValueError("SAFE prepared TokenSIF shape disagrees with configuration.")
                encoders[name] = _HubSIFEncoder(matrix)
            self.encoders = nn.ModuleDict(encoders)
            self.tokenizer, self.captions = None, None
        else:
            self.tokenizer = _Tokenizer(text["language"], text["tokenization"])
            self.captions = _CaptionStore(p["caption_manifest"], text["language"])
            state = _SIFState(dimension, pc_count)
            self.encoders = nn.ModuleDict(
                {
                    name: _SIFEncoder(
                        state, self.tokenizer, length, getattr(self, f"_{name}_inputs")
                    )
                    for name, length in lengths.items()
                }
            )
        self.implementation = p.get("implementation", "author")
        if self.implementation not in {"author", "paper"}:
            raise ValueError("SAFE implementation must be author or paper.")
        branches = ("body", "caption") if self.implementation == "paper" else tuple(lengths)
        # Paper Sec3 describes one text CNN and one image-caption CNN.
        # Author code instead uses a shared 300->32 transform and three CNNs.
        self.shared = (
            nn.Identity()
            if self.implementation == "paper"
            else nn.Linear(dimension, p["shared_dim"])
        )
        cnn_input = dimension if self.implementation == "paper" else p["shared_dim"]
        self.cnns = nn.ModuleDict(
            {
                name: _TextCNN(cnn_input, p["filter_num"], kernels, p["output_dim"], p["dropout"])
                for name in branches
            }
        )
        self.classifier = nn.Linear(len(branches) * p["output_dim"], 2)
        for layer in (self.shared, self.classifier):
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.trunc_normal_(layer.weight, std=0.1, a=-0.2, b=0.2)
            nn.init.constant_(layer.bias, 0.1)
        self.loss_weights, self.epsilon = tuple(p["loss_weights"]), p["epsilon"]

    @staticmethod
    def _body_inputs(rows):
        return [row["text"] for row in rows]

    @staticmethod
    def _headline_inputs(rows):
        values = []
        for row in rows:
            title = row.get("metadata", {}).get("title")
            if title is not None and not isinstance(title, str):
                raise ValueError("SAFE metadata.title must be a string when provided.")
            values.append(title if title is not None and title.strip() else row["text"])
        return values

    def _caption_inputs(self, rows):
        return self.captions([row["image_path"] for row in rows])

    def _fit_sif(self, sequences, identity):
        options = self.config["model"]["parameters"]["text"]
        counts = Counter(word for words in sequences for word in words)
        words = sorted(counts, key=lambda word: (-counts[word], word))[
            : options["max_vocab_size"] - 2
        ]
        if not words:
            raise ValueError("SAFE training text produced an empty vocabulary.")
        wanted, found = set(words), {}
        with Path(options["vector_path"]).expanduser().open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                fields = line.split()
                if index == 0 and len(fields) == 2 and all(value.isdigit() for value in fields):
                    if int(fields[1]) != options["dimension"]:
                        raise ValueError(
                            "SAFE vector header dimension does not match configuration."
                        )
                    continue
                if not fields or fields[0] not in wanted:
                    continue
                values = torch.tensor([float(value) for value in fields[1:]])
                if values.shape != (options["dimension"],) or not torch.isfinite(values).all():
                    raise ValueError(f"Invalid SAFE pretrained vector for {fields[0]}.")
                if fields[0] in found:
                    raise ValueError(f"Duplicate SAFE pretrained vector for {fields[0]}.")
                found[fields[0]] = values
        if not found:
            raise ValueError(
                "SAFE training vocabulary has no pretrained vectors; check language/resource."
            )
        vocabulary = {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}
        unknown = torch.stack(list(found.values())).mean(0)
        vectors = torch.stack(
            [torch.zeros_like(unknown), unknown, *[found.get(w, unknown) for w in words]]
        )
        a, total = options["sif_a"], sum(counts.values())
        unknown_count = sum(count for word, count in counts.items() if word not in wanted)
        weights = torch.tensor(
            [
                0.0,
                a / (a + unknown_count / total),
                *[a / (a + counts[word] / total) for word in words],
            ]
        )
        gram = torch.zeros(options["dimension"], options["dimension"], dtype=torch.float64)
        means = []
        for sequence in sequences:
            if sequence:
                ids = torch.tensor([vocabulary.get(word, 1) for word in sequence])
                mean = (vectors[ids] * weights[ids, None]).double().mean(0)
                means.append(mean)
                if len(means) == 512:
                    matrix = torch.stack(means)
                    gram.add_(matrix.T @ matrix)
                    means.clear()
        if means:
            matrix = torch.stack(means)
            gram.add_(matrix.T @ matrix)
        components = torch.zeros(options["pc_count"], options["dimension"])
        if options["pc_count"] and gram.abs().sum() > 0:
            _, directions = torch.linalg.eigh(gram)
            components.copy_(directions[:, -options["pc_count"] :].T.float())
        return {
            "vectors": vectors,
            "weights": weights,
            "components": components,
            "vocabulary": vocabulary,
            "identity": {**identity, "matched_words": len(found), "vocabulary_words": len(words)},
        }

    def prepare(self, train_dataset):
        if hub_inputs(self.config):
            rows = list(train_dataset)
            for encoder in self.encoders.values():
                for key in encoder._keys(encoder.inputs_from_rows(rows)):
                    if key not in encoder.matrix.rows:
                        raise ValueError("SAFE prepared features do not cover training inputs.")
            return
        rows, sequences = list(train_dataset), []
        for encoder in self.encoders.values():
            sequences.extend(
                self.tokenizer(text)[: encoder.max_length]
                for text in encoder.inputs_from_rows(rows)
            )
        options = self.config["model"]["parameters"]["text"]
        identity = {
            "encoder": "safe_token_sif_v1",
            "vector_sha256": file_digest(options["vector_path"]),
            "dimension": options["dimension"],
            "sif_a": options["sif_a"],
            "pc_count": options["pc_count"],
            "max_vocab_size": options["max_vocab_size"],
            "tokenization": self.tokenizer.spec,
            "training_tokens": fingerprint(sequences),
            "caption_generator": self.captions.generator,
        }

        def build():
            return self._fit_sif(sequences, identity)

        preprocessing = self.config.get("preprocessing", {})
        payload = (
            load_resource(preprocessing["cache_dir"], identity, build)
            if preprocessing.get("enabled", False)
            else build()
        )
        state = self.encoders["headline"].network
        state.load_state_dict(
            {
                "vectors": payload["vectors"],
                "weights": payload["weights"],
                "components": payload["components"],
                "_extra_state": {
                    "vocabulary": payload["vocabulary"],
                    "identity": payload["identity"],
                },
            },
            strict=True,
        )
        for encoder in self.encoders.values():
            if encoder.preprocessing is not None:
                encoder.preprocessing.reset()

    @staticmethod
    def _rows(batch):
        count = len(batch["text"])
        metadata = batch.get("metadata", [{} for _ in range(count)])
        if len(metadata) != count or len(batch["image_path"]) != count:
            raise ValueError("SAFE batch fields must have equal lengths.")
        return [
            {"text": text, "image_path": path, "metadata": data}
            for text, path, data in zip(batch["text"], batch["image_path"], metadata)
        ]

    def forward(self, batch):
        rows = self._rows(batch)
        features = {
            name: self.cnns[name](self.shared(encoder(encoder.inputs_from_rows(rows))))
            for name, encoder in self.encoders.items()
            if name in self.cnns
        }
        combined_text = (
            features["body"]
            if self.implementation == "paper"
            else torch.cat([features["headline"], features["body"]], -1)
        )
        combined_image = (
            features["caption"]
            if self.implementation == "paper"
            else features["caption"].repeat(1, 2)
        )
        cosine = (combined_text * combined_image).sum(-1) / (
            combined_text.norm(dim=-1) * combined_image.norm(dim=-1) + self.epsilon
        )
        similarity = ((1 + cosine) / 2).clamp(self.epsilon, 1 - self.epsilon)
        logits = self.classifier(
            torch.cat(
                [features[name] for name in self.cnns],
                -1,
            )
        )
        return logits, similarity

    def calculate_loss(self, batch):
        logits, similarity = self(batch)
        probabilities = torch.stack([similarity, 1 - similarity], -1)
        return {
            "classification": self.loss_weights[0] * F.cross_entropy(logits, batch["labels"]),
            "similarity": self.loss_weights[1] * F.nll_loss(probabilities.log(), batch["labels"]),
        }

    def predict(self, batch):
        mode = self.training
        self.eval()
        try:
            return self(batch)[0]
        finally:
            self.train(mode)
