"""Train-only sparse TF-IDF and binary Logistic Regression (C02)."""

import math
import re
import unicodedata
import warnings
from collections import Counter
from copy import deepcopy

import numpy as np
import scipy
import sklearn
import torch
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.data.features import fingerprint
from mmfact.models_classification._classical_baselines import (
    _checksum,
    _fit_seed,
    _training_data,
    _validate_identity,
)
from mmfact.registry import register_model

_RECIPE = {
    "name": "tfidf_logistic",
    "version": 1,
    "unicode": "NFC",
    "whitespace": "unicode-collapse-strip",
    "en_tokens": r"(?u)\b\w+\b",
    "zh_tokens": "codepoints-with-internal-space-and-punctuation",
    "use_idf": True,
    "dtype": "float64",
    "layout": "csr",
    "logits": "[0,z]",
}


def _analyze(text, language, options):
    if not isinstance(text, str):
        raise ValueError("TF-IDF input text must be a string.")
    text = unicodedata.normalize("NFC", text)
    if options["lowercase"]:
        text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"(?u)\b\w+\b", text) if language == "en" else list(text)
    lower, upper = options["ngram_range"]
    join = " ".join if language == "en" else "".join
    return [
        join(tokens[index : index + size])
        for size in range(lower, min(upper, len(tokens)) + 1)
        for index in range(len(tokens) - size + 1)
    ]


def _validate_parameters(p):
    if (
        not isinstance(p, dict)
        or set(p) != {"language", "tfidf", "logistic"}
        or p["language"] not in ("en", "zh")
    ):
        raise ValueError("TFIDFLogistic requires language=en/zh, tfidf and logistic.")
    t, lr = p["tfidf"], p["logistic"]
    if not isinstance(t, dict) or set(t) != {
        "ngram_range",
        "max_features",
        "min_df",
        "max_df",
        "sublinear_tf",
        "smooth_idf",
        "norm",
        "lowercase",
    }:
        raise ValueError("TF-IDF parameter fields are invalid.")
    ngrams = t["ngram_range"]
    if (
        not isinstance(ngrams, list)
        or len(ngrams) != 2
        or any(type(n) is not int or n < 1 for n in ngrams)
        or ngrams[0] > ngrams[1]
    ):
        raise ValueError("ngram_range must be an ordered pair of positive integers.")
    if type(t["max_features"]) is not int or t["max_features"] < 1 or t["norm"] != "l2":
        raise ValueError("TF-IDF requires positive integer max_features and norm=l2.")
    for name in ("sublinear_tf", "smooth_idf", "lowercase"):
        if type(t[name]) is not bool:
            raise ValueError(f"tfidf.{name} must be a boolean.")
    for name in ("min_df", "max_df"):
        value = t[name]
        if not (
            type(value) is int
            and value >= 1
            or type(value) is float
            and math.isfinite(value)
            and 0 <= value <= 1
        ):
            raise ValueError(f"tfidf.{name} must be a positive count or fraction in [0,1].")
    if not isinstance(lr, dict) or set(lr) != {
        "penalty",
        "solver",
        "C",
        "tol",
        "max_iter",
        "fit_intercept",
        "intercept_scaling",
        "class_weight",
        "dual",
        "warm_start",
    }:
        raise ValueError("Logistic parameter fields are invalid.")
    if (
        lr["penalty"] != "l2"
        or lr["solver"] != "liblinear"
        or lr["class_weight"] is not None
        or lr["dual"] is not False
        or lr["warm_start"] is not False
    ):
        raise ValueError(
            "Logistic baseline requires L2/liblinear, no class weights, dual or warm start."
        )
    if (
        type(lr["max_iter"]) is not int
        or lr["max_iter"] < 1
        or type(lr["fit_intercept"]) is not bool
    ):
        raise ValueError("Logistic requires positive integer max_iter and boolean fit_intercept.")
    for name in ("C", "tol", "intercept_scaling"):
        if type(lr[name]) not in (int, float) or not math.isfinite(lr[name]) or lr[name] <= 0:
            raise ValueError(f"logistic.{name} must be finite and positive.")


@register_model("TFIDFLogistic")
class TFIDFLogistic(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = deepcopy(config["model"]["parameters"])
        _validate_parameters(p)
        self.parameters_spec, self.resource, self._vocabulary = p, None, {}
        self.register_buffer("_device", torch.empty(0), persistent=False)

    def training_stages(self):
        return (TrainingStage("fit", mode="fit"),)

    def calculate_loss(self, batch):
        raise RuntimeError("TFIDFLogistic uses a single fit stage, not gradient training.")

    def _analyze(self, text):
        return _analyze(text, self.parameters_spec["language"], self.parameters_spec["tfidf"])

    def fit_stage(self, stage, train_dataset, val_loader, options):
        seed = _fit_seed(stage, options)
        texts, labels, identity = _training_data(train_dataset, text=True)
        counts = [labels.count(0), labels.count(1)]
        if not all(counts):
            raise ValueError("TFIDFLogistic requires both training classes.")
        t, lr = self.parameters_spec["tfidf"], self.parameters_spec["logistic"]
        vectorizer = TfidfVectorizer(
            analyzer=self._analyze,
            lowercase=False,
            token_pattern=None,
            dtype=np.float64,
            **{
                key: t[key]
                for key in (
                    "max_features",
                    "min_df",
                    "max_df",
                    "sublinear_tf",
                    "smooth_idf",
                    "norm",
                )
            },
        )
        features = vectorizer.fit_transform(texts)
        estimator = LogisticRegression(**lr, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(features, labels)
        iterations = int(estimator.n_iter_.max())
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise ValueError(
                f"Logistic did not converge: C={lr['C']}, tol={lr['tol']}, max_iter={lr['max_iter']}, n_iter={iterations}."
            )
        resource = {
            "ready": True,
            "recipe": deepcopy(_RECIPE),
            "training_sha256": identity,
            "num_samples": len(labels),
            "class_counts": counts,
            "vocabulary": vectorizer.get_feature_names_out().tolist(),
            "idf": torch.from_numpy(vectorizer.idf_.copy()),
            "coef": torch.from_numpy(estimator.coef_.copy()),
            "intercept": torch.tensor(np.atleast_1d(estimator.intercept_), dtype=torch.float64),
            "classes": estimator.classes_.tolist(),
            "n_iter": iterations,
            "seed": seed,
            "sklearn_version": sklearn.__version__,
            "scipy_version": scipy.__version__,
        }
        resource["sha256"] = _checksum(resource)
        self._validate_resource(resource)
        self.resource = resource
        self._vocabulary = {term: index for index, term in enumerate(resource["vocabulary"])}
        return {
            **{
                key: deepcopy(resource[key])
                for key in (
                    "num_samples",
                    "class_counts",
                    "training_sha256",
                    "n_iter",
                    "seed",
                    "sklearn_version",
                    "scipy_version",
                )
            },
            "vocabulary_size": len(self._vocabulary),
            "converged": True,
        }

    def _features(self, texts):
        if self.resource is None:
            raise RuntimeError("TFIDFLogistic must be fitted or restored before prediction.")
        data, indices, indptr = [], [], [0]
        idf, options = self.resource["idf"].numpy(), self.parameters_spec["tfidf"]
        for text in texts:
            counts = Counter(self._analyze(text))
            entries = sorted(
                (self._vocabulary[term], count)
                for term, count in counts.items()
                if term in self._vocabulary
            )
            for index, count in entries:
                data.append(
                    (1 + math.log(count) if options["sublinear_tf"] else count) * idf[index]
                )
                indices.append(index)
            indptr.append(len(data))
        matrix = csr_matrix(
            (np.asarray(data, dtype=np.float64), indices, indptr),
            shape=(len(texts), len(self._vocabulary)),
        )
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        matrix.data /= np.repeat(
            np.maximum(norms, np.finfo(np.float64).tiny), np.diff(matrix.indptr)
        )
        return matrix

    @torch.no_grad()
    def predict(self, batch):
        features = self._features(batch["text"])
        values = (
            features @ self.resource["coef"].numpy().ravel() + self.resource["intercept"].item()
        )
        if not np.isfinite(values).all():
            raise ValueError("TFIDFLogistic prediction must produce finite logits.")
        return torch.from_numpy(np.column_stack((np.zeros_like(values), values))).to(
            self._device.device
        )

    def get_extra_state(self):
        return {
            "version": 1,
            "parameters": deepcopy(self.parameters_spec),
            "resource": deepcopy(self.resource),
        }

    def _validate_resource(self, resource):
        if not isinstance(resource, dict) or set(resource) != {
            "ready",
            "recipe",
            "training_sha256",
            "num_samples",
            "class_counts",
            "sha256",
            "vocabulary",
            "idf",
            "coef",
            "intercept",
            "classes",
            "n_iter",
            "seed",
            "sklearn_version",
            "scipy_version",
        }:
            raise ValueError("TFIDFLogistic checkpoint resource fields are invalid.")
        _validate_identity(resource, _RECIPE)
        vocabulary = resource["vocabulary"]
        if (
            not isinstance(vocabulary, list)
            or not vocabulary
            or any(not isinstance(term, str) or not term for term in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or len(vocabulary) > self.parameters_spec["tfidf"]["max_features"]
        ):
            raise ValueError("TFIDFLogistic checkpoint vocabulary is invalid.")
        width = len(vocabulary)
        for key, shape in (("idf", (width,)), ("coef", (1, width)), ("intercept", (1,))):
            value = resource[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or tuple(value.shape) != shape
                or not torch.isfinite(value).all()
                or value.requires_grad
            ):
                raise ValueError(
                    f"TFIDFLogistic checkpoint {key} must be finite CPU float64 with shape {shape}."
                )
        n = resource["num_samples"]
        maximum = (
            1 + math.log((1 + n) / 2)
            if self.parameters_spec["tfidf"]["smooth_idf"]
            else 1 + math.log(n)
        )
        if (resource["idf"] < 1).any() or (resource["idf"] > maximum + 1e-12).any():
            raise ValueError("TFIDFLogistic checkpoint IDF is outside its training bounds.")
        if (
            not isinstance(resource["classes"], list)
            or resource["classes"] != [0, 1]
            or any(type(value) is not int for value in resource["classes"])
            or not all(resource["class_counts"])
        ):
            raise ValueError("TFIDFLogistic checkpoint must have both ordered classes [0,1].")
        if (
            type(resource["n_iter"]) is not int
            or not 1 <= resource["n_iter"] <= self.parameters_spec["logistic"]["max_iter"]
        ):
            raise ValueError("TFIDFLogistic checkpoint iteration count is invalid.")
        _fit_seed("fit", resource)
        if any(
            not isinstance(resource[key], str) or not resource[key].strip()
            for key in ("sklearn_version", "scipy_version")
        ):
            raise ValueError("TFIDFLogistic checkpoint library versions are invalid.")
        if (
            not self.parameters_spec["logistic"]["fit_intercept"]
            and resource["intercept"].item() != 0
        ):
            raise ValueError("A Logistic model without an intercept must restore zero intercept.")

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "parameters", "resource"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or fingerprint(state["parameters"]) != fingerprint(self.parameters_spec)
        ):
            raise ValueError("TFIDFLogistic checkpoint parameters or schema differ.")
        resource = deepcopy(state["resource"])
        if resource is not None:
            self._validate_resource(resource)
        vocabulary = (
            {}
            if resource is None
            else {term: index for index, term in enumerate(resource["vocabulary"])}
        )
        self.resource, self._vocabulary = resource, vocabulary
