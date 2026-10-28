"""Training-only multilabel domains; pseudo topics are not human domain annotations."""

import hashlib
import json
import math
import re
import unicodedata
from copy import deepcopy
from importlib.metadata import version

import torch
from torch import nn


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _name(value):
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and not re.fullmatch(r"(?:[+-]?\d+|(?:cluster|topic)[_-]?\d+)", value, re.I)
    )


def _select_topics(similarities, ratio, max_labels):
    """Top-1 always exists; a positive runner-up may supply the second label."""
    order = similarities.argsort(dim=1, descending=True, stable=True)
    selected = torch.zeros_like(similarities, dtype=torch.float32)
    selected.scatter_(1, order[:, :1], 1)
    if max_labels == 2:
        scores = similarities.gather(1, order[:, :2])
        second = (scores[:, 1] > 0) & (scores[:, 1] >= ratio * scores[:, 0])
        selected[second, order[second, 1]] = 1
    return selected, order[:, 0]


def _tfidf_recipe(options):
    return {
        "analyzer": "word" if options["language"] == "en" else "char",
        "ngram_range": tuple(options["ngram_range"]),
        "max_features": options["max_features"],
        "sublinear_tf": True,
        "lowercase": True,
        "norm": "l2",
        "token_pattern": r"(?u)\b\w+\b" if options["language"] == "en" else None,
    }


def _fingerprint(state):
    digest = hashlib.sha256()
    provenance = deepcopy(state["provenance"])
    provenance.pop("resource_sha256", None)
    fitted = provenance.pop("fitted", {})
    digest.update(
        _json([state["options"], state["sample_ids"], state["domain_names"], provenance]).encode()
    )
    for key, value in sorted({"labels": state["labels"], **fitted}.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(_json([str(value.dtype), list(value.shape)]).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(_json(value).encode())
    return digest.hexdigest()


class DomainTargets(nn.Module):
    """Freeze a train-ID→multi-hot mapping, never an inference-time text classifier.

    Topic validity requires all K cosine top-1 domains to be occupied and targets
    to vary across training samples. These are explicit adaptation checks, not
    guarantees of Euclidean KMeans. Metadata uses a separate complete vocabulary.
    """

    def __init__(
        self,
        mode="topics",
        num_domains=9,
        language="en",
        seed=42,
        max_features=20000,
        secondary_ratio=0.8,
        max_labels=2,
        ngram_range=(1, 2),
        domain_names=None,
        field="domain_labels",
    ):
        super().__init__()
        if mode not in ("topics", "metadata") or language not in ("en", "zh"):
            raise ValueError("Domain mode must be topics/metadata and language en/zh.")
        if (
            type(num_domains) is not int
            or num_domains < 2
            or type(max_features) is not int
            or max_features < 1
            or type(seed) is not int
            or not 0 <= seed < 2**32
            or type(max_labels) is not int
            or max_labels not in (1, 2)
            or type(secondary_ratio) not in (int, float)
            or not math.isfinite(secondary_ratio)
            or not 0 <= secondary_ratio <= 1
        ):
            raise ValueError("Domain sizes, seed or secondary-label settings are invalid.")
        if (
            not isinstance(ngram_range, (tuple, list))
            or len(ngram_range) != 2
            or any(type(value) is not int or value < 1 for value in ngram_range)
            or ngram_range[0] > ngram_range[1]
        ):
            raise ValueError("Domain ngram_range must contain two ordered positive integers.")
        if domain_names is not None and (
            mode != "metadata"
            or not isinstance(domain_names, list)
            or len(domain_names) != num_domains
            or not all(_name(name) for name in domain_names)
            or len(set(domain_names)) != num_domains
        ):
            raise ValueError("Explicit domain_names require K distinct semantic metadata names.")
        self.num_domains = num_domains
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Domain metadata field must be a nonempty string.")
        self._options = {
            "mode": mode,
            "num_domains": num_domains,
            "language": language,
            "seed": seed,
            "max_features": max_features,
            "secondary_ratio": secondary_ratio,
            "max_labels": max_labels,
            "ngram_range": list(ngram_range),
            "domain_names": deepcopy(domain_names),
            "field": field,
        }
        self._state = {
            "version": 1,
            "options": deepcopy(self._options),
            "sample_ids": [],
            "domain_names": [],
            "labels": torch.empty(0, num_domains),
            "provenance": {},
        }
        self._indices = {}

    @property
    def domain_names(self):
        return list(self._state["domain_names"])

    @property
    def provenance(self):
        return deepcopy(self._state["provenance"])

    def prepare(self, train_rows):
        """Read only training IDs and text, or complete domain_labels metadata."""
        rows = list(train_rows)
        ids = [row["sample_id"] for row in rows]
        if (
            not rows
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Domains require training rows with unique nonempty IDs.")
        rows = [row for _, row in sorted(zip(ids, rows), key=lambda pair: pair[0])]
        ids = sorted(ids)
        if self._options["mode"] == "topics":
            texts = [row["text"] for row in rows]
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError("Topic domains require nonempty training text.")
            texts = [unicodedata.normalize("NFC", text) for text in texts]
            if len(set(texts)) < self.num_domains:
                raise ValueError("Topic domains require at least K different training contents.")
            labels, names, audit = self._topics(texts)
            identity = list(zip(ids, texts))
        else:
            annotations = []
            for row in rows:
                try:
                    metadata = row["metadata"]
                except KeyError as error:
                    raise ValueError(
                        "Every training row needs complete domain_labels metadata."
                    ) from error
                if not isinstance(metadata, dict):
                    raise ValueError("Every training row needs complete domain_labels metadata.")
                values = metadata.get(self._options["field"])
                annotations.append(
                    [values]
                    if self._options["field"] != "domain_labels" and isinstance(values, str)
                    else values
                )
            if any(
                not isinstance(values, list)
                or not values
                or not all(_name(value) for value in values)
                or len(set(values)) != len(values)
                for values in annotations
            ):
                raise ValueError(
                    "Every training row needs a nonempty unique list[str] domain_labels."
                )
            observed = {name for values in annotations for name in values}
            names = self._options["domain_names"] or sorted(observed)
            if len(observed) != self.num_domains or set(names) != observed:
                raise ValueError(
                    "Metadata domains must exactly match the complete K-name vocabulary."
                )
            indices = {name: index for index, name in enumerate(names)}
            labels = torch.zeros(len(rows), self.num_domains)
            for index, values in enumerate(annotations):
                labels[index, [indices[name] for name in values]] = 1
            audit = {"recipe": "complete-semantic-metadata", "fitted": {}}
            identity = [(sample_id, sorted(values)) for sample_id, values in zip(ids, annotations)]
        if labels.unique(dim=0).shape[0] < 2:
            raise ValueError("Training domains must not collapse to constant multi-hot targets.")
        state = {
            "version": 1,
            "options": deepcopy(self._options),
            "sample_ids": ids,
            "domain_names": list(names),
            "labels": labels,
            "provenance": {
                **audit,
                "scope": "training-only",
                "training_size": len(ids),
                "training_sha256": hashlib.sha256(_json(identity).encode()).hexdigest(),
                "domain_counts": labels.sum(0).long().tolist(),
                "unicode_version": unicodedata.unidata_version,
            },
        }
        state["provenance"]["resource_sha256"] = _fingerprint(state)
        self.set_extra_state(state)

    def _topics(self, texts):
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from threadpoolctl import threadpool_limits

        p = self._options
        recipe = _tfidf_recipe(p)
        vectorizer = TfidfVectorizer(**recipe, dtype=np.float64)
        try:
            features = vectorizer.fit_transform(texts).tocsr()
        except ValueError as error:
            raise ValueError("Training text could not form a topic vocabulary.") from error
        if (features.getnnz(axis=1) == 0).any():
            raise ValueError("Every training text must have a nonempty topic feature vector.")
        features.sort_indices()
        signatures = {
            hashlib.sha256(
                features.indices[start:end].tobytes() + features.data[start:end].tobytes()
            ).digest()
            for start, end in zip(features.indptr[:-1], features.indptr[1:])
        }
        if len(signatures) < self.num_domains:
            raise ValueError("Topic domains require at least K distinct TF-IDF vectors.")
        clusters = KMeans(n_clusters=self.num_domains, random_state=p["seed"], n_init=10)
        with threadpool_limits(limits=1, user_api="openmp"):
            clusters.fit(features)
        similarities = torch.from_numpy(cosine_similarity(features, clusters.cluster_centers_))
        labels, primary = _select_topics(similarities, p["secondary_ratio"], p["max_labels"])
        counts = torch.bincount(primary, minlength=self.num_domains)
        if (counts == 0).any():
            raise ValueError("Topic adaptation requires all K cosine top-1 domains to be occupied.")
        return (
            labels,
            [f"topic_{index}" for index in range(self.num_domains)],
            {
                "recipe": {"tfidf": recipe, "kmeans": {"n_init": 10, "algorithm": "lloyd"}},
                "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
                "primary_counts": counts.tolist(),
                "fitted": {
                    "vocabulary": vectorizer.get_feature_names_out().tolist(),
                    "idf": torch.from_numpy(vectorizer.idf_.copy()),
                    "centroids": torch.from_numpy(clusters.cluster_centers_.copy()),
                },
            },
        )

    def targets(self, sample_ids, device=None):
        if not self._indices:
            raise RuntimeError("Prepare training domains or restore a checkpoint first.")
        if isinstance(sample_ids, (str, bytes)):
            raise ValueError("Domain targets require a sequence of training sample IDs.")
        try:
            indices = [self._indices[sample_id] for sample_id in sample_ids]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Domain targets only accept IDs from the prepared training split."
            ) from error
        return self._state["labels"][indices].to(device=device)

    def get_extra_state(self):
        return deepcopy(self._state)

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("options") != self._options
        ):
            raise ValueError("Domain resource version or configuration does not match.")
        ids, names, labels = state.get("sample_ids"), state.get("domain_names"), state.get("labels")
        provenance = state.get("provenance")
        if (
            not isinstance(ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(names, list)
            or not isinstance(labels, torch.Tensor)
            or labels.shape != (len(ids), self.num_domains)
            or labels.dtype != torch.float32
            or not torch.isfinite(labels).all()
            or not ((labels == 0) | (labels == 1)).all()
            or not isinstance(provenance, dict)
        ):
            raise ValueError("Domain resource IDs or multi-hot targets are invalid.")
        if ids:
            if (
                len(names) != self.num_domains
                or not all(isinstance(name, str) and name for name in names)
                or len(set(names)) != self.num_domains
                or (labels.sum(1) == 0).any()
                or (labels.sum(0) == 0).any()
                or labels.unique(dim=0).shape[0] < 2
                or provenance.get("scope") != "training-only"
                or provenance.get("resource_sha256") != _fingerprint(state)
            ):
                raise ValueError("Domain resource vocabulary, occupancy or fingerprint is invalid.")
        elif names or state.get("provenance"):
            raise ValueError("Unprepared domain resources must be empty.")
        self._state = deepcopy(state)
        self._state["labels"] = labels.detach().cpu().float().clone()
        self._indices = {sample_id: index for index, sample_id in enumerate(ids)}


class SingleDomainRouter(nn.Module):
    """Fit on training content, then route new text or explicit semantic domains.

    DomainTargets remains a training-ID supervision resource. Only its fitter
    and numeric state are reused here; inference never looks up training IDs.
    """

    def __init__(
        self,
        mode="topics",
        num_domains=9,
        language="en",
        seed=3074,
        max_features=20000,
        ngram_range=(1, 2),
        domain_names=None,
    ):
        super().__init__()
        self._options = dict(
            mode=mode,
            num_domains=num_domains,
            language=language,
            seed=seed,
            max_features=max_features,
            ngram_range=list(ngram_range)
            if isinstance(ngram_range, (list, tuple))
            else ngram_range,
            domain_names=deepcopy(domain_names),
        )
        fitter = self._fitter()  # Reuse all domain configuration validation.
        self.num_domains = num_domains
        self._state = fitter.get_extra_state()
        self._vectorizer = None

    def _fitter(self):
        return DomainTargets(**self._options, max_labels=1)

    @property
    def domain_names(self):
        return list(self._state["domain_names"])

    @property
    def provenance(self):
        return deepcopy(self._state["provenance"])

    @staticmethod
    def _annotations(metadata):
        if not isinstance(metadata, (list, tuple)) or not metadata:
            raise ValueError("Domain routing needs a nonempty metadata sequence.")
        names = []
        for item in metadata:
            if not isinstance(item, dict) or not _name(item.get("domain")):
                raise ValueError("Domain routing requires one semantic string metadata.domain.")
            names.append(item["domain"])
        return names

    def prepare(self, train_rows):
        rows = list(train_rows)
        if self._options["mode"] == "metadata":
            try:
                names = self._annotations([row["metadata"] for row in rows])
            except KeyError as error:
                raise ValueError("Domain routing requires complete metadata.domain.") from error
            rows = [
                {"sample_id": row["sample_id"], "metadata": {"domain_labels": [name]}}
                for row, name in zip(rows, names)
            ]
        fitter = self._fitter()
        fitter.prepare(rows)
        resource = fitter.get_extra_state()
        resource["provenance"]["routing"] = self._protocol
        resource["provenance"]["resource_sha256"] = _fingerprint(resource)
        self.set_extra_state({"version": 1, "options": self._options, "resource": resource})

    @property
    def _protocol(self):
        return (
            "frozen-tfidf-cosine-top1-v1"
            if self._options["mode"] == "topics"
            else "explicit-semantic-single-domain-v1"
        )

    def route(self, batch, device=None):
        if not self._state["sample_ids"]:
            raise RuntimeError("Prepare a domain router or restore its checkpoint first.")
        if self._options["mode"] == "metadata":
            try:
                names = self._annotations(batch["metadata"])
            except KeyError as error:
                raise ValueError("Domain routing requires metadata.domain at inference.") from error
            indices = {name: index for index, name in enumerate(self.domain_names)}
            if any(name not in indices for name in names):
                raise ValueError("Unknown metadata.domain outside the fitted training vocabulary.")
            return torch.tensor([indices[name] for name in names], dtype=torch.long, device=device)

        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = batch["text"]
        if (
            not isinstance(texts, (list, tuple))
            or not texts
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ValueError("Topic routing requires a nonempty sequence of nonempty text.")
        fitted = self._state["provenance"]["fitted"]
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                **_tfidf_recipe(self._options),
                vocabulary={word: index for index, word in enumerate(fitted["vocabulary"])},
                dtype=np.float64,
            )
            self._vectorizer.idf_ = fitted["idf"].numpy().copy()
        features = self._vectorizer.transform(
            [unicodedata.normalize("NFC", text) for text in texts]
        )
        if (features.getnnz(axis=1) == 0).any():
            raise ValueError("Topic routing cannot route empty or all-OOV TF-IDF vectors.")
        similarities = cosine_similarity(features, fitted["centroids"].numpy())
        return torch.from_numpy(similarities.argmax(axis=1)).long().to(device=device)

    def get_extra_state(self):
        return {"version": 1, "options": deepcopy(self._options), "resource": deepcopy(self._state)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "options", "resource"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or state["options"] != self._options
        ):
            raise ValueError("Domain router version or configuration does not match.")
        resource = state["resource"]
        if not isinstance(resource, dict) or set(resource) != {
            "version",
            "options",
            "sample_ids",
            "domain_names",
            "labels",
            "provenance",
        }:
            raise ValueError("Domain router resource schema is invalid.")
        fitter = self._fitter()
        try:
            fitter.set_extra_state(resource)
        except (KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError(f"Domain router resource is invalid: {error}") from error
        resource = fitter.get_extra_state()
        ids, labels, audit = resource["sample_ids"], resource["labels"], resource["provenance"]
        if ids:
            counts = labels.sum(0).long().tolist()
            if (
                ids != sorted(ids)
                or not labels.sum(-1).eq(1).all()
                or audit.get("routing") != self._protocol
                or audit.get("training_size") != len(ids)
                or audit.get("domain_counts") != counts
                or not isinstance(audit.get("training_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", audit["training_sha256"])
            ):
                raise ValueError(
                    "Domain router training identity or single-domain targets are invalid."
                )
            self._validate_fitted(resource)
        self._state = resource
        self._vectorizer = None

    def _validate_fitted(self, resource):
        audit, names = resource["provenance"], resource["domain_names"]
        fitted = audit.get("fitted")
        if self._options["mode"] == "metadata":
            expected = self._options["domain_names"] or sorted(names)
            if (
                names != expected
                or not all(_name(name) for name in names)
                or fitted != {}
                or audit.get("recipe") != "complete-semantic-metadata"
            ):
                raise ValueError("Domain router semantic metadata vocabulary is invalid.")
            return
        if (
            not isinstance(fitted, dict)
            or set(fitted) != {"vocabulary", "idf", "centroids"}
            or audit.get("recipe")
            != {
                "tfidf": _tfidf_recipe(self._options),
                "kmeans": {"n_init": 10, "algorithm": "lloyd"},
            }
            or audit.get("primary_counts") != audit["domain_counts"]
            or names != [f"topic_{index}" for index in range(self.num_domains)]
        ):
            raise ValueError("Domain router fitted topic recipe is invalid.")
        vocabulary, idf, centers = (fitted[key] for key in ("vocabulary", "idf", "centroids"))
        if (
            not isinstance(vocabulary, list)
            or not vocabulary
            or len(vocabulary) > self._options["max_features"]
            or any(not isinstance(word, str) or not word for word in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or not isinstance(idf, torch.Tensor)
            or idf.dtype != torch.float64
            or idf.shape != (len(vocabulary),)
            or not torch.isfinite(idf).all()
            or (idf < 1).any()
            or not isinstance(centers, torch.Tensor)
            or centers.dtype != torch.float64
            or centers.shape != (self.num_domains, len(vocabulary))
            or not torch.isfinite(centers).all()
            or (centers < 0).any()
            or centers.norm(dim=-1).eq(0).any()
            or (centers / centers.norm(dim=-1, keepdim=True)).unique(dim=0).shape[0]
            != self.num_domains
        ):
            raise ValueError("Domain router vocabulary, IDF or centroids are invalid.")
        fitted["idf"] = idf.detach().cpu().clone()
        fitted["centroids"] = centers.detach().cpu().clone()
