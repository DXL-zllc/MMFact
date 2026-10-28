"""Small provenance and input checks shared by the two parameterless baselines."""

import hashlib
import json
import re

from mmfact.data.features import _value_digest, fingerprint


def _training_data(dataset, *, text=False):
    texts, labels, ids = [], [], set()
    digest = hashlib.sha256()
    for row in dataset:
        sample_id, label = row["sample_id"], row["label"]
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in ids:
            raise ValueError("Training sample IDs must be nonempty and unique.")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Training labels must be integer Real=0 or Fake=1.")
        ids.add(sample_id)
        labels.append(label)
        identity = [sample_id, label]
        if text:
            value = row["text"]
            if not isinstance(value, str):
                raise ValueError("Training text must be a string.")
            texts.append(value)
            identity.append(value)
        digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    if not labels:
        raise ValueError("A fit stage requires nonempty training data.")
    return texts, labels, digest.hexdigest()


def _fit_seed(stage, options):
    if stage != "fit":
        raise ValueError("Classical baselines have only the fit stage.")
    seed = options["seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("The fit seed must be an integer in [0, 2**32-1].")
    return seed


def _checksum(resource):
    return _value_digest({key: value for key, value in resource.items() if key != "sha256"})


def _validate_identity(resource, recipe):
    try:
        matches = fingerprint(resource["recipe"]) == fingerprint(recipe)
    except (TypeError, ValueError) as error:
        raise ValueError("Classical checkpoint recipe must contain finite basic values.") from error
    if resource["ready"] is not True or not matches:
        raise ValueError("Classical checkpoint recipe or readiness is invalid.")
    count, counts = resource["num_samples"], resource["class_counts"]
    if (
        type(count) is not int
        or count < 1
        or not isinstance(counts, list)
        or len(counts) != 2
        or any(type(value) is not int or value < 0 for value in counts)
        or sum(counts) != count
    ):
        raise ValueError("Classical checkpoint training counts are invalid.")
    if any(
        not isinstance(resource[key], str) or not re.fullmatch(r"[0-9a-f]{64}", resource[key])
        for key in ("training_sha256", "sha256")
    ):
        raise ValueError("Classical checkpoint training identity or checksum is invalid.")
    if resource["sha256"] != _checksum(resource):
        raise ValueError("Classical checkpoint checksum does not match its resources.")
