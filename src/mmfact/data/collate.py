"""Batch raw examples; tokenization and image processing belong to models."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def collate_news(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect raw strings/metadata and a one-dimensional integer label tensor."""

    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "text": [sample["text"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "metadata": [sample["metadata"] for sample in samples],
    }
