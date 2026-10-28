"""Validated, model-independent image-text classification data."""

from .collate import collate_news
from .dataset import NewsDataset, load_datasets

__all__ = ["NewsDataset", "collate_news", "load_datasets"]
