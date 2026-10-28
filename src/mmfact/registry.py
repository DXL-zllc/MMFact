"""Load built-in models by name, with optional third-party modules."""

import importlib
import keyword
import pkgutil
import re

import mmfact.models_classification
from mmfact.base import AbstractDetector

_MODELS: dict[str, type[AbstractDetector]] = {}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MODEL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


def _model_key(name: str) -> str:
    if not isinstance(name, str) or not _MODEL_NAME.fullmatch(name):
        raise ValueError("Model name must be an identifier, optionally with hyphens.")
    key = name.lower().replace("-", "_")
    if keyword.iskeyword(key):
        raise ValueError("Model name must be an identifier, optionally with hyphens.")
    return key


def _module_name(module: str) -> str:
    if not isinstance(module, str) or not all(
        _IDENTIFIER.fullmatch(part) and not keyword.iskeyword(part) for part in module.split(".")
    ):
        raise ValueError("model.module must be a dotted Python module name.")
    return module


def register_model(name: str):
    """Register a detector class under a case-insensitive experiment name."""
    key = _model_key(name)

    def decorate(cls: type[AbstractDetector]):
        if not isinstance(cls, type) or not issubclass(cls, AbstractDetector):
            raise TypeError("Registered models must extend AbstractDetector.")
        if key in _MODELS and _MODELS[key] is not cls:
            raise ValueError(f"Model {name!r} is already registered.")
        _MODELS[key] = cls
        return cls

    return decorate


def list_models() -> list[str]:
    """List built-in module names and registered adapters without loading backbones."""
    builtins = {
        _model_key(module.name)
        for module in pkgutil.iter_modules(mmfact.models_classification.__path__)
        if not module.name.startswith("_")
        and _IDENTIFIER.fullmatch(module.name)
        and not keyword.iskeyword(module.name)
    }
    return sorted(builtins | _MODELS.keys())


def get_model_class(name: str, module: str | None = None) -> type[AbstractDetector]:
    key = _model_key(name)
    import_error = None
    if module is not None or key not in _MODELS:
        target = (
            _module_name(module) if module is not None else f"mmfact.models_classification.{key}"
        )
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as error:
            # A missing adapter is different from a missing dependency inside
            # an existing adapter, which should retain its original traceback.
            if error.name != target:
                raise
            import_error = error
    if import_error is not None or key not in _MODELS:
        available = ", ".join(list_models()) or "none (framework scaffold)"
        raise ValueError(
            f"Model {name!r} is not implemented/registered. Available: {available}. "
            f"Add mmfact.models_classification.{key} with a registered AbstractDetector, "
            "or set model.module for an external implementation. "
            "Use --dry-run to inspect data/configuration without a model."
        ) from import_error
    return _MODELS[key]
