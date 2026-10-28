"""Resolve explicit shared material views without changing method training policies."""

from copy import deepcopy
from pathlib import Path

import yaml

from mmfact import configs
from mmfact.config import ConfigError, _resolve_encoder_presets


def apply_encoding_profile(config, profile="shared_v1"):
    """Return a resolved copy; only listed, audited semantic-vector roles change.

    Apply before localizing model paths. Save this returned configuration for
    both preprocessing and later experiments so consumers use the same recipe.
    Passing ``None`` explicitly preserves the ordinary model configuration.
    """
    result = deepcopy(config)
    if profile is None:
        return result
    path = Path(configs.__file__).parent / "material_encoders.yaml"
    definitions = yaml.safe_load(path.read_text("utf-8"))
    if not isinstance(profile, str) or profile not in definitions:
        raise ConfigError(f"Unknown material encoding profile: {profile!r}")
    definition = definitions[profile]
    dataset, model = result["dataset"]["name"], result["model"]["name"]
    if dataset not in definition["datasets"]:
        raise ConfigError(f"Material profile {profile!r} does not support {dataset!r}.")
    specification = definition["document"]
    paths = specification["models"].get(model, ())
    if not paths:
        return result
    replacement = _resolve_encoder_presets(
        {"encoder_preset": specification["encoder_preset"]}, dataset
    )
    for dotted in paths:
        branch = result["model"]["parameters"]
        for part in dotted.split("."):
            if part not in branch or not isinstance(branch[part], dict):
                raise ConfigError(f"Material profile requires {model}.{dotted}.")
            branch = branch[part]
        if branch.get("output", "pooled") not in {"pooled", "cls"}:
            raise ConfigError(f"Refusing to replace non-vector text role {model}.{dotted}.")
        branch.pop("encoder_preset", None)
        branch.update(replacement)
    result["model"].setdefault("provenance", {})["material_encoding"] = {
        "profile": profile,
        "roles": list(paths),
        "document": replacement,
        "adaptation": definition["description"],
    }
    return result
