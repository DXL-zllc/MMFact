"""Public experiment entry point."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def quick_start(
    model: str = "EANN",
    dataset: str = "weibo17",
    config_files: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    *,
    dry_run: bool = False,
    mode: str = "train",
    checkpoint: str | Path | None = None,
    input_mode: str | None = None,
    materials_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train, evaluate, or prepare one model on a selected dataset."""
    from mmfact.runner import quick_start as run

    return run(
        model,
        dataset,
        config_files,
        overrides,
        dry_run=dry_run,
        mode=mode,
        checkpoint=checkpoint,
        input_mode=input_mode,
        materials_root=materials_root,
    )
