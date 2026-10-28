"""Detector hooks consumed by the shared training and evaluation lifecycle."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True)
class TrainingStage:
    """One complete fit/selection/restore cycle; settings may share a YAML key."""

    name: str
    settings: str | None = None
    module: str | None = None
    heads: tuple[str, ...] = ()
    mode: str = "gradient"


class AbstractDetector(nn.Module, ABC):
    """Own model-specific processors, encoders, fusion and losses in one adapter.

    Batches contain raw text and image paths, labels, sample IDs and metadata.
    Tensor fields are moved to the configured device by the trainer. A model may
    tokenize text, decode images or read its own versioned feature cache.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = config

    def prepare(self, train_dataset) -> None:
        """Fit training-only resources, such as vocabulary or event clusters.

        Called once before optimization. Learned preprocessing must be included
        in state_dict (for example through get_extra_state/set_extra_state).
        Validation and test data are never passed to this hook.
        """

    @abstractmethod
    def calculate_loss(self, batch: Mapping[str, Any]) -> Tensor | Mapping[str, Tensor] | None:
        """Return a scalar loss, or named weighted scalar terms to be summed.

        For a mapping, do not also include a total loss: every entry contributes
        once. Auxiliary losses and their weights belong to the model adapter.
        Return None when a batch is inapplicable to the current training phase.
        """

    @abstractmethod
    def predict(self, batch: Mapping[str, Any]) -> Tensor:
        """Return finite logits of shape [batch_size, 2], ordered [Real, Fake]."""

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Optional hook for model-specific loss schedules; epoch starts at 1."""

    def training_stages(self) -> tuple[TrainingStage, ...]:
        """Ordered complete fits, distinct from alternating per-epoch phases."""
        return (TrainingStage("detection"),)

    def set_training_stage(self, stage: str | None) -> None:
        """Select the active head and trainable parameters; None means inference."""

    def training_stage_data(self, stage: str, train_data, val_data, options):
        """Build stage loaders; a declared data fallback may adjust selection options."""
        return train_data, val_data

    def fit_stage(self, stage: str, train_dataset, val_loader, options) -> dict:
        """Fit once without torch gradients; return a finite JSON-compatible report."""
        raise NotImplementedError("A fit stage requires fit_stage.")

    def validation_loss(self, batch: Mapping[str, Any]) -> Tensor | Mapping[str, Tensor]:
        """Evaluation-only objective for stages explicitly selected by loss."""
        return self.calculate_loss(batch)

    def predict_heads(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        """Validation logits keyed by each declared independent stage module path."""
        raise NotImplementedError("A stage with heads requires predict_heads.")

    def training_phases(self) -> tuple[str, ...]:
        """Ordered tasks, applied per pass or per batch according to the interval."""
        return ("detection",)

    def training_phase_interval(self) -> str:
        """Use 'epoch' for complete passes, or 'batch' for sequential batch tasks."""
        return "epoch"

    def optimizer_parameter_groups(self) -> Mapping[str, Iterable[nn.Parameter]]:
        """Partition all trainable parameters; numeric options stay in training."""
        return {
            "default": (parameter for parameter in self.parameters() if parameter.requires_grad)
        }

    def set_training_phase(self, phase: str) -> None:
        """Select a loss/gradient path; optimizer parameters stay registered."""

    def loss_weight(self, batch: Mapping[str, Any]) -> int:
        """Number of examples represented by a mean loss, for epoch logging."""
        return len(batch["sample_id"])

    def on_epoch_end(self, epoch: int) -> None:
        """Optional hook for model-specific state updates."""
