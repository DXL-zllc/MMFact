"""Metrics for a fixed real=0, fake=1 label space, including failed predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import groupby
from numbers import Integral, Real
from statistics import mean, stdev
from typing import Any


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int | None],
    score_fake: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    """Compute binary classification metrics over every example, including failures.

    ``None`` means a prediction failed. A failure counts as incorrect for
    accuracy and as a false negative for the true class; it is not a false
    positive for either class. Macro F1 always averages the two fixed classes,
    even when a class is absent. Undefined precision/recall/F1 are zero.

    ``confusion_matrix`` has two rows (true real, true fake) and three columns
    (predicted real, predicted fake, prediction failure). Quality metrics and
    ``failure_rate`` are percentages in [0, 100], including ROC-AUC, equal error
    rate (EER) and average precision (AP, not trapezoidal PR-AUC). EER linearly
    interpolates the ROC intersection where FPR equals FNR; it does not change
    the supplied predictions or their classification threshold. MCC and kappa use
    [-1, 1]; log loss uses natural logarithms and Brier score uses the fake probability's squared
    error. Supports and ``num_samples`` are counts. The normalized confusion
    matrix contains row percentages, including the failure column.

    Probability metrics require every prediction and fake probability; otherwise
    they are ``None``. Single-class truth makes ROC-AUC, EER, AP and balanced accuracy
    undefined (``None``). Failures make MCC/kappa undefined; a degenerate MCC
    denominator gives zero, while undefined kappa is ``None``. Empty input,
    invalid labels and nonfinite or out-of-range probabilities raise ``ValueError``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    matrix = [[0, 0, 0], [0, 0, 0]]
    for index, (truth, prediction) in enumerate(zip(y_true, y_pred)):
        if not isinstance(truth, Integral) or isinstance(truth, bool) or truth not in (0, 1):
            raise ValueError(f"y_true[{index}] must be integer 0 (real) or 1 (fake)")
        if prediction is not None and (
            not isinstance(prediction, Integral)
            or isinstance(prediction, bool)
            or prediction not in (0, 1)
        ):
            raise ValueError(f"y_pred[{index}] must be integer 0, integer 1, or None")
        column = 2 if prediction is None else int(prediction)
        matrix[int(truth)][column] += 1

    count = len(y_true)
    result: dict[str, Any] = {
        "accuracy": 100.0 * (matrix[0][0] + matrix[1][1]) / count,
    }
    supports = [sum(row) for row in matrix]
    class_scores = []
    for label, name in ((0, "real"), (1, "fake")):
        true_positive = matrix[label][label]
        predicted_count = matrix[0][label] + matrix[1][label]
        true_count = supports[label]
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / true_count if true_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"precision_{name}"] = 100.0 * precision
        result[f"recall_{name}"] = 100.0 * recall
        result[f"f1_{name}"] = 100.0 * f1
        result[f"support_{name}"] = true_count
        class_scores.append((precision, recall, f1))
    for index, metric in enumerate(("precision", "recall", "f1")):
        result[f"macro_{metric}"] = 100.0 * sum(row[index] for row in class_scores) / 2
        result[f"weighted_{metric}"] = 100.0 * sum(
            support * row[index] for support, row in zip(supports, class_scores)
        ) / count
    result["balanced_accuracy"] = result["macro_recall"] if all(supports) else None
    failures = matrix[0][2] + matrix[1][2]
    correct = matrix[0][0] + matrix[1][1]
    micro_precision = correct / (count - failures) if count > failures else 0.0
    micro_recall = correct / count
    result["micro_precision"] = 100.0 * micro_precision
    result["micro_recall"] = 100.0 * micro_recall
    result["micro_f1"] = (
        200.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    result["failure_rate"] = 100.0 * failures / count
    result["mcc"] = None
    result["cohen_kappa"] = None
    if not failures:
        tn, fp = matrix[0][:2]
        fn, tp = matrix[1][:2]
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        result["mcc"] = (tp * tn - fp * fn) / denominator if denominator else 0.0
        expected = supports[0] * (tn + fn) + supports[1] * (tp + fp)
        result["cohen_kappa"] = (
            ((tp + tn) * count - expected) / (count * count - expected)
            if expected != count * count else None
        )
    result.update(_probability_metrics(y_true, score_fake, bool(failures)))
    result["num_samples"] = count
    result["confusion_matrix"] = matrix
    result["confusion_matrix_normalized"] = [
        [100.0 * value / support if support else 0.0 for value in row]
        for row, support in zip(matrix, supports)
    ]
    return result


def _probability_metrics(
    y_true: Sequence[int], score_fake: Sequence[float | None] | None, failures: bool
) -> dict[str, float | None]:
    result = dict.fromkeys((
        "roc_auc", "eer", "average_precision_fake", "average_precision_real",
        "macro_average_precision", "log_loss", "brier_score",
    ))
    if score_fake is None:
        return result
    if len(score_fake) != len(y_true):
        raise ValueError("score_fake and y_true must have the same length")
    for index, score in enumerate(score_fake):
        if score is not None and (
            not isinstance(score, Real) or isinstance(score, bool)
            or not math.isfinite(score) or not 0.0 <= score <= 1.0
        ):
            raise ValueError(f"score_fake[{index}] must be a finite probability in [0, 1] or None")
    if failures or any(score is None for score in score_fake):
        return result
    scores = [float(score) for score in score_fake]
    epsilon = math.ulp(1.0)
    clipped = [min(max(score, epsilon), 1.0 - epsilon) for score in scores]
    result["log_loss"] = -mean(
        math.log(score if truth == 1 else 1.0 - score)
        for truth, score in zip(y_true, clipped)
    )
    result["brier_score"] = mean((score - truth) ** 2 for truth, score in zip(y_true, scores))
    if len(set(y_true)) == 2:
        auc, ap_fake, eer = _ranking_metrics(y_true, scores)
        _, ap_real, _ = _ranking_metrics([1 - truth for truth in y_true], [1 - p for p in scores])
        result.update(
            roc_auc=100.0 * auc,
            eer=100.0 * eer,
            average_precision_fake=100.0 * ap_fake,
            average_precision_real=100.0 * ap_real,
            macro_average_precision=50.0 * (ap_fake + ap_real),
        )
    return result


def _ranking_metrics(
    y_true: Sequence[int], scores: Sequence[float],
) -> tuple[float, float, float]:
    """Return ROC-AUC, threshold-based AP and interpolated EER, grouping ties."""
    total_positive = sum(y_true)
    total_negative = len(y_true) - total_positive
    tp = fp = 0
    auc = ap = 0.0
    previous_fpr, previous_fnr = 0.0, 1.0
    eer = 0.0
    crossed = False
    for _, group in groupby(sorted(zip(scores, y_true), reverse=True), key=lambda row: row[0]):
        labels = [label for _, label in group]
        positive = sum(labels)
        negative = len(labels) - positive
        tp += positive
        fp += negative
        auc += positive * (total_negative - fp + 0.5 * negative)
        ap += positive * tp / (tp + fp)
        fpr, fnr = fp / total_negative, 1.0 - tp / total_positive
        if not crossed and fpr >= fnr:
            weight = (previous_fnr - previous_fpr) / (
                fpr - previous_fpr + previous_fnr - fnr
            )
            eer = previous_fpr + weight * (fpr - previous_fpr)
            crossed = True
        previous_fpr, previous_fnr = fpr, fnr
    return auc / (total_positive * total_negative), ap / total_positive, eer


def aggregate_metrics(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Summarize repeated runs with means and sample standard deviations.

    Numeric or null metric fields are aggregated, except counts and matrices.
    Any null run makes that metric's mean and standard deviation null; runs are
    never silently omitted. A single valid run has standard deviation zero.
    Every run must expose the same scalar metric fields; other metadata is
    ignored. Values retain their original units.
    """
    if not results:
        raise ValueError("At least one result is required to aggregate metrics")
    excluded = {"num_samples", "confusion_matrix", "confusion_matrix_normalized"}
    numeric_results = []
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise ValueError(f"Result {index} must be a mapping")
        numeric = {
            key: None if value is None else float(value)
            for key, value in result.items()
            if key not in excluded and not key.startswith("support_")
            and (value is None or isinstance(value, Real) and not isinstance(value, bool))
        }
        if any(value is not None and not math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"Result {index} contains a nonfinite metric")
        if numeric_results and numeric.keys() != numeric_results[0].keys():
            raise ValueError("All runs must contain the same numeric metric fields")
        numeric_results.append(numeric)
    if not numeric_results[0]:
        raise ValueError("No numeric metrics are available to aggregate")
    aggregated = {}
    for key in numeric_results[0]:
        values = [result[key] for result in numeric_results]
        aggregated[key] = (
            {"mean": None, "std": None}
            if any(value is None for value in values)
            else {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}
        )
    return aggregated
