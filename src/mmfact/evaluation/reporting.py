"""Per-class and domain-sliced reports for reproducible tables and heatmaps."""

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import aggregate_metrics, classification_metrics

GROUP_FIELDS = ("domain", "domain_3", "domain_4", "domains")
CLASS_NAMES = ("Real", "Fake")
PREDICTION_NAMES = (*CLASS_NAMES, "Failure")


def _metrics(rows):
    return classification_metrics(
        [row["label"] for row in rows],
        [row["prediction"] for row in rows],
        [row.get("score_fake") for row in rows],
    )


def classification_breakdown(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group the same binary predictions, without treating domains as targets.

    Every available grouping is reported for every model. Multi-domain examples
    contribute once to each assigned domain. A null group retains missing labels;
    an entirely unavailable grouping is omitted. Overall examples are never
    duplicated. Both Real and Fake remain in every report, including absent classes.
    """
    reports = [{"grouping": "overall", "group": None, "metrics": _metrics(rows)}]
    for field in GROUP_FIELDS:
        if not any(field in row for row in rows):
            continue
        groups = defaultdict(list)
        for row in rows:
            value = row.get(field)
            values = value if field == "domains" and isinstance(value, list) else [value]
            if field == "domains" and value is not None and not isinstance(value, list):
                raise ValueError("domains must be a list of domain names or null")
            if any(item is not None and (not isinstance(item, str) or not item.strip())
                   for item in values):
                raise ValueError(f"{field} must contain nonempty domain names")
            for group in set(values) or {None}:
                groups[group].append(row)
        for group in sorted(groups, key=lambda value: (value is None, value or "")):
            reports.append({"grouping": field, "group": group, "metrics": _metrics(groups[group])})
    return reports


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def write_breakdown(directory, split, rows, *, model, dataset, seed):
    """Write final-checkpoint reports; no extra inference or API requests."""
    directory = Path(directory)
    context = {"model": model, "dataset": dataset, "seed": seed}
    groups = classification_breakdown(rows)
    report = {
        **context, "split": split, "true_classes": CLASS_NAMES,
        "predicted_classes": PREDICTION_NAMES, "groups": groups,
    }
    _write_json(directory / f"{split}_breakdown.json", report)
    by_group, per_class, confusion = [], [], []
    for group in groups:
        identity = {**context, "grouping": group["grouping"], "group": group["group"]}
        metrics = group["metrics"]
        by_group.append({**identity, **{
            key: value for key, value in metrics.items()
            if value is None or isinstance(value, (int, float))
        }})
        for label, name in enumerate(CLASS_NAMES):
            suffix = name.lower()
            per_class.append({
                **identity, "label": label, "class": name,
                **{metric: metrics[f"{metric}_{suffix}"]
                   for metric in ("precision", "recall", "f1", "support")},
            })
            for predicted, prediction_name in enumerate(PREDICTION_NAMES):
                confusion.append({
                    **identity, "true_class": name, "predicted_class": prediction_name,
                    "count": metrics["confusion_matrix"][label][predicted],
                    "row_percent": metrics["confusion_matrix_normalized"][label][predicted],
                })
    for suffix, table in (("by_group", by_group), ("per_class", per_class), ("confusion", confusion)):
        _write_csv(directory / f"{split}_{suffix}.csv", table)
    return report


def write_breakdown_summary(directory, split, reports):
    """Aggregate matching slices across seeds, keeping count and matrix axes explicit."""
    if not reports or any(report["split"] != split for report in reports):
        raise ValueError("Expected nonempty breakdown reports for the requested split")
    first = reports[0]
    seeds = [report["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Breakdown reports must have unique seeds")
    keys = [(group["grouping"], group["group"]) for group in first["groups"]]
    for report in reports:
        if any(report[key] != first[key] for key in ("model", "dataset", "split")):
            raise ValueError("Breakdown model, dataset and split must match across seeds")
        if [(group["grouping"], group["group"]) for group in report["groups"]] != keys:
            raise ValueError("Breakdown groups must match across seeds")
    context = {key: first[key] for key in ("model", "dataset")}
    groups, table, confusion = [], [], []
    for index, (grouping, group) in enumerate(keys):
        metrics = [report["groups"][index]["metrics"] for report in reports]
        counts = {key: metrics[0][key] for key in ("num_samples", "support_real", "support_fake")}
        if any(any(item[key] != count for key, count in counts.items()) for item in metrics):
            raise ValueError("Breakdown sample counts must match across seeds")
        aggregate = aggregate_metrics(metrics)
        identity = {**context, "grouping": grouping, "group": group}
        groups.append({"grouping": grouping, "group": group, **counts, "metrics": aggregate})
        table.append({**identity, **counts, **{
            f"{key}_{stat}": value
            for key, stats in aggregate.items() for stat, value in stats.items()
        }})
        for label, name in enumerate(CLASS_NAMES):
            for predicted, prediction_name in enumerate(PREDICTION_NAMES):
                cells = aggregate_metrics([
                    {"count": item["confusion_matrix"][label][predicted],
                     "row_percent": item["confusion_matrix_normalized"][label][predicted]}
                    for item in metrics
                ])
                confusion.append({
                    **identity, "true_class": name, "predicted_class": prediction_name,
                    **{f"{key}_{stat}": value
                       for key, stats in cells.items() for stat, value in stats.items()},
                })
    directory = Path(directory)
    _write_json(directory / f"{split}_breakdown_summary.json", {
        **context, "split": split, "seeds": seeds, "true_classes": CLASS_NAMES,
        "predicted_classes": PREDICTION_NAMES, "groups": groups,
    })
    _write_csv(directory / f"{split}_by_group_summary.csv", table)
    _write_csv(directory / f"{split}_confusion_summary.csv", confusion)
