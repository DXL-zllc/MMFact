"""Run a real official VinVL AttrRCNN in its own installed environment over JSON stdin.

This standalone adapter needs no MMFact installation in the legacy detector environment.
Official API reference: microsoft/scene_graph_benchmark@8e14944f3cc9f178b1c5626482fad509068f1395,
tools/demo/{demo_image,detect_utils}.py (MIT). This adapter is independently written.
"""

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REFERENCE = "8e14944f3cc9f178b1c5626482fad509068f1395"


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def dense_labels(detections, labels, attributes, settings):
    """Deterministically retain confidence-qualified object/attribute names, not new captions."""
    expected = {"object_threshold", "attribute_threshold", "max_objects"}
    if (
        not isinstance(settings, dict)
        or set(settings) != expected
        or type(settings["max_objects"]) is not int
        or settings["max_objects"] < 1
        or any(
            type(settings[key]) not in (int, float)
            or not math.isfinite(settings[key])
            or not 0 <= settings[key] <= 1
            for key in ("object_threshold", "attribute_threshold")
        )
    ):
        raise ValueError("NLIN VinVL settings are invalid.")
    selected = sorted(
        (row for row in detections if row["conf"] >= settings["object_threshold"]),
        key=lambda row: (-row["conf"], row["class"], tuple(row["rect"])),
    )[: settings["max_objects"]]
    result = []
    for row in selected:
        if "attr" not in row or "attr_conf" not in row or len(row["attr"]) != len(row["attr_conf"]):
            raise ValueError("VinVL prediction is missing real attribute labels or confidences.")
        qualified = sorted(
            (
                (score, attributes[index])
                for index, score in zip(row["attr"], row["attr_conf"])
                if score >= settings["attribute_threshold"]
            ),
            key=lambda item: (-item[0], item[1]),
        )
        result.append(
            {
                "object": labels[row["class"]],
                "attributes": list(dict.fromkeys(item[1] for item in qualified)),
            }
        )
    return result


def infer(request, repository, config_file, weights, device="cuda", revision=REFERENCE):
    if not isinstance(request, dict) or set(request) != {"image_path", "image_sha256", "tool"}:
        raise ValueError("VinVL adapter accepts only the image and declared tool identity.")
    if request["image_sha256"] != _digest(request["image_path"]):
        raise ValueError("VinVL adapter image digest mismatch.")
    tool = request["tool"]
    if tool.get("name") != "VinVL" or Path(weights).name != tool.get("weights"):
        raise ValueError("VinVL adapter requires the declared actual detector weights.")
    repository = Path(repository).resolve()
    actual_revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != revision:
        raise ValueError("VinVL installed source differs from the selected fixed revision.")
    dirty = subprocess.run(
        ["git", "-C", str(repository), "diff", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if dirty or any(Path(path).suffix in {".py", ".pyi"} for path in untracked):
        raise ValueError("VinVL source must have no tracked edits or untracked Python modules.")
    sys.path.insert(0, str(repository))
    # Keep stdout exclusively for the response; legacy imports/checkpointers may print logs.
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        from maskrcnn_benchmark.config import cfg
        from maskrcnn_benchmark.data.transforms import build_transforms
        from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
        from scene_graph_benchmark.AttrRCNN import AttrRCNN
        from scene_graph_benchmark.config import sg_cfg
        from tools.demo.detect_utils import detect_objects_on_single_image

        cfg = cfg.clone()
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        cfg.merge_from_file(str(Path(config_file).resolve()))
        cfg.merge_from_list(["MODEL.DEVICE", device, "MODEL.WEIGHT", str(Path(weights).resolve())])
        cfg.freeze()
        if cfg.MODEL.META_ARCHITECTURE != "AttrRCNN" or not cfg.MODEL.ATTRIBUTE_ON:
            raise ValueError("NLIN needs VinVL's actual attribute-capable AttrRCNN configuration.")
        labelmap = Path(cfg.DATASETS.LABELMAP_FILE)
        if not labelmap.is_absolute():
            labelmap = Path(cfg.DATA_DIR) / labelmap
        if not labelmap.is_absolute():
            labelmap = repository / labelmap
        with labelmap.open(encoding="utf-8") as stream:
            names = json.load(stream)
        labels = {value: key for key, value in names["label_to_idx"].items()}
        attributes = {value: key for key, value in names["attribute_to_idx"].items()}
        network = AttrRCNN(cfg).to(cfg.MODEL.DEVICE).eval()
        DetectronCheckpointer(cfg, network).load(cfg.MODEL.WEIGHT)
        image = cv2.imread(request["image_path"])
        if image is None:
            raise ValueError("VinVL could not decode the actual source image.")
        detections = detect_objects_on_single_image(
            network, build_transforms(cfg, is_train=False), image
        )
    return {
        "tool": tool,
        "objects": dense_labels(detections, labels, attributes, tool["settings"]),
        "weights_sha256": _digest(weights),
        "adapter_sha256": _digest(__file__),
        "raw": {
            "detections": detections,
            "repository_commit": actual_revision,
            "config": cfg.dump(),
            "config_sha256": _digest(config_file),
            "labelmap_sha256": _digest(labelmap),
            "image_sha256": request["image_sha256"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default=REFERENCE)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    response = infer(
        request, args.repository, args.config, args.weights, args.device, args.revision
    )
    json.dump(response, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
