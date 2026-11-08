"""Build report figures and source metrics from local TensorBoard logs and submissions.

Requires matplotlib in addition to project dependencies. Run from repository root.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.expeiment_to_md import loss_at
from src.rle import rle_decode


def collect_runs(outputs: Path) -> dict:
    runs = {}
    for config in sorted(outputs.glob("*/.hydra/config.yaml")):
        cfg = OmegaConf.load(config)
        run = config.parent.parent
        for logdir in sorted({path.parent for path in run.rglob("*tfevents*")}):
            accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalars = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
            valid = scalars.get("valid_iou", [])
            if not valid:
                continue
            epochs = {point.step: int(point.value) + 1 for point in scalars.get("epoch", [])}
            train = {point.step: point.value for point in scalars.get("train_iou", [])}
            points = []
            for index, point in enumerate(valid):
                train_loss = loss_at(scalars, "train", point.step)
                valid_loss = loss_at(scalars, "valid", point.step)
                points.append({
                    "epoch": epochs.get(point.step, index + 1), "step": point.step,
                    "train_iou": train.get(point.step), "valid_iou": point.value,
                    "train_loss": float(train_loss.rstrip("*")) if train_loss != "—" else None,
                    "valid_loss": float(valid_loss.rstrip("*")) if valid_loss != "—" else None,
                    "batch_loss_fallback": train_loss.endswith("*") or valid_loss.endswith("*"),
                })
            name = str(cfg.name)
            if name in runs:
                raise ValueError(f"Multiple runs named {name}; select one before building report")
            runs[name] = {"name": name, "run": str(run), "config": str(config), "tensorboard": str(logdir),
                          "input_size": list(cfg.data.image_size), "best": max(points, key=lambda row: row["valid_iou"]),
                          "curves": points}
    return dict(sorted(runs.items()))


def find(runs, prefix):
    return next(value for key, value in runs.items() if key.startswith(prefix + "_"))


def learning_plot(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for prefix, color, label in [("001", "#b74b3f", "001: без аугментаций"), ("002", "#28768e", "002: с аугментациями")]:
        points = find(runs, prefix)["curves"]
        x = [row["epoch"] for row in points]
        for ax, suffix in zip(axes, ["loss", "iou"]):
            ax.plot(x, [row[f"train_{suffix}"] for row in points], color=color, linestyle="--", label=f"{label}, train")
            ax.plot(x, [row[f"valid_{suffix}"] for row in points], color=color, label=f"{label}, valid")
    for ax, title, ylabel in zip(axes, ["Loss: расхождение train и validation", "IoU: что изменили аугментации"], ["BCE loss", "Mean IoU"]):
        ax.set(title=title, xlabel="Эпоха", ylabel=ylabel)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)
    fig.savefig(output / "baseline_augmentations.png", dpi=170)
    plt.close(fig)


def pretraining_plot(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, title, pair in zip(axes, ["U-Net / ResNet18", "SegFormer / MiT-B0"], [("005", "006"), ("008", "011")]):
        for prefix, color, label in zip(pair, ["#7d8792", "#25745a"], ["С нуля", "Энкодер ImageNet"]):
            run = find(runs, prefix)
            ax.plot([point["epoch"] for point in run["curves"]], [point["valid_iou"] for point in run["curves"]],
                    color=color, label=f"{prefix}: {label}")
            best = run["best"]
            ax.scatter([best["epoch"]], [best["valid_iou"]], color=color, s=25)
        ax.set(title=title, xlabel="Эпоха", ylabel="Valid Mean IoU", ylim=(0.55, 0.95))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, frameon=False)
    fig.savefig(output / "pretraining.png", dpi=170)
    plt.close(fig)


def comparisons(predictions, images, output):
    prefixes = ["001", "009", "013", "014", "015"]
    ids = [143, 333, 0]
    submissions, provenance = [], []
    for prefix in prefixes:
        folder = next(folder for folder in sorted(predictions.iterdir()) if folder.name.startswith(prefix + "_"))
        path = folder / "submission.csv"
        with path.open(newline="") as file:
            rows = {int(row["ImageID"]): row["RLE"] for row in csv.DictReader(file)}
        metadata_path = path.with_suffix(".csv.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        size = metadata.get("input_size")
        label = f"{prefix} | input {size[0]}" if size else prefix
        submissions.append((label, rows))
        provenance.append({"csv": str(path), "metadata": metadata})
    size = 320
    def tile(image, label):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        image = cv2.copyMakeBorder(image, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(image, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return image
    grids = []
    for image_id in ids:
        image = cv2.imread(str(images / f"{image_id}.png"))
        if image is None:
            raise OSError(image_id)
        panels = [tile(image, f"Original | ID {image_id}")]
        for label, masks in submissions:
            mask = rle_decode(masks[image_id], image.shape[:2])
            overlay = image.copy()
            overlay[mask] = (image[mask] * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
            panels.append(tile(overlay, label))
        grids.append(np.concatenate(panels, axis=1))
    path = output / "test_comparisons.jpg"
    if not cv2.imwrite(str(path), np.concatenate(grids, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(path)
    (output / "comparison_sources.json").write_text(json.dumps({"image_ids": ids, "submissions": provenance}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("report_assets"))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    runs = collect_runs(Path("outputs"))
    (output / "metrics.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", "best_epoch", "valid_iou", "train_iou", "train_loss", "valid_loss", "recorded_epochs", "config"])
        for name, run in runs.items():
            best = run["best"]
            writer.writerow([name, best["epoch"], best["valid_iou"], best["train_iou"], best["train_loss"], best["valid_loss"], len(run["curves"]), run["config"]])
    learning_plot(runs, output)
    pretraining_plot(runs, output)
    comparisons(Path("predictions/test"), Path("data/test/test/images"), output)
    shutil.copy2("predictions/augmentations/013_unet_resnet18_pretrained_strong_color/pug_159.jpg", output / "augmentations.jpg")
    shutil.copy2("predictions/dataset/coco_people_train2017/images/coco_000000106168.jpg", output / "person_pet_example.jpg")
    print(f"Built report assets from {len(runs)} runs: {output}")


if __name__ == "__main__":
    main()
