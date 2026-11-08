"""Verify prepared masks against COCO polygons; optionally exclude other animals."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from prepare_coco_people import ANIMAL_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/external/coco_people_train2017"))
    parser.add_argument("--annotations", type=Path, default=Path("data/external/cache/instances_train2017.json"))
    parser.add_argument("--exclude-other-animals", action="store_true")
    args = parser.parse_args()
    root = args.data_root
    manifest = json.loads((root / "manifest.json").read_text())
    data = json.loads(args.annotations.read_text())
    categories = {item["id"]: item["name"] for item in data["categories"]}
    pet_ids = {key for key, value in categories.items() if value in {"cat", "dog"}}
    other_ids = {key for key, value in categories.items() if value in ANIMAL_CLASSES - {"cat", "dog"}}
    selected_ids = {sample["coco_image_id"] for sample in manifest["samples"]}
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["image_id"] in selected_ids:
            annotations[annotation["image_id"]].append(annotation)
    info = {image["id"]: image for image in data["images"] if image["id"] in selected_ids}
    del data
    errors, excluded, kept = [], [], []
    for sample in manifest["samples"]:
        image_id = sample["coco_image_id"]
        source = annotations[image_id]
        animal_names = sorted({categories[ann["category_id"]] for ann in source if ann["category_id"] in other_ids})
        if animal_names:
            excluded.append({"name": sample["name"], "coco_image_id": image_id,
                             "reason": "other_animal_classes", "classes": animal_names})
        source_pets = [ann for ann in source if ann["category_id"] in pet_ids]
        expected = np.zeros((info[image_id]["height"], info[image_id]["width"]), np.uint8)
        for ann in source_pets:
            if ann.get("iscrowd", 0) or not isinstance(ann["segmentation"], list):
                raise ValueError(f"Unsupported pet annotation: {sample['name']}")
            for polygon in ann["segmentation"]:
                cv2.fillPoly(expected, [np.round(np.array(polygon).reshape(-1, 2)).astype(np.int32)], 1)
        actual = cv2.imread(str(root / sample["mask"]), cv2.IMREAD_GRAYSCALE)
        if actual is None or not np.array_equal(actual, expected):
            errors.append({"name": sample["name"], "reason": "mask_differs_from_source_polygons"})
        expected_group = "person_pet" if source_pets else "person_only"
        if sample["group"] != expected_group:
            errors.append({"name": sample["name"], "reason": "wrong_group"})
        if not args.exclude_other_animals or not animal_names:
            kept.append(sample)
    report = {"checked_samples": len(manifest["samples"]), "mask_errors": errors,
              "other_animal_samples": excluded, "remaining_groups": dict(Counter(row["group"] for row in kept))}
    (root / "mask_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise ValueError("Mask audit failed; see mask_audit.json")
    if args.exclude_other_animals:
        # Quarantine instead of deleting: retain originals and the reason for exclusion.
        kept_ids = {sample["coco_image_id"] for sample in kept}
        removed = [sample for sample in manifest["samples"] if sample["coco_image_id"] not in kept_ids]
        for sample in removed:
            for key in ("image", "mask"):
                source = root / sample[key]
                target = root / "excluded_other_animals" / sample[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), target)
            preview = Path("predictions/dataset") / root.name / "images" / f"{sample['name']}.jpg"
            if preview.exists():
                target = preview.parent.parent / "excluded_other_animals" / preview.name
                target.parent.mkdir(exist_ok=True)
                shutil.move(str(preview), target)
        manifest["samples"] = kept
        manifest["excluded_animal_classes"] = sorted(ANIMAL_CLASSES - {"cat", "dog"})
        manifest.setdefault("curation_exclusions", []).extend(excluded)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (root / "index.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(kept[0]))
            writer.writeheader()
            writer.writerows(kept)
        audit_path = root / "audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            audit["samples"] = len(kept)
            audit["groups"] = dict(Counter(row["group"] for row in kept))
            audit["nonempty_masks"] = {key: (count if key == "person_pet" else 0) for key, count in audit["groups"].items()}
            audit["curation"] = "Other-animal samples removed; duplicate-screen results still apply to this subset."
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
