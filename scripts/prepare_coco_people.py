"""Download a small, reproducible COCO subset for pet/person discrimination.

Only COCO train2017 photos selected by instance annotations are downloaded.
Foreground is the union of cat/dog polygons; person-only examples have zero masks.
Crowd/nonpolygon target annotations are excluded, not treated as background.
"""

import argparse
from collections import defaultdict
import csv
import io
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import urllib.request
import zipfile

import cv2
import numpy as np
from tqdm import tqdm


ANNOTATIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_URL = "https://s3.amazonaws.com/images.cocodataset.org/train2017/"


class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP range reader: fetch only the needed ZIP member."""

    def __init__(self, url, prefix_path=None):
        self.url = url
        self.prefix_path = prefix_path
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative ZIP offset")
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size) if size >= 0 else self.size - self.position
        if size <= 0:
            return b""
        start = self.position
        cached = b""
        if self.prefix_path and self.prefix_path.exists():
            available = max(0, self.prefix_path.stat().st_size - start)
            if available:
                with self.prefix_path.open("rb") as file:
                    file.seek(start)
                    cached = file.read(min(size, available))
        chunk_size = 8 * 1024 * 1024
        ranges = [(offset, min(offset + chunk_size, start + size) - 1)
                  for offset in range(start + len(cached), start + size, chunk_size)]
        def fetch(bounds):
            begin, end = bounds
            for attempt in range(3):
                try:
                    request = urllib.request.Request(self.url, headers={"Range": f"bytes={begin}-{end}"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {begin}-"):
                            raise OSError("COCO server did not honor HTTP range request")
                        part = response.read()
                    if len(part) != end - begin + 1:
                        raise OSError("Incomplete COCO annotation range")
                    return part
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1)
        with ThreadPoolExecutor(max_workers=8) as executor:
            result = cached + b"".join(executor.map(fetch, ranges))
        self.position += len(result)
        return result


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            partial.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def dhash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes(), "big")


def candidates(data: dict) -> dict[str, list[tuple[dict, list]]]:
    categories = {category["name"]: category["id"] for category in data["categories"]}
    pet_ids = {categories["cat"], categories["dog"]}
    person_id = categories["person"]
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    result = {"person_pet": [], "person_only": []}
    for image in data["images"]:
        annotations = by_image[image["id"]]
        people = [ann for ann in annotations if ann["category_id"] == person_id]
        pets = [ann for ann in annotations if ann["category_id"] in pet_ids]
        if not people:
            continue
        if any(ann.get("iscrowd", 0) for ann in people + pets):
            continue
        if any(not isinstance(ann["segmentation"], list) or not ann["segmentation"] for ann in pets):
            continue
        result["person_pet" if pets else "person_only"].append((image, pets))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/coco_people_train2017"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/cache"))
    parser.add_argument("--pets-root", type=Path, default=Path("data/train/data"))
    parser.add_argument("--test-root", type=Path, default=Path("data/test/test/images"))
    parser.add_argument("--per-group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("per-group must be positive")
    if not (args.pets_root / "images/images").is_dir():
        parser.error("pets-root must contain Oxford images for duplicate screening")
    if not args.test_root.is_dir():
        parser.error("test-root must contain competition test images for duplicate screening")
    annotations = args.cache_dir / "instances_train2017.json"
    if not annotations.exists():
        print("Fetching only train2017 instance annotations from official ZIP...", flush=True)
        archive = args.cache_dir / "annotations_trainval2017.zip"
        prefix = archive.with_suffix(archive.suffix + ".part")
        with zipfile.ZipFile(archive if archive.exists() else RemoteZipFile(ANNOTATIONS_URL, prefix)) as file:
            content = file.read("annotations/instances_train2017.json")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        annotations.write_bytes(content)
        del content
    data = json.loads(annotations.read_text())
    groups = candidates(data)
    print("Available candidates:", {key: len(value) for key, value in groups.items()}, flush=True)
    # Screen against Oxford AND competition test photos: external annotations
    # must not introduce training on the hidden test's ground-truth masks.
    hashes = []
    reference_paths = (sorted((args.pets_root / "images/images").glob("*.jpg"))
                       + sorted(args.test_root.glob("*.png")))
    for path in tqdm(reference_paths, desc="Oxford/test duplicate index"):
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"Cannot read {path}")
        hashes.append(dhash(image))
    output = args.output_dir
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    rows, rejected, seen = [], [], []
    for group, pool in groups.items():
        rng.shuffle(pool)
        count = 0
        # Cache candidate photos concurrently; process selection in seeded order.
        chosen = pool[:min(len(pool), args.per_group + 50)]
        def fetch(item):
            image, _ = item
            download(IMAGES_URL + image["file_name"], args.cache_dir / "train2017" / image["file_name"])
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(executor.map(fetch, chosen), total=len(chosen), desc=f"Download {group}"))
        for info, pets in chosen:
            if count >= args.per_group:
                break
            path = args.cache_dir / "train2017" / info["file_name"]
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (info["height"], info["width"]):
                raise OSError(f"Invalid image: {path}")
            fingerprint = dhash(image)
            if any((fingerprint ^ other).bit_count() <= 4 for other in hashes + seen):
                rejected.append(info["id"])
                continue
            mask = np.zeros(image.shape[:2], np.uint8)
            for pet in pets:
                for polygon in pet["segmentation"]:
                    points = np.round(np.array(polygon).reshape(-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
            if pets and not mask.any():
                raise ValueError(f"Empty foreground in positive image {info['id']}")
            name = f"coco_{info['id']:012d}"
            image_path = output / "images" / f"{name}.jpg"
            mask_path = output / "masks" / f"{name}.png"
            image_path.write_bytes(path.read_bytes())
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"Cannot save {mask_path}")
            rows.append({"name": name, "group": group, "image": f"images/{name}.jpg",
                         "mask": f"masks/{name}.png", "coco_image_id": info["id"],
                         "source_url": IMAGES_URL + info["file_name"],
                         "flickr_url": info.get("flickr_url"), "license": info.get("license")})
            seen.append(fingerprint)
            count += 1
        print(f"Selected {group}: {count}", flush=True)
    manifest = {"dataset": "COCO train2017", "training_allowed": True, "annotations_url": ANNOTATIONS_URL,
                "seed": args.seed, "rasterization": "cv2.fillPoly on rounded COCO polygon coordinates",
                "foreground": ["cat", "dog"], "licenses": data["licenses"],
                "duplicate_screen": "64-bit dHash distance <= 4 against all Oxford photos, competition test photos and selected COCO photos; heuristic, not a guarantee",
                "rejected_duplicates": rejected, "samples": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "index.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "group", "image", "mask", "coco_image_id", "source_url", "flickr_url", "license"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} examples to {output}")


if __name__ == "__main__":
    main()
