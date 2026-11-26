from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.
    """
    raw_root = Path(raw_root)
    if not raw_root.exists() or not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")

    splits_report: dict[str, dict] = {}
    total_images = 0

    for split_name in ("train", "validation", "test"):
        expected_count = EXPECTED_SPLITS[split_name]
        split_dir = raw_root / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            raise FileNotFoundError(f"Missing required split directory: {split_dir}")

        actual_classes = {d.name for d in split_dir.iterdir() if d.is_dir()}
        if actual_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Split '{split_name}' classes mismatch: expected {sorted(EXPECTED_CLASSES)}, found {sorted(actual_classes)}"
            )

        class_counts: dict[str, int] = {}
        split_total = 0

        for cls_name in sorted(EXPECTED_CLASSES):
            cls_dir = split_dir / cls_name
            image_files = sorted(
                p for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_files:
                raise RuntimeError(f"Class '{cls_name}' in split '{split_name}' is empty.")

            for img_path in image_files:
                # Laisse Pillow lever OSError ou UnidentifiedImageError si corrompue
                with Image.open(img_path) as img:
                    img.load()

            count = len(image_files)
            class_counts[cls_name] = count
            split_total += count

        if split_total == 0:
            raise RuntimeError(f"Split '{split_name}' is empty.")

        if split_total != expected_count:
            raise RuntimeError(
                f"Split '{split_name}' count mismatch: expected {expected_count}, found {split_total}"
            )

        splits_report[split_name] = {
            "images": split_total,
            "class_counts": class_counts,
        }
        total_images += split_total

    expected_total = sum(EXPECTED_SPLITS.values())
    if total_images != expected_total:
        raise RuntimeError(
            f"Total image count mismatch: expected {expected_total}, found {total_images}"
        )

    return {
        "status": "PASS",
        "classes": sorted(EXPECTED_CLASSES),
        "splits": splits_report,
        "total_images": total_images,
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
