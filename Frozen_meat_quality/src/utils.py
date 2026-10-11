"""Small reusable utilities for the frozen-meat packaging quality project.

The functions are intentionally simple and are used for dataset checks, model
selection, and basic input validation.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_class_names(yaml_path):
    """Read class names from a YOLO dataset YAML file.

    Args:
        yaml_path (str or pathlib.Path): Path to the dataset YAML file.

    Returns:
        list[str]: Class names in class-ID order.
    """
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return list(names)


def audit_dataset(dataset_path, class_names):
    """Count images, label files, empty labels, and class annotations.

    Args:
        dataset_path (str or pathlib.Path): Dataset root with train/valid/test.
        class_names (list[str]): Dataset class names.

    Returns:
        pandas.DataFrame: One summary row for each split.
    """
    dataset_path = Path(dataset_path)
    rows = []

    for split in ["train", "valid", "test"]:
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"
        images = [p for p in image_dir.glob("*") if p.is_file()]
        labels = list(label_dir.glob("*.txt"))

        counts = Counter()
        empty_labels = 0

        for label_file in labels:
            lines = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
                continue
            for line in lines:
                values = line.split()
                if not values:
                    continue
                class_id = int(values[0])
                if 0 <= class_id < len(class_names):
                    counts[class_id] += 1

        row = {
            "split": split,
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty_labels,
        }
        for class_id, class_name in enumerate(class_names):
            row[class_name] = counts[class_id]
        rows.append(row)

    return pd.DataFrame(rows)


def check_dataset_errors(dataset_path, class_names):
    """Check YOLO dataset files for missing or invalid annotations.

    Empty label files are recorded for analysis but are not automatically
    treated as corrupted data because they may represent valid no-object images.

    Args:
        dataset_path (str or pathlib.Path): Dataset root with train/valid/test.
        class_names (list[str]): Dataset class names.

    Returns:
        pandas.DataFrame: Diagnostic issues with split, file, type, and detail.
    """
    dataset_path = Path(dataset_path)
    issues = []

    for split in ["train", "valid", "test"]:
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        images = [p for p in image_dir.glob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
        labels = list(label_dir.glob("*.txt"))
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        for image in images:
            if image.stem not in label_stems:
                issues.append({
                    "split": split,
                    "file": image.name,
                    "error_type": "missing_label",
                    "detail": "Image does not have a label file",
                })

        for label in labels:
            if label.stem not in image_stems:
                issues.append({
                    "split": split,
                    "file": label.name,
                    "error_type": "missing_image",
                    "detail": "Label file does not have a matching image",
                })

            lines = [line.strip() for line in label.read_text().splitlines() if line.strip()]
            if not lines:
                issues.append({
                    "split": split,
                    "file": label.name,
                    "error_type": "empty_label",
                    "detail": "No object annotation in this image",
                })
                continue

            for line_number, line in enumerate(lines, start=1):
                values = line.split()
                if len(values) != 5:
                    issues.append({
                        "split": split,
                        "file": label.name,
                        "error_type": "invalid_format",
                        "detail": f"Line {line_number}: expected 5 values",
                    })
                    continue

                try:
                    class_id = int(values[0])
                    x_center, y_center, width, height = map(float, values[1:5])
                except ValueError:
                    issues.append({
                        "split": split,
                        "file": label.name,
                        "error_type": "invalid_value",
                        "detail": f"Line {line_number}: non-numeric value",
                    })
                    continue

                if not 0 <= class_id < len(class_names):
                    issues.append({
                        "split": split,
                        "file": label.name,
                        "error_type": "invalid_class_id",
                        "detail": f"Line {line_number}: class ID {class_id}",
                    })

                if not all(0 <= value <= 1 for value in [x_center, y_center, width, height]):
                    issues.append({
                        "split": split,
                        "file": label.name,
                        "error_type": "invalid_coordinate",
                        "detail": f"Line {line_number}: coordinate outside 0-1",
                    })

    return pd.DataFrame(issues, columns=["split", "file", "error_type", "detail"])


def choose_best_model(comparison_df, metric="mAP50_95"):
    """Select the model with the highest validation metric.

    Args:
        comparison_df (pandas.DataFrame): Model comparison table.
        metric (str): Metric column used for selection.

    Returns:
        str: Model name from the best row.
    """
    best_index = comparison_df[metric].astype(float).idxmax()
    return str(comparison_df.loc[best_index, "model"])


def validate_image_file(image_path):
    """Check whether an image path exists and can be opened.

    Args:
        image_path (str or pathlib.Path): Image file to validate.

    Returns:
        tuple[bool, str]: Success flag and a short explanation.
    """
    path = Path(image_path)
    if not path.exists():
        return False, "file does not exist"
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, "unsupported image extension"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False, "image cannot be opened"
    return True, "ok"
