from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from src.utils import (
    audit_dataset,
    check_dataset_errors,
    choose_best_model,
    read_class_names,
    validate_image_file,
)


def make_split(root, split):
    (root / split / "images").mkdir(parents=True)
    (root / split / "labels").mkdir(parents=True)


def test_read_class_names(tmp_path):
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(yaml.safe_dump({"names": ["a", "b"]}))
    assert read_class_names(yaml_file) == ["a", "b"]


def test_audit_dataset(tmp_path):
    for split in ["train", "valid", "test"]:
        make_split(tmp_path, split)
        Image.new("RGB", (20, 20), "white").save(tmp_path / split / "images" / "image1.jpg")
        (tmp_path / split / "labels" / "image1.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    result = audit_dataset(tmp_path, ["defect"])
    assert len(result) == 3
    assert result["images"].sum() == 3
    assert result["defect"].sum() == 3


def test_dataset_error_check_records_empty_label(tmp_path):
    for split in ["train", "valid", "test"]:
        make_split(tmp_path, split)
    Image.new("RGB", (20, 20), "white").save(tmp_path / "train" / "images" / "empty.jpg")
    (tmp_path / "train" / "labels" / "empty.txt").write_text("")

    result = check_dataset_errors(tmp_path, ["defect"])
    assert (result["error_type"] == "empty_label").sum() == 1


def test_dataset_error_check_invalid_coordinate(tmp_path):
    for split in ["train", "valid", "test"]:
        make_split(tmp_path, split)
    Image.new("RGB", (20, 20), "white").save(tmp_path / "train" / "images" / "bad.jpg")
    (tmp_path / "train" / "labels" / "bad.txt").write_text("0 1.2 0.5 0.2 0.2\n")

    result = check_dataset_errors(tmp_path, ["defect"])
    assert "invalid_coordinate" in set(result["error_type"])


def test_choose_best_model():
    data = pd.DataFrame([
        {"model": "model_a", "mAP50_95": 0.40},
        {"model": "model_b", "mAP50_95": 0.55},
    ])
    assert choose_best_model(data) == "model_b"


def test_validate_image_file(tmp_path):
    valid_path = tmp_path / "image.jpg"
    Image.new("RGB", (20, 20), "white").save(valid_path)
    assert validate_image_file(valid_path)[0] is True
    assert validate_image_file(tmp_path / "missing.jpg")[0] is False
