"""Small synthetic fixtures test SOFTWARE ONLY, not IDS accuracy on UNSW-NB15."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ids_project.core import (
    FEATURES, NUMERIC, CATEGORICAL, DatasetLoader, DataValidationError,
    ModelTrainer, PredictionService, ThresholdSelector, get_metrics,
)


def fake_csv(path: Path, *, rows: int = 80, labeled: bool = True) -> Path:
    random = np.random.default_rng(123)
    labels = np.tile([0, 1], rows // 2 + 1)[:rows]
    vals = {col: random.uniform(0.01, 1.0, rows) for col in NUMERIC}
    vals["sbytes"] = labels * 30 + random.normal(1, 0.08, rows)
    vals["rate"] = labels * 12 + random.normal(1, 0.08, rows)
    vals["proto"] = np.where(labels == 1, "tcp", "udp")
    vals["service"] = np.where(labels == 1, "http", "dns")
    vals["state"] = np.where(labels == 1, "FIN", "CON")
    vals["id"] = range(1, rows + 1)
    vals["attack_cat"] = np.where(labels == 1, "FixtureAttack", "Normal")
    if labeled:
        vals["label"] = labels
    pd.DataFrame(vals).to_csv(path, index=False)
    return path


def test_schema_and_sha(tmp_path: Path):
    path = fake_csv(tmp_path / "sample.csv")
    item = DatasetLoader().load(path, labels_required=True)
    assert len(item.table) == 80
    assert len(item.checksum) == 64
    assert all(c in item.table for c in FEATURES)


def test_missing_required_field(tmp_path: Path):
    path = fake_csv(tmp_path / "incomplete.csv")
    df = pd.read_csv(path).drop(columns=["proto"])
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="proto"):
        DatasetLoader().load(path, labels_required=True)


def test_unknown_label(tmp_path: Path):
    path = fake_csv(tmp_path / "wrong.csv")
    df = pd.read_csv(path)
    df.loc[0, "label"] = 7
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="label"):
        DatasetLoader().load(path, labels_required=True)


def test_metrics_known_values():
    report = get_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]))
    assert [report[k] for k in ("TN", "FP", "FN", "TP")] == [1, 1, 1, 1]
    assert report["f1"] == 0.5
    threshold, _ = ThresholdSelector().choose(np.array([0, 1]), np.array([0.20, 0.80]))
    assert 0.20 < threshold <= 0.8


def test_real_pipeline_on_fixture_only(tmp_path: Path):
    train_csv = fake_csv(tmp_path / "train_fixture.csv", rows=80)
    test_csv = fake_csv(tmp_path / "test_fixture.csv", rows=50)
    training = ModelTrainer().run(train_csv, tmp_path / "outputs", seed=42)
    metadata = json.loads((training / "run_config.json").read_text(encoding="utf-8"))
    assert metadata["all_training_rows"] == 80
    assert metadata["validation_rows"] == 16
    assert metadata["test_data_used_for_selection"] is False
    assert metadata["selected_model"] in {"logistic_regression", "random_forest"}
    evaluated = PredictionService.run(training, test_csv, tmp_path / "outputs", require_labels=True)
    metrics = json.loads((evaluated / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rows"] == 50
    assert metrics["TN"] + metrics["FP"] + metrics["FN"] + metrics["TP"] == 50
    assert (evaluated / "predictions.csv").exists()
    assert (evaluated / "per_attack_category.csv").exists()
    assert (evaluated / "confusion_matrix.png").exists()


def test_unlabeled_predict(tmp_path: Path):
    train_csv = fake_csv(tmp_path / "train_fixture.csv", rows=60)
    plain_csv = fake_csv(tmp_path / "unlabeled_fixture.csv", rows=20, labeled=False)
    model = ModelTrainer().run(train_csv, tmp_path / "runs")
    result = PredictionService.run(model, plain_csv, tmp_path / "runs", require_labels=False)
    info = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    assert info["measurement_status"] == "inference without target evaluation"
    assert "f1" not in info


def test_no_same_csv_as_independent_test(tmp_path: Path):
    train_csv = fake_csv(tmp_path / "train_fixture.csv", rows=60)
    model = ModelTrainer().run(train_csv, tmp_path / "runs")
    with pytest.raises(DataValidationError, match="самий файл"):
        PredictionService.run(model, train_csv, tmp_path / "runs", require_labels=True)
