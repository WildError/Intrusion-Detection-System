"""Reproducible, *offline flow-based* binary IDS on UNSW-NB15.

No live packet capture. Never select a threshold on the official test data.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["proto", "service", "state"]
FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
NUMERIC = [field for field in FEATURES if field not in CATEGORICAL]
TARGET = "label"
VERSION = "0.1.0"
SEED = 42


class DataValidationError(ValueError):
    """Predictable validation error for malformed CSV."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(prefix: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{prefix}-" + uuid.uuid4().hex[:8]


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


@dataclass(frozen=True)
class LoadedDataset:
    table: pd.DataFrame
    source_path: Path
    checksum: str


class SchemaValidator:
    """Validate an UNSW-NB15 feature table; label is optional in prediction."""

    def validate(self, df: pd.DataFrame, *, labels_required: bool) -> pd.DataFrame:
        if df.empty:
            raise DataValidationError("CSV-файл порожній")
        if not df.columns.is_unique:
            raise DataValidationError("У CSV виявлено повторювані назви колонок")
        missing = [field for field in FEATURES if field not in df.columns]
        if missing:
            raise DataValidationError(f"Відсутні колонки UNSW-NB15: {', '.join(missing[:12])}")
        if labels_required and TARGET not in df.columns:
            raise DataValidationError("Відсутня обов'язкова колонка label")
        output = df.copy()
        for field in NUMERIC:
            numeric = pd.to_numeric(output[field], errors="coerce")
            bad_values = output[field].notna() & numeric.isna()
            if bad_values.any():
                raise DataValidationError(f"Некоректне числове значення в колонці {field}")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            if numeric.notna().sum() == 0:
                raise DataValidationError(f"У колонці {field} немає числових значень")
            output[field] = numeric.astype("float64")
        for field in CATEGORICAL:
            if output[field].notna().sum() == 0:
                raise DataValidationError(f"Категоріальна колонка {field} повністю порожня")
            output[field] = output[field].astype("string").fillna("missing").astype(str)
        if TARGET in output.columns:
            numeric_label = pd.to_numeric(output[TARGET], errors="coerce")
            if numeric_label.isna().any() or not numeric_label.isin([0, 1]).all():
                raise DataValidationError("Колонка label має містити тільки 0 або 1")
            output[TARGET] = numeric_label.astype(int)
        return output


class DatasetLoader:
    def __init__(self) -> None:
        self.validator = SchemaValidator()

    def load(self, path: str | Path, *, labels_required: bool) -> LoadedDataset:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV не знайдено: {path.name}")
        if path.suffix.lower() != ".csv":
            raise DataValidationError("Потрібен файл у форматі CSV")
        try:
            # low_memory=False prevents inconsistent inference on mixed CSV columns.
            table = pd.read_csv(path, low_memory=False)
        except Exception as error:
            raise DataValidationError("Помилка читання CSV: перевірте кодування та розділювач") from error
        return LoadedDataset(self.validator.validate(table, labels_required=labels_required), path, sha256_file(path))


class PipelineFactory:
    @staticmethod
    def create(name: str, seed: int) -> Pipeline:
        if name not in {"logistic_regression", "random_forest"}:
            raise ValueError(f"Невідома модель: {name}")
        numerical_steps = [("imputer", SimpleImputer(strategy="median"))]
        if name == "logistic_regression":
            numerical_steps.append(("scaler", StandardScaler()))
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:  # scikit-learn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", encoder),
        ])
        prep = ColumnTransformer([
            ("numerical", Pipeline(numerical_steps), NUMERIC),
            ("categorical", categorical_pipe, CATEGORICAL),
        ], remainder="drop")
        if name == "logistic_regression":
            clf = LogisticRegression(solver="liblinear", max_iter=400, class_weight="balanced", random_state=seed)
        else:
            clf = RandomForestClassifier(
                n_estimators=60, max_depth=14, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
            )
        return Pipeline([("preprocess", prep), ("classifier", clf)])


def get_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, predicted, labels=[0, 1]).ravel())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, labels=[1], average="binary", zero_division=0,
    )
    return {
        "sample_count": len(y_true), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
    }


class ThresholdSelector:
    """Search on INTERNAL validation only, never on the official test split."""

    GRID = [round(i / 100, 2) for i in range(5, 96)]

    def choose(self, y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, Any]]:
        candidates = []
        for threshold in self.GRID:
            scores = get_metrics(y_true, probs >= threshold)
            candidates.append((threshold, scores))
        # Highest F1, then lowest false positive rate, then highest threshold.
        threshold, best = max(candidates, key=lambda item: (item[1]["f1"], -item[1]["fpr"], item[0]))
        return threshold, best


def make_log(run_dir: Path, *, phase: str, status: str, **kwargs: Any) -> None:
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"timestamp": now_utc(), "phase": phase, "status": status, **kwargs}, ensure_ascii=False) + "\n")


def _software() -> dict[str, str]:
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}


class ModelTrainer:
    def run(self, train_csv: str | Path, output_root: str | Path, *, seed: int = SEED) -> Path:
        run_dir = Path(output_root) / make_run_id("train")
        run_dir.mkdir(parents=True, exist_ok=False)
        make_log(run_dir, phase="training", status="started")
        try:
            ds = DatasetLoader().load(train_csv, labels_required=True)
            labels = ds.table[TARGET]
            if labels.value_counts().min() < 3 or labels.nunique() < 2:
                raise DataValidationError("Для стратифікованого поділу потрібні обидва класи з ≥3 записами")
            train_idx, val_idx = train_test_split(
                np.arange(len(ds.table)), test_size=0.20, stratify=labels, random_state=seed,
            )
            x_train = ds.table.iloc[train_idx][FEATURES]
            y_train = labels.iloc[train_idx].to_numpy()
            x_val = ds.table.iloc[val_idx][FEATURES]
            y_val = labels.iloc[val_idx].to_numpy()
            rows = []
            pipelines: dict[str, Pipeline] = {}
            for model_name in ("logistic_regression", "random_forest"):
                pipeline = PipelineFactory.create(model_name, seed)
                pipeline.fit(x_train, y_train)
                probs = pipeline.predict_proba(x_val)[:, 1]
                threshold, metrics = ThresholdSelector().choose(y_val, probs)
                rows.append({"model": model_name, "threshold": threshold, **metrics})
                pipelines[model_name] = pipeline
                make_log(run_dir, phase="validation", status="ok", model=model_name, f1=metrics["f1"], threshold=threshold)
            selection = pd.DataFrame(rows).sort_values(["f1", "fpr", "threshold"], ascending=[False, True, False])
            winner = selection.iloc[0].to_dict()
            winning_model = str(winner["model"])
            model_path = run_dir / "model.joblib"
            # Do not load joblib files from an untrusted source.
            joblib.dump(pipelines[winning_model], model_path)
            selection.to_csv(run_dir / "validation_comparison.csv", index=False)
            metadata = {
                "schema_version": VERSION, "created_utc": now_utc(), "run_id": run_dir.name,
                "dataset_name": "UNSW-NB15", "training_filename": ds.source_path.name,
                "training_sha256": ds.checksum, "all_training_rows": len(ds.table),
                "fitting_rows": len(train_idx), "validation_rows": len(val_idx),
                "class_distribution": {str(k): int(v) for k, v in labels.value_counts().items()},
                "seed": seed, "split": "official training file -> stratified 80/20 internal train/validation",
                "selected_model": winning_model, "decision_threshold": float(winner["threshold"]),
                "validation_scores": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in winner.items()},
                "selection_criterion": "F1 descending, FPR ascending, threshold descending on internal validation",
                "test_data_used_for_selection": False, "features": FEATURES,
                "excluded_fields": ["id", "attack_cat", "label"], "software": _software(),
                "model_sha256": sha256_file(model_path),
            }
            write_json(run_dir / "run_config.json", metadata)
            make_log(run_dir, phase="training", status="ok", selected_model=winning_model)
            return run_dir
        except Exception as error:
            make_log(run_dir, phase="training", status="failed", error_type=type(error).__name__, message=str(error))
            raise


class PredictionService:
    @staticmethod
    def run(model_dir: str | Path, dataset_csv: str | Path, output_root: str | Path, *, require_labels: bool) -> Path:
        model_dir = Path(model_dir)
        model_file = model_dir / "model.joblib"
        metadata_file = model_dir / "run_config.json"
        if not model_file.exists() or not metadata_file.exists():
            raise FileNotFoundError("Не знайдено збереженої моделі або run_config.json")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("features") != FEATURES:
            raise DataValidationError("Схема ознак збереженої моделі не відповідає версії програми")
        if sha256_file(model_file) != metadata.get("model_sha256"):
            raise DataValidationError("Контрольна сума моделі не збігається")
        # SECURITY: joblib is pickle-based; use only locally created/trusted model files.
        pipeline: Pipeline = joblib.load(model_file)
        mode = "evaluate" if require_labels else "predict"
        run_dir = Path(output_root) / make_run_id(mode)
        run_dir.mkdir(parents=True, exist_ok=False)
        make_log(run_dir, phase=mode, status="started")
        try:
            ds = DatasetLoader().load(dataset_csv, labels_required=require_labels)
            if require_labels and ds.checksum == metadata.get("training_sha256"):
                raise DataValidationError("Той самий файл не можна використовувати як офіційний незалежний тест")
            probabilities = pipeline.predict_proba(ds.table[FEATURES])[:, 1]
            threshold = float(metadata["decision_threshold"])
            labels = (probabilities >= threshold).astype(int)
            out = pd.DataFrame({
                "row_id": ds.table["id"] if "id" in ds.table else np.arange(len(ds.table)),
                "probability_attack": np.round(probabilities, 9),
                "predicted_label": labels,
                "decision_threshold": threshold,
                "run_id": run_dir.name,
            })
            if TARGET in ds.table:
                out["true_label"] = ds.table[TARGET].to_numpy()
            if "attack_cat" in ds.table:
                out["attack_cat"] = ds.table["attack_cat"].astype(str).to_numpy()
            out.to_csv(run_dir / "predictions.csv", index=False)
            report = {
                "run_id": run_dir.name, "model_training_run": metadata["run_id"],
                "dataset_filename": ds.source_path.name, "dataset_sha256": ds.checksum,
                "rows": len(ds.table), "predicted_normal": int((labels == 0).sum()),
                "predicted_attack": int((labels == 1).sum()), "threshold": threshold,
                "model": metadata["selected_model"], "created_utc": now_utc(),
                "measurement_status": "independent labeled evaluation" if require_labels else "inference without target evaluation",
            }
            if require_labels:
                y_true = ds.table[TARGET].to_numpy()
                report.update(get_metrics(y_true, labels))
                if "attack_cat" in ds.table:
                    tmp = pd.DataFrame({"attack_cat": ds.table["attack_cat"].astype(str), "true": y_true, "pred": labels})
                    rows = []
                    for name, group in tmp[tmp["true"] == 1].groupby("attack_cat"):
                        support = len(group)
                        rows.append({"attack_cat": name, "support": support, "true_positives": int((group["pred"] == 1).sum()), "missed": int((group["pred"] == 0).sum()), "recall": float((group["pred"] == 1).mean())})
                    pd.DataFrame(rows).to_csv(run_dir / "per_attack_category.csv", index=False)
                _save_confusion(run_dir / "confusion_matrix.png", report)
            write_json(run_dir / "metrics.json", report)
            write_json(run_dir / "run_config.json", {
                "run_id": run_dir.name, "model_training_run": metadata["run_id"],
                "source_sha256": ds.checksum, "source_name": ds.source_path.name,
                "mode": mode, "decision_threshold": threshold, "features": FEATURES,
                "software": _software(), "timestamp": now_utc(),
            })
            make_log(run_dir, phase=mode, status="ok", rows=len(ds.table))
            return run_dir
        except Exception as error:
            make_log(run_dir, phase=mode, status="failed", error_type=type(error).__name__, message=str(error))
            raise


def _save_confusion(path: Path, result: dict[str, Any]) -> None:
    matrix = np.array([[result["TN"], result["FP"]], [result["FN"], result["TP"]]])
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=160)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f"{val:,}".replace(",", " "), ha="center", va="center", fontsize=12, color=("white" if val > matrix.max() / 2 else "black"))
    ax.set_xticks([0, 1], ["0 / normal", "1 / attack"])
    ax.set_yticks([0, 1], ["0 / normal", "1 / attack"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (independent test)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
