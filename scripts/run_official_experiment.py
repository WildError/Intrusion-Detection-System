#!/usr/bin/env python3
"""Run the actual official UNSW-NB15 experiment and publish only aggregate evidence.

This script DOES NOT contain pre-computed results, and it does NOT manufacture logs.
It calls the repository's existing ModelTrainer and PredictionService.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ids_project.core import DatasetLoader, ModelTrainer, PredictionService

EXPECTED_TRAIN_ROWS = 175341
EXPECTED_TEST_ROWS = 82332
EXPECTED_TRAIN_NAME = "UNSW_NB15_training-set.csv"
EXPECTED_TEST_NAME = "UNSW_NB15_testing-set.csv"

THESIS_REPORT = {
    "selected_model": "random_forest",
    "decision_threshold": 0.35,
    "validation_f1": 0.9649,
    "test_precision": 0.8025,
    "test_recall": 0.9911,
    "test_f1": 0.8869,
    "test_fpr": 0.2988,
    "test_accuracy": 0.8608,
    "TN": 25944, "FP": 11056, "FN": 404, "TP": 44928,
}

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def copy_evidence(src: Path, dst: Path, names: tuple[str, ...]) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    done = []
    for name in names:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            done.append(name)
    return done

def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                           check=True, capture_output=True, timeout=6)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None

def fmt_percent(v):
    return f"{float(v)*100:.2f}%"

def create_summary(path: Path, meta: dict, metrics: dict, train_run: Path, test_run: Path) -> None:
    winner = meta["validation_scores"]
    actual = {
        "selected_model": meta["selected_model"],
        "decision_threshold": float(meta["decision_threshold"]),
        "validation_f1": float(winner["f1"]),
        **{k: metrics[k] for k in ("TN", "FP", "FN", "TP", "precision", "recall", "f1", "fpr", "accuracy")},
    }
    checks = [
        ("Вибрана модель", THESIS_REPORT["selected_model"], actual["selected_model"], None),
        ("Поріг", THESIS_REPORT["decision_threshold"], actual["decision_threshold"], 0.00001),
        ("F1, внутрішня валідація", THESIS_REPORT["validation_f1"], actual["validation_f1"], 0.0002),
        ("Precision, тест", THESIS_REPORT["test_precision"], metrics["precision"], 0.0002),
        ("Recall, тест", THESIS_REPORT["test_recall"], metrics["recall"], 0.0002),
        ("F1, тест", THESIS_REPORT["test_f1"], metrics["f1"], 0.0002),
        ("FPR, тест", THESIS_REPORT["test_fpr"], metrics["fpr"], 0.0002),
        ("Accuracy, тест", THESIS_REPORT["test_accuracy"], metrics["accuracy"], 0.0002),
    ]
    for key in ("TN", "FP", "FN", "TP"):
        checks.append((key, THESIS_REPORT[key], metrics[key], 0))
    def status(expected, found, tolerance):
        if tolerance is None:
            return "ЗБІГ" if expected == found else "ВІДМІННІСТЬ"
        return "ЗБІГ" if abs(float(expected) - float(found)) <= tolerance else "ВІДМІННІСТЬ"
    lines = [
        "# Звіт реального запуску: UNSW-NB15",
        "",
        "",
        f"- Навчальний файл: `{meta['training_filename']}`; SHA-256: `{meta['training_sha256']}`",
        f"- Тестовий файл: `{metrics['dataset_filename']}`; SHA-256: `{metrics['dataset_sha256']}`",
        f"- Навчальних записів: **{meta['all_training_rows']}**; fit: **{meta['fitting_rows']}**, validation: **{meta['validation_rows']}**",
        f"- Незалежний тест: **{metrics['rows']}** записів",
        f"- Вибрана модель: **{meta['selected_model']}**, поріг **{meta['decision_threshold']:.2f}**",
        f"- Навчальний запуск: `{train_run.name}`; тестовий запуск: `{test_run.name}`",
        "",
        "## Показники незалежного тесту",
        "",
        "| TN | FP | FN | TP | Precision | Recall | F1 | Accuracy | FPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['TN']} | {metrics['FP']} | {metrics['FN']} | {metrics['TP']} | {fmt_percent(metrics['precision'])} | {fmt_percent(metrics['recall'])} | {fmt_percent(metrics['f1'])} | {fmt_percent(metrics['accuracy'])} | {fmt_percent(metrics['fpr'])} |",
        "",
        "## Порівняння зі значеннями диплома",
        "",
        "| Показник | У дипломі | Повторний запуск | Контроль |",
        "|---|---:|---:|---|",
    ]
    for label, exp, act, tolerance in checks:
        s=status(exp,act,tolerance)
        lines.append(f"| {label} | {exp} | {act} | **{s}** |")
    lines += [
        "",
        "",
        "## Файли експерименту",
        "",
        "- `train/validation_comparison.csv`: порівняння моделей на внутрішній валідації",
        "- `train/run_config.json`: схема ознак, хеші, seed, бібліотеки, поріг і вибрана модель",
        "- `train/events.jsonl`: журнал навчання та валідації",
        "- `test/metrics.json`: числові метрики незалежного тесту",
        "- `test/confusion_matrix.png`: матриця помилок",
        "- `test/events.jsonl`: журнал незалежного тестування",
        "- `test/per_attack_category.csv`: оцінка атак за категоріями (якщо є `attack_cat`)",
        "- `manifest.json`: ідентифікатори запусків та git-ревізія",
        "",
        "Дані, повні прогнози для кожного рядка та файли моделей **не** публікуються.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="Real UNSW-NB15 train/test run + publish safe experiment artifacts")
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "data" / EXPECTED_TRAIN_NAME)
    parser.add_argument("--test-file", type=Path, default=REPO_ROOT / "data" / EXPECTED_TEST_NAME)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "documentation" / "experiments" / "official_unsw_nb15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_file, test_file = args.train_file.resolve(), args.test_file.resolve()
    output_dir = args.out.resolve()
    for path, filename in ((train_file, EXPECTED_TRAIN_NAME), (test_file, EXPECTED_TEST_NAME)):
        if not path.is_file():
            parser.error(f"Не знайдено {path}. Завантажте офіційні CSV з UNSW.")
        if path.name != filename:
            parser.error(f"Невірна назва: {path.name}, очікується {filename}")
    if train_file == test_file:
        parser.error("Навчальний і тестовий CSV повинні бути різними")
    if output_dir in (REPO_ROOT, REPO_ROOT / "data"):
        parser.error("Вкажіть окрему папку для звіту")
    # Refuse to overwrite prior evidence without an explicit new destination.
    if (output_dir / "test" / "metrics.json").exists():
        parser.error(f"У {output_dir} вже є звіт. Вкажіть нову папку через --out, щоб не затерти докази.")
    print("[1/5] Перевірка офіційних CSV...")
    check_train = DatasetLoader().load(train_file, labels_required=True)
    check_test = DatasetLoader().load(test_file, labels_required=True)
    if len(check_train.table) != EXPECTED_TRAIN_ROWS or len(check_test.table) != EXPECTED_TEST_ROWS:
        parser.error("Кількість записів не відповідає офіційним файлам: train=175341, test=82332")
    print("[2/5] Навчання двох класифікаторів і внутрішня валідація...")
    train_run = ModelTrainer().run(train_file, REPO_ROOT / "runs", seed=args.seed)
    print(f"Навчання: {train_run}")
    print("[3/5] Незалежне тестування замороженої моделі...")
    test_run = PredictionService.run(train_run, test_file, REPO_ROOT / "runs", require_labels=True)
    print(f"Тест: {test_run}")
    meta = load_json(train_run / "run_config.json")
    metrics = load_json(test_run / "metrics.json")
    if metrics["rows"] != EXPECTED_TEST_ROWS:
        raise RuntimeError("Неочікувана кількість рядків у результатах тестування")
    print("[4/5] Копіювання машинних файлів без наборів даних і моделей...")
    files_train = copy_evidence(train_run, output_dir / "train", (
        "run_config.json", "validation_comparison.csv", "events.jsonl"))
    files_test = copy_evidence(test_run, output_dir / "test", (
        "metrics.json", "run_config.json", "confusion_matrix.png", "per_attack_category.csv", "events.jsonl"))
    if any(x not in files_train for x in ("run_config.json", "validation_comparison.csv", "events.jsonl")):
        raise RuntimeError("Неповні файли доказів навчання")
    if any(x not in files_test for x in ("metrics.json", "events.jsonl", "confusion_matrix.png")):
        raise RuntimeError("Неповні файли доказів тесту")
    manifest = {
        "protocol": "Official UNSW-NB15 train/test; internal 80/20 stratified validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_commit(),
        "seed": args.seed,
        "train_run": train_run.name,
        "test_run": test_run.name,
        "train_dataset_sha256": meta["training_sha256"],
        "test_dataset_sha256": metrics["dataset_sha256"],
        "train_evidence_files": files_train,
        "test_evidence_files": files_test,
    }
    save_json(output_dir / "manifest.json", manifest)
    create_summary(output_dir / "SUMMARY.md", meta, metrics, train_run, test_run)
    print("[5/5] Готово. Звіт:", output_dir / "SUMMARY.md")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ПОМИЛКА: {exc}", file=sys.stderr)
        raise
