"""Streamlit local web interface. Run: streamlit run app.py."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ids_project.core import DatasetLoader, ModelTrainer, PredictionService, DataValidationError

APP_ROOT = Path("app_data")
UPLOAD_DIR = APP_ROOT / "uploads"
RUNS_DIR = APP_ROOT / "runs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="IDS — дослідницький прототип", page_icon="🛡️", layout="wide")
st.title("🛡️ Система виявлення вторгнень")
st.caption("Навчально-дослідницький прототип: класифікація мережевих потоків UNSW-NB15, без перехоплення живого трафіку")


def save_upload(uploaded, prefix: str) -> Path:
    path = UPLOAD_DIR / f"{prefix}_{uploaded.name}"
    # Names are trimmed to basename; Streamlit upload bytes are not executed.
    path = UPLOAD_DIR / Path(path.name).name
    path.write_bytes(uploaded.getvalue())
    return path


def show_report(run_path: Path) -> None:
    path = run_path / "metrics.json"
    if not path.exists():
        st.warning("Файл метрик відсутній")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Проаналізовано", report.get("rows", 0))
    c2.metric("Спрогнозовано атак", report.get("predicted_attack", 0))
    c3.metric("Поріг", f'{report.get("threshold", 0):.2f}')
    c4.metric("Модель", report.get("model", "—"))
    if "f1" in report:
        a, b, c, d = st.columns(4)
        a.metric("Precision", f'{report["precision"]:.4f}')
        b.metric("Recall", f'{report["recall"]:.4f}')
        c.metric("F1", f'{report["f1"]:.4f}')
        d.metric("FPR", f'{report["fpr"]:.4f}')
    image_path = run_path / "confusion_matrix.png"
    if image_path.exists():
        st.image(str(image_path), caption="Матриця помилок для незалежного тестування")
    csv_path = run_path / "predictions.csv"
    if csv_path.exists():
        sample = pd.read_csv(csv_path, nrows=300)
        st.subheader("Перші 300 прогнозів")
        st.dataframe(sample, use_container_width=True)
        st.download_button("Завантажити predictions.csv", csv_path.read_bytes(), file_name="predictions.csv", mime="text/csv")
    st.download_button("Завантажити metrics.json", path.read_bytes(), file_name="metrics.json", mime="application/json")
    st.caption(f"Каталог запуску: {run_path}")


tab_data, tab_train, tab_result = st.tabs(["Дані", "Навчання", "Результати"])
with tab_data:
    st.subheader("Перевірка структури даних")
    data_file = st.file_uploader("CSV UNSW-NB15", type=["csv"], key="inspect")
    if data_file and st.button("Перевірити CSV", key="check"):
        try:
            path = save_upload(data_file, "inspect")
            ds = DatasetLoader().load(path, labels_required=False)
            st.success(f"Файл коректний: {len(ds.table):,} рядків; SHA-256: {ds.checksum}")
            if "label" in ds.table:
                st.write("Розподіл міток:")
                st.dataframe(ds.table["label"].value_counts().rename_axis("Клас").to_frame("Кількість"))
            st.dataframe(ds.table.head(20), use_container_width=True)
        except (DataValidationError, ValueError, OSError) as error:
            st.error(str(error))
with tab_train:
    st.subheader("Навчання на офіційній навчальній вибірці")
    st.info("Використовуйте лише UNSW_NB15_training-set.csv. Параметри й поріг вибираються на внутрішній валідації, НЕ на тестовому файлі.")
    train_csv = st.file_uploader("Офіційний training-set.csv", type=["csv"], key="train")
    seed = st.number_input("Випадкове зерно", min_value=1, max_value=100000, value=42, step=1)
    if train_csv and st.button("Навчити й порівняти дві моделі", type="primary"):
        try:
            path = save_upload(train_csv, "train")
            with st.spinner("Навчання LogisticRegression і RandomForest; будь ласка, не закривайте вкладку"):
                run_dir = ModelTrainer().run(path, RUNS_DIR, seed=int(seed))
            st.session_state["model_dir"] = str(run_dir)
            st.success(f"Навчання завершено: {run_dir.name}")
            st.dataframe(pd.read_csv(run_dir / "validation_comparison.csv"), use_container_width=True)
            st.json(json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")))
        except (DataValidationError, ValueError, OSError) as error:
            st.error(str(error))
with tab_result:
    st.subheader("Перевірка збереженої моделі")
    model_dir = st.text_input("Каталог навченої моделі", value=st.session_state.get("model_dir", ""), placeholder="app_data/runs/20261110-...-train-...")
    mode = st.radio("Режим", options=["Незалежне тестування (з label)", "Прогнозування (без обов'язкових міток)"], horizontal=True)
    test_file = st.file_uploader("CSV для аналізу", type=["csv"], key="result")
    if test_file and st.button("Виконати аналіз", type="primary"):
        try:
            path = save_upload(test_file, "eval")
            with st.spinner("Аналіз записів"):
                run_dir = PredictionService.run(model_dir, path, RUNS_DIR, require_labels=mode.startswith("Незалежне"))
            st.session_state["result_dir"] = str(run_dir)
            st.success("Аналіз виконано, артефакти збережені")
        except (DataValidationError, ValueError, FileNotFoundError, OSError) as error:
            st.error(str(error))
    if st.session_state.get("result_dir"):
        show_report(Path(st.session_state["result_dir"]))

st.divider()
st.caption("Безпека: завантажуйте лише довірені моделі, створені цим застосунком. joblib не є безпечним форматом для чужих файлів.")
