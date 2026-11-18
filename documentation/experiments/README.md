# Експерименти й відтворюваність

Тут розміщуються **машинні результати реального запуску** IDS на офіційних CSV UNSW-NB15. Репозиторій не включає самі датасети, повні прогнози (`predictions.csv`) і `model.joblib`.

1. Завантажте два офіційні CSV з [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
2. Покладіть їх до `data/`.
3. Виконайте `run_official_experiment_windows.bat` або `python scripts/run_official_experiment.py`.
4. Після успішного запуску у `official_unsw_nb15/` з'являться `SUMMARY.md`, `manifest.json`, підпапки `train/` та `test/`.
5. Лише після цього синхронізуйте **фактичні** звіти на GitHub.

Докладні показники, **перенесені з диплома** (а не з повторного запуску), наведені в основному [`README.md`](../../README.md).

⚠️ Поки у `official_unsw_nb15/` немає `train/events.jsonl` і `test/metrics.json`, **не слід стверджувати, що поточний GitHub уже містить журнали відтвореного експерименту**.
