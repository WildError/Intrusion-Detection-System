# Експерименти й відтворюваність

У каталозі наведено результати навчання та незалежного тестування системи IDS на офіційних вибірках UNSW-NB15. Основні показники збережено у [SUMMARY.md](official_unsw_nb15/SUMMARY.md), детальні метрики — у [metrics.json](official_unsw_nb15/test/metrics.json).

1. Завантажте два офіційні CSV з [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
2. Покладіть їх до `data/`.
3. Виконайте `python scripts/run_official_experiment.py`.
4. Результати зберігаються у `official_unsw_nb15/`: `SUMMARY.md`, `manifest.json`, `train/` та `test/`.

Порівняння моделей, результати валідації та незалежного тестування описані також в основному [`README.md`](../../README.md).
