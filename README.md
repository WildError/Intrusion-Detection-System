# IDS: reproducible binary classification of UNSW-NB15 flows

**Purpose.** Academic prototype for **offline** flow-level classification of UNSW-NB15 records. It does not capture packets or provide live IPS/IDS protection. The output `predicted_label=1` means the model flagged a flow; it is not proof that an actual breach occurred.

## Official source of research data

University of New South Wales — [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Download from the official university download link, and obtain **both** `UNSW_NB15_training-set.csv` (175,341 records) and `UNSW_NB15_testing-set.csv` (82,332 records). Dataset is intentionally **not included** in this archive. Cite dataset authors Nour Moustafa and Jill Slay (MilCIS 2015) in the thesis.

## Windows 11 installation (Python 3.11 recommended)

In PowerShell or CMD opened in the project directory:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
mkdir data
```

Copy official files into `data/` without renaming.

## Mandatory reproducible experiment

```bat
python -m ids_project.cli inspect --file data/UNSW_NB15_training-set.csv
python -m ids_project.cli train --train data/UNSW_NB15_training-set.csv --out runs --seed 42
```

The second command prints the created `runs/YYYYMMDD-HHMMSS-train-XXXXXXXX` directory. Substitute that exact directory in the next command:

```bat
python -m ids_project.cli evaluate --model runs/YYYYMMDD-HHMMSS-train-XXXXXXXX --input data/UNSW_NB15_testing-set.csv --out runs
```

Do not tune or retrain after viewing the official test results unless documenting a new protocol and keeping a fresh independent test set. To predict a compatible unlabeled CSV instead of measuring known labels:

```bat
python -m ids_project.cli predict --model runs/YYYYMMDD-HHMMSS-train-XXXXXXXX --input data/new_flows.csv --out runs
```

## Graphical application

```bat
python -m streamlit run app.py
```

The Streamlit interface has three tabs: Data, Training, Results. Navigate in browser to the address printed by Streamlit; for local-only use prefer `python -m streamlit run app.py --server.address 127.0.0.1`.

## What the code actually does

1. Checks that all 42 expected features are present and types are usable.
2. Removes `id`, `attack_cat`, and `label` from the ML feature matrix to reduce target leakage.
3. Stratifies the **official training** CSV 80/20 (fixed random seed 42); learns all imputers/encoders/scalers only from the internal fitting partition via sklearn Pipeline.
4. Fits Logistic Regression and Random Forest (60 trees, depth 14), selecting a decision threshold 0.05–0.95 based on validation F1; ties are decided by lower FPR and then higher threshold. Selects a model on **validation only**.
5. Saves the selected model and metadata, including dataset SHA-256, feature list, library versions and validation comparison.
6. Applies the frozen model to the **independent official test** CSV. Saves `predictions.csv`, `metrics.json`, `run_config.json`, `confusion_matrix.png`, `per_attack_category.csv`, and JSON-lines event log.
7. When the CSV has no `label`, predicts but does not manufacture precision, recall or F1 measurements.

## Test suite (engineering validation, not scientific benchmark)

```bat
python -m pytest -q
```

The tests use tiny **synthetic software fixtures** intentionally designed to verify parsing, reject invalid input, evaluate metric arithmetic, produce output files and prevent direct training-file reuse as a test. Test results from those fixtures **MUST NEVER** be reported as UNSW-NB15 IDS effectiveness metrics.

## Security, scope and limitations

- Joblib is Python pickle serialization. Only load model files generated locally by the application and stored in a trusted directory. Do not open untrusted models from the Internet.
- The prototype works with **precomputed UNSW-NB15 features**. It does not automatically reconstruct all 42 features from live packets, support arbitrary CIC-IDS CSV columns or enforce firewall rules.
- Different distribution, network, protocol mixes and time period may change performance; official UNSW test scores do not measure real-world detection in your network.
- `attack_cat` is **never** an ML input; used solely for post-hoc error breakdown when official labels are present.
- No dataset or full experimental output is bundled. For the academic report you must run the official experiment and insert the **actual values and figures**, listing hashes and software versions.

## Mapping to project design (chapter 2)

| Designed component | Source implementation |
|---|---|
| DatasetLoader / SchemaValidator | `ids_project/core.py` |
| PipelineFactory / ModelTrainer / ThresholdSelector | `ids_project/core.py` |
| PredictionService / Metrics calculation | `ids_project/core.py` |
| CLI interface | `ids_project/cli.py` |
| Streamlit interface | `app.py` |
| Software tests | `tests/test_core.py` |

The package contains the executable IDS prototype and reproducible software checks. Official UNSW-NB15 CSV files are obtained separately from the university source.
