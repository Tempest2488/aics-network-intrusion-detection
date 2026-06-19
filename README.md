# AICS CIC-IDS2017 Machine Learning IDS

This project builds and evaluates a classical machine learning intrusion detection system using the CIC-IDS2017 machine-learning CSV flow data. The task is binary classification: each network-flow record is classified as either `Benign` or `Malicious`.

The implementation compares several supervised learning models, applies hyperparameter tuning to the best-performing model family, evaluates the final model on a held-out test set, and saves the trained model and generated evidence for reproducibility.

## Project Summary

- Dataset: CIC-IDS2017 `MachineLearningCSV` flow-feature files
- Task: binary intrusion detection, `Benign` vs `Malicious`
- Models: `DummyClassifier`, Logistic Regression, Random Forest, Extra Trees, and `HistGradientBoostingClassifier`
- Split: stratified `70/20/10` train, validation, and test split
- Best model: `HistGradientBoostingClassifier`
- Final held-out test accuracy: approximately `99.9536%`
- Final held-out weighted F1-score: approximately `0.999536`

The high accuracy is interpreted carefully in the report because the experiment uses a selected benchmark subset and binary classification rather than live multi-class intrusion detection.

## Repository Structure

```text
.
├── config/
│   └── selected_csvs.txt
├── data/
│   └── MachineLearningCVE/
├── figures/
├── models/
├── notebooks/
│   └── 01_binary_classification.ipynb
├── report/
│   └── AICS_Coursework_Report_Draft.docx
├── src/
│   ├── aics_ids_pipeline.py
│   ├── inspect_dataset.py
│   ├── test_saved_model.py
│   └── train_binary_models.py
├── README.md
└── requirements.txt
```

## Dataset

The code expects the selected CIC-IDS2017 CSV files to be available under:

```text
data/MachineLearningCVE/
```

The selected files are listed in `config/selected_csvs.txt`:

```text
MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv
MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv
MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

These files come from the CIC-IDS2017 `MachineLearningCSV` archive.

Important: some dataset CSV files are larger than GitHub's normal per-file limit. If the dataset is stored in this GitHub repository, use Git LFS for the CSV files. Otherwise, place the full project folder in OneDrive and share that link instead.

## Setup

Create a virtual environment and install the required packages:

```fish
uv venv .venv
uv pip install -r requirements.txt
```

If using standard Python tooling instead of `uv`:

```fish
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the Notebook

Open the main notebook:

```fish
.venv/bin/jupyter notebook notebooks/01_binary_classification.ipynb
```

The notebook contains the executed experiment outputs, including tables, graphs, model comparison results, and final evaluation evidence.

## Run the Scripts

Inspect the selected dataset files:

```fish
.venv/bin/python src/inspect_dataset.py
```

Train and compare the models:

```fish
.venv/bin/python src/train_binary_models.py
```

Reload the saved best model and reproduce the final test metrics:

```fish
.venv/bin/python src/test_saved_model.py
```

Run a faster training check without the GridSearchCV tuning step:

```fish
.venv/bin/python src/train_binary_models.py --skip-tuning
```

## Outputs

Generated outputs are saved into:

- `models/model_comparison.csv`
- `models/confusion_matrices.txt`
- `models/hist_gradient_boosting_tuning.csv`
- `models/hist_gradient_boosting.joblib`
- `figures/class_balance.png`
- `figures/model_comparison_weighted_f1.png`
- `figures/best_model_confusion_matrix.png`
- `figures/feature_importance.png`
