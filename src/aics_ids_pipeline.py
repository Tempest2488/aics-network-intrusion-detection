"""Reusable helpers for the AICS CIC-IDS2017 IDS experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "Label"
TUNING_CV_FOLDS = 3
TUNING_SCORING = "f1_weighted"
HIST_GRADIENT_BOOSTING_PARAM_GRID = {
    "learning_rate": [0.05, 0.08],
    "max_iter": [150, 200],
    "max_leaf_nodes": [31, 63],
}


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def find_csv_files(data_dir: str | Path) -> list[Path]:
    """Return CSV files under the data directory."""
    root = Path(data_dir)
    # The active dataset is controlled by the CSV files placed in this folder.
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def load_csv_files(
    csv_files: Iterable[Path],
    rows_per_file: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and combine CIC-IDS2017 CSV files, optionally sampling each file."""
    frames: list[pd.DataFrame] = []

    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, low_memory=False)

        # Sampling keeps local reruns practical; passing None uses every row.
        if rows_per_file is not None and len(frame) > rows_per_file:
            frame = frame.sample(n=rows_per_file, random_state=random_state)

        # Strip column names here so later preprocessing can use consistent names.
        frame.columns = frame.columns.str.strip()
        frame["source_file"] = csv_file.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No CSV files found. Place CIC-IDS2017 MachineLearningCSV files in the data folder."
        )

    return pd.concat(frames, ignore_index=True)


def find_label_column(df: pd.DataFrame) -> str:
    """Find the label column even if the source CSV has whitespace around the name."""
    for column in df.columns:
        if column.strip().lower() == LABEL_COLUMN.lower():
            return column
    raise KeyError("Could not find a Label column in the dataset.")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean common CIC-IDS2017 issues before modelling."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    label_column = find_label_column(cleaned)
    cleaned[label_column] = cleaned[label_column].astype(str).str.strip()

    # CIC-IDS2017 can contain infinite values and missing values after feature extraction.
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=0)
    cleaned = cleaned.drop_duplicates()

    return cleaned


def prepare_binary_dataset(
    df: pd.DataFrame,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """Prepare binary Benign/Malicious train, validation, and test data."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    label_column = find_label_column(df)

    # Convert the original attack names into the binary IDS target.
    y = df[label_column].apply(lambda value: "Benign" if value == "BENIGN" else "Malicious")
    X = df.drop(columns=[label_column])

    # The model uses numeric flow features only; text columns such as source_file are excluded.
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric feature columns were found for modelling.")

    X = X[numeric_columns]

    # Remove duplicate feature rows before splitting to reduce train/test leakage risk.
    unique_feature_mask = ~X.duplicated(keep="first")
    X = X.loc[unique_feature_mask]
    y = y.loc[unique_feature_mask]

    # First hold out the final test split, then split the remaining data for validation.
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_train_validation,
    )

    return PreparedData(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_models(random_state: int = 42) -> dict[str, object]:
    """Create classical ML models for comparison."""
    return {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent",
            random_state=random_state,
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=100,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            class_weight="balanced",
            max_features="sqrt",
            n_estimators=300,
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
    }


def train_and_evaluate(
    prepared: PreparedData,
    models: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Train each model and collect validation and test evaluation results."""
    results: dict[str, dict[str, object]] = {}

    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(prepared.X_train, prepared.y_train)

        # Validation scores select the model; test scores estimate final performance.
        validation_predictions = model.predict(prepared.X_validation)
        test_predictions = model.predict(prepared.X_test)

        results[name] = {
            "model": model,
            "validation_accuracy": accuracy_score(prepared.y_validation, validation_predictions),
            "validation_balanced_accuracy": balanced_accuracy_score(
                prepared.y_validation,
                validation_predictions,
            ),
            "validation_classification_report": classification_report(
                prepared.y_validation,
                validation_predictions,
                output_dict=True,
                zero_division=0,
            ),
            "validation_confusion_matrix": confusion_matrix(
                prepared.y_validation,
                validation_predictions,
                labels=["Benign", "Malicious"],
            ),
            "test_accuracy": accuracy_score(prepared.y_test, test_predictions),
            "test_balanced_accuracy": balanced_accuracy_score(prepared.y_test, test_predictions),
            "test_classification_report": classification_report(
                prepared.y_test,
                test_predictions,
                output_dict=True,
                zero_division=0,
            ),
            "test_confusion_matrix": confusion_matrix(
                prepared.y_test,
                test_predictions,
                labels=["Benign", "Malicious"],
            ),
        }

    return results


def evaluate_model(prepared: PreparedData, model: object) -> dict[str, object]:
    """Evaluate a fitted model on the validation and held-out test splits."""
    validation_predictions = model.predict(prepared.X_validation)
    test_predictions = model.predict(prepared.X_test)

    return {
        "model": model,
        "validation_accuracy": accuracy_score(prepared.y_validation, validation_predictions),
        "validation_balanced_accuracy": balanced_accuracy_score(
            prepared.y_validation,
            validation_predictions,
        ),
        "validation_classification_report": classification_report(
            prepared.y_validation,
            validation_predictions,
            output_dict=True,
            zero_division=0,
        ),
        "validation_confusion_matrix": confusion_matrix(
            prepared.y_validation,
            validation_predictions,
            labels=["Benign", "Malicious"],
        ),
        "test_accuracy": accuracy_score(prepared.y_test, test_predictions),
        "test_balanced_accuracy": balanced_accuracy_score(prepared.y_test, test_predictions),
        "test_classification_report": classification_report(
            prepared.y_test,
            test_predictions,
            output_dict=True,
            zero_division=0,
        ),
        "test_confusion_matrix": confusion_matrix(
            prepared.y_test,
            test_predictions,
            labels=["Benign", "Malicious"],
        ),
    }


def tune_hist_gradient_boosting(
    prepared: PreparedData,
    max_tuning_rows: int = 30000,
    random_state: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Tune HistGradientBoosting with cross-validation, then test the tuned model."""
    if len(prepared.X_train) > max_tuning_rows:
        # Cross-validation is capped so tuning remains feasible on a laptop.
        X_tune, _, y_tune, _ = train_test_split(
            prepared.X_train,
            prepared.y_train,
            train_size=max_tuning_rows,
            random_state=random_state,
            stratify=prepared.y_train,
        )
    else:
        X_tune = prepared.X_train
        y_tune = prepared.y_train

    search = GridSearchCV(
        estimator=HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        param_grid=HIST_GRADIENT_BOOSTING_PARAM_GRID,
        scoring=TUNING_SCORING,
        cv=TUNING_CV_FOLDS,
        n_jobs=-1,
    )
    search.fit(X_tune, y_tune)

    # Refit the chosen parameters on the full training split before evaluation.
    tuned_model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=random_state,
        **search.best_params_,
    )
    tuned_model.fit(prepared.X_train, prepared.y_train)

    result = evaluate_model(prepared, tuned_model)
    result["best_params"] = search.best_params_
    result["best_cv_weighted_f1"] = search.best_score_

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return result, cv_results


def best_model_name(results: dict[str, dict[str, object]]) -> str:
    """Choose the model with the highest validation weighted F1-score."""
    return max(
        results,
        key=lambda name: results[name]["validation_classification_report"]["weighted avg"]["f1-score"],
    )


def save_model(model: object, output_path: str | Path) -> None:
    """Save a trained model to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
