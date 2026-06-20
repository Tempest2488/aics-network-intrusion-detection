"""Reload a saved model and reproduce test-set metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from aics_ids_pipeline import (
    clean_dataset,
    find_csv_files,
    load_csv_files,
    prepare_binary_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a saved CIC-IDS2017 binary classification model.")
    parser.add_argument(
        "--data-dir",
        default="data/dataset",
        help="Folder containing the CSV files to test with.",
    )
    parser.add_argument(
        "--model-path",
        default="models/hist_gradient_boosting.joblib",
        help="Saved .joblib model to load and test.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=75000,
        help="Random rows to sample per CSV file. Use 0 for all rows.",
    )
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split size.")
    args = parser.parse_args()

    rows_per_file = None if args.rows_per_file == 0 else args.rows_per_file

    # Recreate the same preprocessing and split process used during training.
    csv_files = find_csv_files(args.data_dir)

    if not csv_files:
        print("No dataset files found yet.")
        print("Place CIC-IDS2017 CSV files in data/dataset, then run this again.")
        return

    df = load_csv_files(csv_files, rows_per_file=rows_per_file)
    cleaned = clean_dataset(df)
    prepared = prepare_binary_dataset(
        cleaned,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )

    model_path = Path(args.model_path)
    model = joblib.load(model_path)

    # The saved model is evaluated on the held-out test split only.
    predictions = model.predict(prepared.X_test)

    accuracy = accuracy_score(prepared.y_test, predictions)
    report = classification_report(prepared.y_test, predictions, zero_division=0)
    matrix = confusion_matrix(prepared.y_test, predictions, labels=["Benign", "Malicious"])
    matrix_table = pd.DataFrame(
        matrix,
        index=["Actual Benign", "Actual Malicious"],
        columns=["Predicted Benign", "Predicted Malicious"],
    )

    print(f"Loaded model: {model_path}")
    print(f"Test accuracy: {accuracy:.6f} ({accuracy * 100:.4f}%)")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix_table.to_string())


if __name__ == "__main__":
    main()
