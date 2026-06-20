"""Train and compare classical ML models for CIC-IDS2017 binary classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aics_ids_pipeline import (
    best_model_name,
    build_models,
    clean_dataset,
    find_csv_files,
    load_csv_files,
    prepare_binary_dataset,
    save_model,
    train_and_evaluate,
    tune_hist_gradient_boosting,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare multiple CIC-IDS2017 binary classification models."
    )
    parser.add_argument(
        "--data-dir",
        default="data/dataset",
        help="Folder containing the CSV files to train on.",
    )
    parser.add_argument("--models-dir", default="models", help="Folder for trained model output.")
    parser.add_argument("--results-dir", default="models", help="Folder for model comparison output.")
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=75000,
        help="Random rows to sample per CSV file. Use 0 for all rows.",
    )
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split size.")
    parser.add_argument(
        "--max-tuning-rows",
        type=int,
        default=30000,
        help="Maximum training rows used for cross-validation hyperparameter tuning.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip GridSearchCV tuning and only train the default model set.",
    )
    args = parser.parse_args()

    rows_per_file = None if args.rows_per_file == 0 else args.rows_per_file

    # Use every CSV currently placed in the active dataset folder.
    csv_files = find_csv_files(args.data_dir)

    if not csv_files:
        print("No dataset files found yet.")
        print("Place CIC-IDS2017 CSV files in data/dataset, then run this again.")
        return

    print("Dataset files used for this run:", flush=True)
    for csv_file in csv_files:
        print(f"- {csv_file}", flush=True)
    print(f"Rows per file: {'all' if rows_per_file is None else rows_per_file}", flush=True)

    # Load, clean, and split the data before any model sees it.
    df = load_csv_files(csv_files, rows_per_file=rows_per_file)
    cleaned = clean_dataset(df)
    prepared = prepare_binary_dataset(
        cleaned,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    print(f"Rows after cleaning: {len(cleaned):,}", flush=True)
    print("Training class balance:", flush=True)
    print(prepared.y_train.value_counts().to_string(), flush=True)
    print("Validation class balance:", flush=True)
    print(prepared.y_validation.value_counts().to_string(), flush=True)
    print("Test class balance:", flush=True)
    print(prepared.y_test.value_counts().to_string(), flush=True)

    # Train the baseline and stronger classical ML models on the same split.
    results = train_and_evaluate(prepared, build_models())

    cv_results = None
    if not args.skip_tuning:
        # Tune only the selected gradient-boosting model to keep runtime realistic.
        print("Tuning HistGradientBoostingClassifier with 3-fold GridSearchCV...", flush=True)
        tuned_result, cv_results = tune_hist_gradient_boosting(
            prepared,
            max_tuning_rows=args.max_tuning_rows,
        )
        print(f"Best tuning parameters: {tuned_result['best_params']}", flush=True)
        print(f"Best CV weighted F1: {tuned_result['best_cv_weighted_f1']:.10f}", flush=True)

    summary_rows = []

    # Flatten the nested sklearn reports into one comparison table.
    for name, result in results.items():
        validation_report = result["validation_classification_report"]
        test_report = result["test_classification_report"]
        summary_rows.append(
            {
                "model": name,
                "validation_accuracy": result["validation_accuracy"],
                "validation_balanced_accuracy": result["validation_balanced_accuracy"],
                "validation_macro_precision": validation_report["macro avg"]["precision"],
                "validation_macro_recall": validation_report["macro avg"]["recall"],
                "validation_macro_f1": validation_report["macro avg"]["f1-score"],
                "validation_weighted_precision": validation_report["weighted avg"]["precision"],
                "validation_weighted_recall": validation_report["weighted avg"]["recall"],
                "validation_weighted_f1": validation_report["weighted avg"]["f1-score"],
                "test_accuracy": result["test_accuracy"],
                "test_balanced_accuracy": result["test_balanced_accuracy"],
                "test_macro_precision": test_report["macro avg"]["precision"],
                "test_macro_recall": test_report["macro avg"]["recall"],
                "test_macro_f1": test_report["macro avg"]["f1-score"],
                "test_weighted_precision": test_report["weighted avg"]["precision"],
                "test_weighted_recall": test_report["weighted avg"]["recall"],
                "test_weighted_f1": test_report["weighted avg"]["f1-score"],
                "cv_weighted_f1": result.get("best_cv_weighted_f1"),
                "best_params": result.get("best_params"),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("validation_weighted_f1", ascending=False)
    print(summary.to_string(index=False))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "model_comparison.csv"
    summary.to_csv(summary_path, index=False)

    if cv_results is not None:
        tuning_path = results_dir / "hist_gradient_boosting_tuning.csv"
        cv_results.to_csv(tuning_path, index=False)
    else:
        tuning_path = None

    matrix_path = results_dir / "confusion_matrices.txt"
    with matrix_path.open("w", encoding="utf-8") as matrix_file:
        for name, result in results.items():
            # Save matrices in a human-readable format for report checking.
            matrix_file.write(f"{name}\n")
            matrix_file.write("Validation confusion matrix:\n")
            matrix_file.write("[[Benign predicted as Benign, Benign predicted as Malicious],\n")
            matrix_file.write(" [Malicious predicted as Benign, Malicious predicted as Malicious]]\n")
            matrix_file.write(f"{result['validation_confusion_matrix']}\n\n")
            matrix_file.write("Test confusion matrix:\n")
            matrix_file.write("[[Benign predicted as Benign, Benign predicted as Malicious],\n")
            matrix_file.write(" [Malicious predicted as Benign, Malicious predicted as Malicious]]\n")
            matrix_file.write(f"{result['test_confusion_matrix']}\n\n")

    winner = best_model_name(results)
    model_path = Path(args.models_dir) / f"{winner}.joblib"
    # Save the best validation model so it can be reloaded without retraining.
    save_model(results[winner]["model"], model_path)
    print(f"\nBest model: {winner}")
    print(f"Saved model: {model_path}")
    print(f"Saved comparison: {summary_path}")
    if tuning_path is not None:
        print(f"Saved tuning results: {tuning_path}")
    print(f"Saved confusion matrices: {matrix_path}")


if __name__ == "__main__":
    main()
