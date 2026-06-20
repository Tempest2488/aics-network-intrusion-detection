"""Inspect CIC-IDS2017 CSV files before modelling."""

from __future__ import annotations

import argparse
from aics_ids_pipeline import (
    clean_dataset,
    find_csv_files,
    find_label_column,
    load_csv_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CIC-IDS2017 CSV files.")
    parser.add_argument(
        "--data-dir",
        default="data/dataset",
        help="Folder containing the CSV files to inspect.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=75000,
        help="Random rows to sample per CSV file for inspection. Use 0 for all rows.",
    )
    args = parser.parse_args()

    rows_per_file = None if args.rows_per_file == 0 else args.rows_per_file

    # Inspect whichever CSV files are currently in the active dataset folder.
    csv_files = find_csv_files(args.data_dir)

    print(f"CSV files found: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"- {csv_file}")

    if not csv_files:
        print("\nNo dataset files found yet.")
        print("Place CIC-IDS2017 CSV files in data/dataset, then run this again.")
        return

    # Load a sample by default so the inspection command runs quickly.
    df = load_csv_files(csv_files, rows_per_file=rows_per_file)
    print(f"\nLoaded shape before cleaning: {df.shape}")
    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    label_column = find_label_column(df)
    print("\nLabel counts before cleaning:")
    print(df[label_column].astype(str).str.strip().value_counts())

    # Show how cleaning affects row counts and the final binary target balance.
    cleaned = clean_dataset(df)
    print(f"\nShape after cleaning: {cleaned.shape}")
    print("\nBinary label counts after cleaning:")
    binary_labels = cleaned[label_column].apply(lambda value: "Benign" if value == "BENIGN" else "Malicious")
    print(binary_labels.value_counts())


if __name__ == "__main__":
    main()
