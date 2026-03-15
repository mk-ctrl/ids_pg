"""
CIC-IDS2017 Data Preprocessor
Loads raw CSVs, cleans data, maps labels, normalizes features, and splits into train/test.

Usage:
    python -m src.data.preprocess
    python -m src.data.preprocess --test-mode          # Quick smoke test with sample
    python -m src.data.preprocess --reduced             # Use top features only
    python -m src.data.preprocess --input-dir data/raw --output-dir data/processed
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.feature_config import (
    RAW_FEATURE_COLUMNS, LABEL_COLUMN, LABEL_MAPPING,
    CLASS_TO_ID, TOP_FEATURES, DATA_RAW_DIR, DATA_PROCESSED_DIR,
    SCALER_FILE, MODEL_DIR,
)

warnings.filterwarnings("ignore")


def load_raw_data(input_dir: Path, sample_size: int = None) -> pd.DataFrame:
    """Load and concatenate all CIC-IDS2017 CSV files."""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"[FAIL] No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files:")
    frames = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}...", end=" ")
        try:
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            # Clean column names (strip whitespace)
            df.columns = df.columns.str.strip()
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            frames.append(df)
            print(f"({len(df):,} rows)")
        except Exception as e:
            print(f"SKIP - {e}")
            continue

    if not frames:
        print("[FAIL] Could not load any CSV files")
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(data):,}")
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle missing values, infinities, and duplicates."""
    initial_rows = len(df)
    print("\n-- Cleaning Data --")

    # Strip whitespace from label column
    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip()
        # Normalize dash-like characters (en-dash, em-dash, encoding artifacts)
        df[LABEL_COLUMN] = df[LABEL_COLUMN].str.replace(r'[–—\x96\u2013\u2014]', '-', regex=True)
        df[LABEL_COLUMN] = df[LABEL_COLUMN].str.replace('Â', '', regex=False)

    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Convert numeric columns to proper types
    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that became NaN after numeric conversion
    df.dropna(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    print(f"  Rows before: {initial_rows:,}")
    print(f"  Rows after:  {len(df):,}")
    print(f"  Removed:     {initial_rows - len(df):,}")

    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw CIC-IDS2017 labels to our 5-class taxonomy."""
    print("\n-- Mapping Labels --")

    # Show original label distribution
    print("  Original labels:")
    for label, count in df[LABEL_COLUMN].value_counts().items():
        print(f"    {label}: {count:,}")

    # Apply mapping
    df["attack_class"] = df[LABEL_COLUMN].map(LABEL_MAPPING)

    # Handle unmapped labels
    unmapped = df["attack_class"].isna().sum()
    if unmapped > 0:
        unmapped_labels = df[df["attack_class"].isna()][LABEL_COLUMN].unique()
        print(f"\n  [WARN] {unmapped} rows with unmapped labels: {unmapped_labels}")
        df = df.dropna(subset=["attack_class"])

    # Convert to numeric IDs
    df["class_id"] = df["attack_class"].map(CLASS_TO_ID)

    # Show mapped distribution
    print("\n  Mapped classes:")
    for cls, count in df["attack_class"].value_counts().items():
        print(f"    {cls}: {count:,}")

    return df


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_path: Path = None,
) -> tuple:
    """Normalize features using StandardScaler."""
    print("\n-- Normalizing Features --")
    print(f"  Features: {len(feature_cols)}")

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for inference
    if scaler_path:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved: {scaler_path}")

    return X_scaled, scaler


def main():
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS2017 dataset")
    parser.add_argument("--input-dir", type=str, default=DATA_RAW_DIR)
    parser.add_argument("--output-dir", type=str, default=DATA_PROCESSED_DIR)
    parser.add_argument("--test-mode", action="store_true",
                        help="Use small sample for smoke testing")
    parser.add_argument("--reduced", action="store_true",
                        help="Use top features only (faster training)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CIC-IDS2017 Data Preprocessor")
    print("=" * 60)

    # 1. Load data
    sample_size = 1000 if args.test_mode else None
    df = load_raw_data(input_dir, sample_size=sample_size)

    # 2. Clean data
    df = clean_data(df)

    # 3. Map labels
    df = map_labels(df)

    # 4. Select features
    if args.reduced:
        feature_cols = [c for c in TOP_FEATURES if c in df.columns]
    else:
        feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]

    print(f"\n  Using {len(feature_cols)} features")

    # 5. Normalize
    scaler_path = Path(MODEL_DIR) / SCALER_FILE
    X_scaled, scaler = normalize_features(df, feature_cols, scaler_path)
    y = df["class_id"].values

    # 6. Train/test split (stratified)
    print(f"\n-- Splitting Data ({1 - args.test_size:.0%} / {args.test_size:.0%}) --")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    print(f"  Train set: {X_train.shape[0]:,} samples")
    print(f"  Test set:  {X_test.shape[0]:,} samples")

    # 7. Save processed data
    print(f"\n-- Saving to {output_dir} --")

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    # Save feature names for reference
    with open(output_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_cols))

    print(f"  [OK] X_train.npy: {X_train.shape}")
    print(f"  [OK] X_test.npy:  {X_test.shape}")
    print(f"  [OK] y_train.npy: {y_train.shape}")
    print(f"  [OK] y_test.npy:  {y_test.shape}")
    print(f"  [OK] feature_names.txt")

    # Summary
    print("\n" + "=" * 60)
    print("[OK] Preprocessing complete!")
    print(f"  Features:  {len(feature_cols)}")
    print(f"  Classes:   {len(CLASS_TO_ID)}")
    print(f"  Train:     {X_train.shape[0]:,}")
    print(f"  Test:      {X_test.shape[0]:,}")
    print("=" * 60)
    print("\nNext: Train models with:")
    print("  python -m src.models.train_rf")
    print("  python -m src.models.train_xgb")
    print("  python -m src.models.train_dnn")
    print("  python -m src.models.train_ensemble")


if __name__ == "__main__":
    main()
