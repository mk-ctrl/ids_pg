"""
XGBoost Classifier Training
Trains an XGBoost model with class imbalance handling.

Usage:
    python -m src.models.train_xgb
    python -m src.models.train_xgb --test-mode
    python -m src.models.train_xgb --tune
"""

import argparse
import time
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.data.feature_config import XGB_MODEL_FILE, NUM_CLASSES
from src.models.model_utils import (
    load_processed_data, evaluate_model, save_sklearn_model,
    plot_confusion_matrix,
)


def compute_sample_weights(y):
    """Compute sample weights to handle class imbalance."""
    counter = Counter(y)
    total = len(y)
    n_classes = len(counter)
    weights = {cls: total / (n_classes * count) for cls, count in counter.items()}
    return np.array([weights[yi] for yi in y])


def train_xgboost(X_train, y_train, tune=False, test_mode=False):
    """Train XGBoost classifier."""
    print("\n-- Training XGBoost --")

    sample_weights = compute_sample_weights(y_train)

    if tune and not test_mode:
        print("  Running GridSearchCV...")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        base_xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
        grid_search = GridSearchCV(
            base_xgb, param_grid,
            cv=3, scoring="f1_macro",
            n_jobs=-1, verbose=1,
        )
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        model = grid_search.best_estimator_
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
    else:
        n_estimators = 20 if test_mode else 200
        max_depth = 3 if test_mode else 6

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    print("=" * 60)
    print("XGBoost Classifier Training")
    print("=" * 60)

    data = load_processed_data(args.data_dir)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    if args.test_mode:
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:200]
        y_test = y_test[:200]
        print("  * Test mode: using small subset")

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Class distribution
    counter = Counter(y_train)
    print("  Class distribution (train):")
    for cls, count in sorted(counter.items()):
        print(f"    Class {cls}: {count:,}")

    # Train
    start = time.time()
    model = train_xgboost(X_train, y_train, tune=args.tune, test_mode=args.test_mode)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = evaluate_model(y_test, y_pred, y_prob, model_name="XGBoost")

    # Feature importance (top 15)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print("\n  Top 15 Features:")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:2d}. Feature[{idx}]: {importances[idx]:.4f}")

    # Save
    save_sklearn_model(model, XGB_MODEL_FILE)
    plot_confusion_matrix(y_test, y_pred, "XGBoost", "models/xgb_confusion_matrix.png")

    print("\n[OK] XGBoost training complete!")


if __name__ == "__main__":
    main()
