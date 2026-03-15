"""
Random Forest Classifier Training
Trains a Random Forest on CIC-IDS2017 data with optional hyperparameter tuning.

Usage:
    python -m src.models.train_rf
    python -m src.models.train_rf --test-mode          # Quick smoke test
    python -m src.models.train_rf --tune               # With GridSearchCV
"""

import argparse
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.data.feature_config import RF_MODEL_FILE, NUM_CLASSES
from src.models.model_utils import (
    load_processed_data, evaluate_model, save_sklearn_model,
    plot_confusion_matrix,
)


def train_random_forest(X_train, y_train, tune=False, test_mode=False):
    """Train Random Forest classifier."""
    print("\n-- Training Random Forest --")

    if tune and not test_mode:
        # Hyperparameter grid search
        print("  Running GridSearchCV (this may take a while)...")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_rf, param_grid,
            cv=3, scoring="f1_macro",
            n_jobs=-1, verbose=1,
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
    else:
        # Default or smoke test parameters
        n_estimators = 10 if test_mode else 200
        max_depth = 5 if test_mode else 20

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest")
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick smoke test with small model")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter grid search")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    print("=" * 60)
    print("Random Forest Classifier Training")
    print("=" * 60)

    # Load data
    data = load_processed_data(args.data_dir)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    if args.test_mode:
        # Use small subset
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:200]
        y_test = y_test[:200]
        print("  * Test mode: using small subset")

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Train
    start = time.time()
    model = train_random_forest(X_train, y_train, tune=args.tune, test_mode=args.test_mode)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = evaluate_model(y_test, y_pred, y_prob, model_name="Random Forest")

    # Feature importance (top 15)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print("\n  Top 15 Features:")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:2d}. Feature[{idx}]: {importances[idx]:.4f}")

    # Save model
    save_sklearn_model(model, RF_MODEL_FILE)

    # Save confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Random Forest", "models/rf_confusion_matrix.png")

    print("\n[OK] Random Forest training complete!")


if __name__ == "__main__":
    main()
