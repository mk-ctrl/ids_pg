"""
Stacking Ensemble Meta-Classifier Training
Combines predictions from RF, XGBoost, and DNN using a Logistic Regression meta-learner.

Usage:
    python -m src.models.train_ensemble
    python -m src.models.train_ensemble --test-mode
"""

import argparse
import time
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.data.feature_config import (
    RF_MODEL_FILE, XGB_MODEL_FILE, DNN_MODEL_FILE,
    META_MODEL_FILE, NUM_CLASSES,
)
from src.models.model_utils import (
    load_processed_data, load_sklearn_model,
    evaluate_model, save_sklearn_model, plot_confusion_matrix,
)


def generate_meta_features(X, rf_model, xgb_model, dnn_model):
    """
    Generate meta-features by stacking base model predictions.
    For each sample, the meta-feature vector is the concatenation of
    probability outputs from all three base models.
    
    Shape: (n_samples, NUM_CLASSES * 3)
    """
    # Random Forest probabilities
    rf_probs = rf_model.predict_proba(X)

    # XGBoost probabilities
    xgb_probs = xgb_model.predict_proba(X)

    # DNN probabilities
    dnn_probs = dnn_model.predict_proba(X)

    # Ensure all have the same number of classes
    # (handle case where a model hasn't seen all classes)
    def pad_probs(probs, n_classes):
        if probs.shape[1] < n_classes:
            padded = np.zeros((probs.shape[0], n_classes))
            padded[:, :probs.shape[1]] = probs
            return padded
        return probs

    rf_probs = pad_probs(rf_probs, NUM_CLASSES)
    xgb_probs = pad_probs(xgb_probs, NUM_CLASSES)
    dnn_probs = pad_probs(dnn_probs, NUM_CLASSES)

    # Stack horizontally: [RF_probs | XGB_probs | DNN_probs]
    meta_features = np.hstack([rf_probs, xgb_probs, dnn_probs])
    return meta_features


def train_meta_classifier(meta_X_train, y_train, test_mode=False):
    """Train the Logistic Regression meta-classifier."""
    print("\n-- Training Meta-Classifier (Logistic Regression) --")

    meta_model = LogisticRegression(
        max_iter=500 if not test_mode else 50,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        C=1.0,
    )

    # Cross-validation score on meta-features
    if not test_mode:
        cv_scores = cross_val_score(
            meta_model, meta_X_train, y_train,
            cv=5, scoring="f1_macro", n_jobs=-1,
        )
        print(f"  CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    meta_model.fit(meta_X_train, y_train)
    return meta_model


def main():
    parser = argparse.ArgumentParser(description="Train Stacking Ensemble")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    print("=" * 60)
    print("Stacking Ensemble Meta-Classifier Training")
    print("=" * 60)

    # 1. Load base models
    print("\n-- Loading Base Models --")
    try:
        rf_model = load_sklearn_model(RF_MODEL_FILE)
        print(f"  [OK] Random Forest loaded")
    except FileNotFoundError:
        print(f"  [FAIL] Random Forest not found. Run: python -m src.models.train_rf")
        return

    try:
        xgb_model = load_sklearn_model(XGB_MODEL_FILE)
        print(f"  [OK] XGBoost loaded")
    except FileNotFoundError:
        print(f"  [FAIL] XGBoost not found. Run: python -m src.models.train_xgb")
        return

    try:
        dnn_model = load_sklearn_model(DNN_MODEL_FILE)
        print(f"  [OK] DNN loaded")
    except FileNotFoundError:
        print(f"  [FAIL] DNN not found. Run: python -m src.models.train_dnn")
        return

    # 2. Load data
    data = load_processed_data(args.data_dir)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    if args.test_mode:
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:200]
        y_test = y_test[:200]
        print("  * Test mode: using small subset")

    # 3. Generate meta-features
    print("\n-- Generating Meta-Features --")
    start = time.time()
    meta_X_train = generate_meta_features(X_train, rf_model, xgb_model, dnn_model)
    meta_X_test = generate_meta_features(X_test, rf_model, xgb_model, dnn_model)
    elapsed = time.time() - start
    print(f"  Meta-feature shape: {meta_X_train.shape}")
    print(f"  Generation time: {elapsed:.1f}s")

    # 4. Train meta-classifier
    start = time.time()
    meta_model = train_meta_classifier(meta_X_train, y_train, test_mode=args.test_mode)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    # 5. Evaluate ensemble
    y_pred = meta_model.predict(meta_X_test)
    y_prob = meta_model.predict_proba(meta_X_test)

    print("\n" + "-" * 50)
    print("INDIVIDUAL MODEL RESULTS (for comparison)")
    print("-" * 50)

    # Compare with individual models
    rf_pred = rf_model.predict(X_test)
    evaluate_model(y_test, rf_pred, model_name="Random Forest (Base)")

    xgb_pred = xgb_model.predict(X_test)
    evaluate_model(y_test, xgb_pred, model_name="XGBoost (Base)")

    dnn_pred = dnn_model.predict(X_test)
    evaluate_model(y_test, dnn_pred, model_name="DNN (Base)")

    print("\n" + "-" * 50)
    print("ENSEMBLE RESULTS")
    print("-" * 50)
    ensemble_metrics = evaluate_model(
        y_test, y_pred, y_prob, model_name="Stacking Ensemble"
    )

    # 6. Save
    save_sklearn_model(meta_model, META_MODEL_FILE)
    plot_confusion_matrix(y_test, y_pred, "Stacking Ensemble", "models/ensemble_confusion_matrix.png")

    # Summary comparison
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for name, preds in [("RF", rf_pred), ("XGB", xgb_pred), ("DNN", dnn_pred), ("Ensemble", y_pred)]:
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        print(f"  {name:10s}  Accuracy: {acc:.4f}  F1-Macro: {f1:.4f}")
    print("=" * 60)

    print("\n[OK] Ensemble training complete!")


if __name__ == "__main__":
    main()
