"""
Model Utilities
Shared functions for loading/saving models, evaluation, and visualization.
"""

import os
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from src.data.feature_config import (
    MODEL_DIR, ID_TO_CLASS, NUM_CLASSES,
    RF_MODEL_FILE, XGB_MODEL_FILE, DNN_MODEL_FILE, META_MODEL_FILE,
)


def ensure_model_dir():
    """Create model directory if it doesn't exist."""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)


def save_sklearn_model(model, filename: str):
    """Save a scikit-learn/XGBoost model with joblib."""
    ensure_model_dir()
    path = Path(MODEL_DIR) / filename
    joblib.dump(model, path)
    print(f"  [OK] Model saved: {path}")


def load_sklearn_model(filename: str):
    """Load a scikit-learn/XGBoost model."""
    path = Path(MODEL_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def load_processed_data(data_dir: str = "data/processed"):
    """Load preprocessed train/test numpy arrays."""
    data_dir = Path(data_dir)
    return {
        "X_train": np.load(data_dir / "X_train.npy"),
        "X_test": np.load(data_dir / "X_test.npy"),
        "y_train": np.load(data_dir / "y_train.npy"),
        "y_test": np.load(data_dir / "y_test.npy"),
    }


def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Compute and print evaluation metrics.
    Returns dict of metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # ROC-AUC (requires probability scores)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if y_prob.ndim == 1:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            metrics["roc_auc"] = None

    # False Positive Rate (for binary: intrusion vs normal)
    y_binary_true = (y_true > 0).astype(int)
    y_binary_pred = (y_pred > 0).astype(int)
    tn = np.sum((y_binary_true == 0) & (y_binary_pred == 0))
    fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Print results
    print(f"\n{'=' * 50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
    if metrics.get("roc_auc"):
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  False Positive Rate: {metrics['fpr']:.4f}")
    print(f"{'=' * 50}")

    # Class-level report
    target_names = [ID_TO_CLASS.get(i, f"Class_{i}") for i in sorted(np.unique(y_true))]
    print("\n" + classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot and optionally save a confusion matrix."""
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [ID_TO_CLASS.get(i, f"Class_{i}") for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=axes[0])
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=target_names, yticklabels=target_names, ax=axes[1])
    axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Confusion matrix saved: {save_path}")
    plt.close()

    return cm
