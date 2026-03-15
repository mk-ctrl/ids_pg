"""
Evaluation Metrics Module
Computes comprehensive metrics for IDS model evaluation.

Usage:
    python -m src.evaluation.metrics --demo
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize

from src.data.feature_config import ID_TO_CLASS, NUM_CLASSES


def compute_all_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Compute comprehensive evaluation metrics.
    
    Returns:
        dict with all metrics
    """
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class metrics
    unique_classes = sorted(np.unique(y_true))
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics["per_class"] = {}
    for i, cls_id in enumerate(unique_classes):
        cls_name = ID_TO_CLASS.get(cls_id, f"Class_{cls_id}")
        metrics["per_class"][cls_name] = {
            "precision": per_class_precision[i],
            "recall": per_class_recall[i],
            "f1": per_class_f1[i],
        }

    # ROC-AUC
    if y_prob is not None:
        try:
            if y_prob.ndim == 1:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # False Positive Rate (binary: intrusion vs normal)
    y_binary_true = (y_true > 0).astype(int)
    y_binary_pred = (y_pred > 0).astype(int)
    tn = np.sum((y_binary_true == 0) & (y_binary_pred == 0))
    fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))
    fn = np.sum((y_binary_true == 1) & (y_binary_pred == 0))
    tp = np.sum((y_binary_true == 1) & (y_binary_pred == 1))
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    metrics["detection_rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return metrics


def print_metrics_table(metrics: dict):
    """Print a formatted metrics table."""
    name = metrics["model_name"]
    print(f"\n{'=' * 55}")
    print(f"  {name} - Evaluation Results")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'-' * 42}")
    print(f"  {'Accuracy':<30s} {metrics['accuracy']:>10.4f}")
    print(f"  {'Precision (macro)':<30s} {metrics['precision_macro']:>10.4f}")
    print(f"  {'Recall (macro)':<30s} {metrics['recall_macro']:>10.4f}")
    print(f"  {'F1-Score (macro)':<30s} {metrics['f1_macro']:>10.4f}")
    if metrics.get("roc_auc"):
        print(f"  {'ROC-AUC':<30s} {metrics['roc_auc']:>10.4f}")
    print(f"  {'False Positive Rate':<30s} {metrics['fpr']:>10.4f}")
    print(f"  {'Detection Rate':<30s} {metrics['detection_rate']:>10.4f}")
    print(f"{'=' * 55}")

    # Per-class breakdown
    if "per_class" in metrics:
        print(f"\n  {'Class':<12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
        print(f"  {'-' * 44}")
        for cls_name, cls_metrics in metrics["per_class"].items():
            print(f"  {cls_name:<12s} {cls_metrics['precision']:>10.4f} "
                  f"{cls_metrics['recall']:>10.4f} {cls_metrics['f1']:>10.4f}")


def plot_roc_curves(y_true, y_prob, model_name="Model", save_path=None):
    """Plot multi-class ROC curves."""
    classes = sorted(np.unique(y_true))
    n_classes = len(classes)

    # Binarize labels
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (cls_id, color) in enumerate(zip(classes, colors)):
        cls_name = ID_TO_CLASS.get(cls_id, f"Class_{cls_id}")
        if y_prob.ndim == 2 and i < y_prob.shape[1]:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{cls_name} (AUC={roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} - ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] ROC curves saved: {save_path}")
    plt.close()


def demo():
    """Demo with synthetic predictions."""
    print("=" * 60)
    print("Evaluation Metrics - Demo Mode")
    print("=" * 60)

    np.random.seed(42)
    n = 1000

    # Simulated true labels (5 classes)
    y_true = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])

    # Simulated predictions (mostly correct)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
    y_pred[noise_idx] = np.random.randint(0, 5, size=len(noise_idx))

    # Simulated probabilities
    y_prob = np.zeros((n, 5))
    for i in range(n):
        y_prob[i, y_pred[i]] = 0.7 + np.random.random() * 0.3
        remaining = 1.0 - y_prob[i, y_pred[i]]
        others = [j for j in range(5) if j != y_pred[i]]
        for j in others:
            y_prob[i, j] = remaining / 4

    metrics = compute_all_metrics(y_true, y_pred, y_prob, model_name="Demo Model")
    print_metrics_table(metrics)
    print("\n[OK] Demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Metrics")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        demo()
