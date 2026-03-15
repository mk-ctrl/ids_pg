"""
Baseline vs. Enhanced Comparison
Compares pure Wazuh (baseline) against ML-ensemble-enhanced detection.

Usage:
    python -m src.evaluation.compare
    python -m src.evaluation.compare --data-dir data/processed
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.feature_config import (
    RF_MODEL_FILE, XGB_MODEL_FILE, DNN_MODEL_FILE, META_MODEL_FILE,
    NUM_CLASSES, ID_TO_CLASS,
)
from src.models.model_utils import load_processed_data, load_sklearn_model
from src.evaluation.metrics import compute_all_metrics, print_metrics_table


def simulate_baseline(y_true):
    """
    Simulate Wazuh-only (rule-based) detection baseline.
    
    Assumptions for simulation:
    - High detection for known patterns (DoS, Probe): ~80%
    - Lower detection for subtle attacks (R2L, U2R): ~40-50%
    - Some false positives on normal traffic: ~10%
    """
    np.random.seed(42)
    y_baseline = np.zeros_like(y_true)

    for i, label in enumerate(y_true):
        if label == 0:  # Normal
            # 10% false positive rate
            if np.random.random() < 0.10:
                y_baseline[i] = np.random.choice([1, 2, 3, 4])
            else:
                y_baseline[i] = 0
        elif label == 1:  # DoS - well-known patterns
            y_baseline[i] = 1 if np.random.random() < 0.80 else 0
        elif label == 2:  # Probe - partially detectable
            y_baseline[i] = 2 if np.random.random() < 0.75 else 0
        elif label == 3:  # R2L - harder to detect
            y_baseline[i] = 3 if np.random.random() < 0.45 else 0
        elif label == 4:  # U2R - hardest to detect
            y_baseline[i] = 4 if np.random.random() < 0.40 else 0

    return y_baseline


def run_ensemble_prediction(X_test):
    """Run the full ensemble on test data."""
    rf_model = load_sklearn_model(RF_MODEL_FILE)
    xgb_model = load_sklearn_model(XGB_MODEL_FILE)
    dnn_model = load_sklearn_model(DNN_MODEL_FILE)
    meta_model = load_sklearn_model(META_MODEL_FILE)

    # Generate meta-features
    rf_probs = rf_model.predict_proba(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)
    dnn_probs = dnn_model.predict_proba(X_test)

    def pad(probs, n):
        if probs.shape[1] < n:
            p = np.zeros((probs.shape[0], n))
            p[:, :probs.shape[1]] = probs
            return p
        return probs

    meta_features = np.hstack([
        pad(rf_probs, NUM_CLASSES),
        pad(xgb_probs, NUM_CLASSES),
        pad(dnn_probs, NUM_CLASSES),
    ])

    y_pred = meta_model.predict(meta_features)
    y_prob = meta_model.predict_proba(meta_features)
    return y_pred, y_prob


def plot_comparison(baseline_metrics, ensemble_metrics, save_path="models/comparison_chart.png"):
    """Generate side-by-side comparison bar chart."""
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "detection_rate"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Detection Rate"]

    baseline_vals = [baseline_metrics.get(m, 0) for m in metric_names]
    ensemble_vals = [ensemble_metrics.get(m, 0) for m in metric_names]

    x = np.arange(len(display_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Wazuh Only (Baseline)",
                   color="#e74c3c", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, ensemble_vals, width, label="+ ML Ensemble (Enhanced)",
                   color="#2ecc71", alpha=0.85, edgecolor="white")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Baseline vs. ML-Enhanced IDS Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Comparison chart saved: {save_path}")
    plt.close()


def plot_fpr_comparison(baseline_metrics, ensemble_metrics, save_path="models/fpr_comparison.png"):
    """Plot False Positive Rate comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))

    categories = ["False Positive Rate", "False Negative Rate"]
    baseline_vals = [baseline_metrics["fpr"], baseline_metrics["fnr"]]
    ensemble_vals = [ensemble_metrics["fpr"], ensemble_metrics["fnr"]]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, baseline_vals, width, label="Baseline", color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, ensemble_vals, width, label="Enhanced", color="#2ecc71", alpha=0.85)

    for i, (bv, ev) in enumerate(zip(baseline_vals, ensemble_vals)):
        ax.text(i - width/2, bv + 0.005, f"{bv:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, ev + 0.005, f"{ev:.3f}", ha="center", fontsize=9)

    ax.set_ylabel("Rate (lower is better)")
    ax.set_title("Error Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] FPR comparison chart saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Enhanced Comparison")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    print("=" * 60)
    print("Baseline vs. ML-Enhanced IDS Comparison")
    print("=" * 60)

    # Load data
    data = load_processed_data(args.data_dir)
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"  Test set: {X_test.shape[0]:,} samples")

    # 1. Baseline (simulated Wazuh-only)
    print("\n-- Baseline (Wazuh-Only Simulation) --")
    y_baseline = simulate_baseline(y_test)
    baseline_metrics = compute_all_metrics(y_test, y_baseline, model_name="Wazuh Baseline")
    print_metrics_table(baseline_metrics)

    # 2. ML-Enhanced ensemble
    print("\n-- ML-Enhanced (Stacking Ensemble) --")
    y_ensemble, y_ensemble_prob = run_ensemble_prediction(X_test)
    ensemble_metrics = compute_all_metrics(y_test, y_ensemble, y_ensemble_prob, model_name="ML Ensemble")
    print_metrics_table(ensemble_metrics)

    # 3. Improvement summary
    print("\n" + "=" * 60)
    print("  IMPROVEMENT SUMMARY")
    print("=" * 60)
    for metric_name in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "detection_rate"]:
        baseline_val = baseline_metrics[metric_name]
        ensemble_val = ensemble_metrics[metric_name]
        improvement = (ensemble_val - baseline_val) * 100
        arrow = "^" if improvement > 0 else "v"
        print(f"  {metric_name:<25s}: {baseline_val:.4f} -> {ensemble_val:.4f} ({arrow} {abs(improvement):.1f}%)")

    fpr_reduction = (baseline_metrics["fpr"] - ensemble_metrics["fpr"]) * 100
    print(f"  {'FPR reduction':<25s}: {baseline_metrics['fpr']:.4f} -> {ensemble_metrics['fpr']:.4f} (v {fpr_reduction:.1f}%)")
    print("=" * 60)

    # 4. Charts
    plot_comparison(baseline_metrics, ensemble_metrics)
    plot_fpr_comparison(baseline_metrics, ensemble_metrics)

    print("\n[OK] Comparison complete!")


if __name__ == "__main__":
    main()
