"""
Full IDS ML Pipeline Runner
Runs all steps sequentially: Preprocess → Train RF → Train XGB → Train DNN → Train Ensemble → Compare
"""

import subprocess
import sys
import time

PYTHON = sys.executable

STEPS = [
    ("Step 1/6: Preprocessing CIC-IDS2017 Data", [PYTHON, "-m", "src.data.preprocess"]),
    ("Step 2/6: Training Random Forest",          [PYTHON, "-m", "src.models.train_rf"]),
    ("Step 3/6: Training XGBoost",                [PYTHON, "-m", "src.models.train_xgb"]),
    ("Step 4/6: Training DNN (MLPClassifier)",    [PYTHON, "-m", "src.models.train_dnn"]),
    ("Step 5/6: Training Stacking Ensemble",      [PYTHON, "-m", "src.models.train_ensemble"]),
    ("Step 6/6: Baseline vs Enhanced Comparison", [PYTHON, "-m", "src.evaluation.compare"]),
]


def main():
    print("=" * 70)
    print("  HYBRID ENSEMBLE IDS - FULL PIPELINE RUNNER")
    print("=" * 70)
    
    total_start = time.time()
    results = []

    for i, (name, cmd) in enumerate(STEPS):
        print(f"\n{'#' * 70}")
        print(f"# {name}")
        print(f"{'#' * 70}\n")
        
        step_start = time.time()
        result = subprocess.run(cmd, cwd=".")
        step_time = time.time() - step_start
        
        status = "OK" if result.returncode == 0 else "FAILED"
        results.append((name, status, step_time))
        
        print(f"\n>>> {name}: {status} ({step_time:.1f}s)")
        
        if result.returncode != 0:
            print(f"\n[ERROR] {name} failed with exit code {result.returncode}")
            print("Stopping pipeline.")
            break

    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for name, status, elapsed in results:
        print(f"  [{status:6s}] {name} ({elapsed:.1f}s)")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print("\nAll outputs saved in: models/")
    print("  - random_forest.joblib, xgboost.joblib, dnn_model.joblib")
    print("  - meta_classifier.joblib, scaler.joblib")
    print("  - *_confusion_matrix.png, comparison_chart.png")


if __name__ == "__main__":
    main()
