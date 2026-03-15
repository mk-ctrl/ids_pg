# IDS ML Pipeline — Complete Run Results

## Pipeline Summary

All 6 steps completed successfully in **162.3 minutes** (exit code 0).

| Step | Time | Status |
|------|------|--------|
| Preprocess CIC-IDS2017 | 326s | ✅ |
| Train Random Forest | 1,776s | ✅ |
| Train XGBoost | 806s | ✅ |
| Train DNN (MLP) | 6,428s | ✅ |
| Train Stacking Ensemble | 340s | ✅ |
| Baseline vs Enhanced | 62s | ✅ |

## Dataset

- **Source**: 9 CIC-IDS2017 CSV files + 1 synthetic CSV
- **Total rows loaded**: 2,880,743
- **After cleaning**: 2,520,798 (removed 359,945 rows with NaN/inf/duplicates)
- **Train/Test split**: 2,014,924 / 503,731 (80/20, stratified)
- **Features**: 77, **Classes**: 5 (Normal, DoS, Probe, R2L, U2R)

> [!NOTE]
> 2,143 Web Attack rows (0.08%) were dropped due to an encoding mismatch (â€" vs –) in the label column. This is negligible.

## Model Results

| Model | Accuracy | F1 (Macro) | ROC-AUC | FPR |
|-------|----------|------------|---------|-----|
| Random Forest | 99.56% | 0.8433 | 0.9999 | 0.0051 |
| XGBoost | 99.90% | 0.9445 | 1.0000 | 0.0012 |
| DNN (MLP) | 99.78% | 0.7377 | 0.9672 | 0.0011 |
| **Stacking Ensemble** | **99.90%** | **0.9447** | **1.0000** | **0.0011** |

## Baseline vs. ML-Enhanced Comparison

| Metric | Wazuh Baseline | ML Ensemble | Improvement |
|--------|---------------|-------------|-------------|
| Accuracy | 88.07% | 99.90% | +11.8% |
| Precision (macro) | 0.4767 | 0.9056 | +42.9% |
| Recall (macro) | 0.6279 | 0.9986 | +37.1% |
| F1 (macro) | 0.4853 | 0.9447 | +45.9% |
| Detection Rate | 78.72% | 99.86% | +21.1% |
| False Positive Rate | 10.04% | 0.11% | **−9.9%** |

![Baseline vs Enhanced Comparison](C:/Users/MP/.gemini/antigravity/brain/8b241171-ec26-4c9b-a30c-9c036b5f2f3e/comparison_chart.png)

![Ensemble Confusion Matrix](C:/Users/MP/.gemini/antigravity/brain/8b241171-ec26-4c9b-a30c-9c036b5f2f3e/ensemble_confusion_matrix.png)

![FPR Comparison](C:/Users/MP/.gemini/antigravity/brain/8b241171-ec26-4c9b-a30c-9c036b5f2f3e/fpr_comparison.png)

## Output Artifacts

All saved in [models/](file:///c:/Users/MP/Downloads/ramya-pg/models):

| File | Description |
|------|-------------|
| [random_forest.joblib](file:///c:/Users/MP/Downloads/ramya-pg/models/random_forest.joblib) (43MB) | Trained RF model |
| [xgboost.joblib](file:///c:/Users/MP/Downloads/ramya-pg/models/xgboost.joblib) (2MB) | Trained XGBoost model |
| [dnn_model.joblib](file:///c:/Users/MP/Downloads/ramya-pg/models/dnn_model.joblib) (501KB) | Trained MLP model |
| [meta_classifier.joblib](file:///c:/Users/MP/Downloads/ramya-pg/models/meta_classifier.joblib) (2KB) | Stacking Logistic Regression |
| [scaler.joblib](file:///c:/Users/MP/Downloads/ramya-pg/models/scaler.joblib) (2KB) | StandardScaler for inference |
| [rf_confusion_matrix.png](file:///c:/Users/MP/Downloads/ramya-pg/models/rf_confusion_matrix.png) | RF confusion matrix |
| [xgb_confusion_matrix.png](file:///c:/Users/MP/Downloads/ramya-pg/models/xgb_confusion_matrix.png) | XGBoost confusion matrix |
| [dnn_confusion_matrix.png](file:///c:/Users/MP/Downloads/ramya-pg/models/dnn_confusion_matrix.png) | DNN confusion matrix |
| [ensemble_confusion_matrix.png](file:///c:/Users/MP/Downloads/ramya-pg/models/ensemble_confusion_matrix.png) | Ensemble confusion matrix |
| [comparison_chart.png](file:///c:/Users/MP/Downloads/ramya-pg/models/comparison_chart.png) | Baseline vs Enhanced bar chart |
| [fpr_comparison.png](file:///c:/Users/MP/Downloads/ramya-pg/models/fpr_comparison.png) | FPR/FNR comparison chart |

## Code Changes Made

- [feature_config.py](file:///c:/Users/MP/Downloads/ramya-pg/src/data/feature_config.py): Added extra encoding variants for Web Attack labels
- [preprocess.py](file:///c:/Users/MP/Downloads/ramya-pg/src/data/preprocess.py): Added dash normalization in label cleanup
- [run_pipeline.py](file:///c:/Users/MP/Downloads/ramya-pg/run_pipeline.py): **New** — single-command pipeline runner
