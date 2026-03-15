"""
Ensemble Inference Engine
Loads all trained models and runs the stacking ensemble on feature vectors.

Usage:
    python -m src.inference.ensemble_engine --demo
"""

import os
import logging
import argparse
import numpy as np
from typing import List, Dict, Tuple

from src.data.feature_config import (
    RF_MODEL_FILE, XGB_MODEL_FILE, DNN_MODEL_FILE, META_MODEL_FILE,
    NUM_CLASSES, ID_TO_CLASS, ENSEMBLE_THRESHOLD,
)
from src.models.model_utils import load_sklearn_model

logger = logging.getLogger(__name__)


class EnsembleEngine:
    """Hybrid ensemble inference engine combining RF, XGBoost, DNN, and meta-classifier."""

    def __init__(self):
        """Load all trained models."""
        self.rf_model = None
        self.xgb_model = None
        self.dnn_model = None
        self.meta_model = None
        self._loaded = False

    def load_models(self) -> bool:
        """Load all models. Returns True if all loaded successfully."""
        try:
            self.rf_model = load_sklearn_model(RF_MODEL_FILE)
            logger.info("[OK] Random Forest loaded")
        except FileNotFoundError:
            logger.error(f"[FAIL] Random Forest not found ({RF_MODEL_FILE})")
            return False

        try:
            self.xgb_model = load_sklearn_model(XGB_MODEL_FILE)
            logger.info("[OK] XGBoost loaded")
        except FileNotFoundError:
            logger.error(f"[FAIL] XGBoost not found ({XGB_MODEL_FILE})")
            return False

        try:
            self.dnn_model = load_sklearn_model(DNN_MODEL_FILE)
            logger.info("[OK] DNN loaded")
        except FileNotFoundError:
            logger.error(f"[FAIL] DNN not found ({DNN_MODEL_FILE})")
            return False

        try:
            self.meta_model = load_sklearn_model(META_MODEL_FILE)
            logger.info("[OK] Meta-classifier loaded")
        except FileNotFoundError:
            logger.error(f"[FAIL] Meta-classifier not found ({META_MODEL_FILE})")
            return False

        self._loaded = True
        logger.info("All models loaded successfully")
        return True

    def predict(self, X: np.ndarray) -> List[Dict]:
        """
        Run ensemble inference on feature matrix.
        
        Returns list of prediction dicts:
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "is_intrusion": bool,
            "base_predictions": {"rf": {...}, "xgb": {...}, "dnn": {...}},
            "meta_probabilities": [float, ...]
        }
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Base model predictions (probabilities)
        rf_probs = self._pad_probs(self.rf_model.predict_proba(X))
        xgb_probs = self._pad_probs(self.xgb_model.predict_proba(X))
        dnn_probs = self._pad_probs(self.dnn_model.predict_proba(X))

        # Stack meta-features
        meta_features = np.hstack([rf_probs, xgb_probs, dnn_probs])

        # Meta-classifier prediction
        meta_pred = self.meta_model.predict(meta_features)
        meta_prob = self.meta_model.predict_proba(meta_features)

        results = []
        for i in range(len(X)):
            confidence = float(np.max(meta_prob[i]))
            class_id = int(meta_pred[i])
            class_name = ID_TO_CLASS.get(class_id, f"Unknown_{class_id}")

            result = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "is_intrusion": class_id > 0 and confidence >= ENSEMBLE_THRESHOLD,
                "base_predictions": {
                    "rf": {
                        "class": int(np.argmax(rf_probs[i])),
                        "confidence": float(np.max(rf_probs[i])),
                        "probabilities": rf_probs[i].tolist(),
                    },
                    "xgb": {
                        "class": int(np.argmax(xgb_probs[i])),
                        "confidence": float(np.max(xgb_probs[i])),
                        "probabilities": xgb_probs[i].tolist(),
                    },
                    "dnn": {
                        "class": int(np.argmax(dnn_probs[i])),
                        "confidence": float(np.max(dnn_probs[i])),
                        "probabilities": dnn_probs[i].tolist(),
                    },
                },
                "meta_probabilities": meta_prob[i].tolist(),
            }
            results.append(result)

        return results

    def _pad_probs(self, probs: np.ndarray) -> np.ndarray:
        """Pad probability matrix to NUM_CLASSES columns."""
        if probs.shape[1] < NUM_CLASSES:
            padded = np.zeros((probs.shape[0], NUM_CLASSES))
            padded[:, :probs.shape[1]] = probs
            return padded
        return probs


def demo():
    """Run a demo with synthetic data."""
    print("=" * 60)
    print("Ensemble Inference Engine - Demo Mode")
    print("=" * 60)

    engine = EnsembleEngine()
    if not engine.load_models():
        print("\n[WARN] Models not found. Train models first:")
        print("  python -m src.models.train_rf")
        print("  python -m src.models.train_xgb")
        print("  python -m src.models.train_dnn")
        print("  python -m src.models.train_ensemble")
        return

    # Generate random feature vectors for demo
    n_features = engine.rf_model.n_features_in_
    print(f"\nGenerating {5} synthetic feature vectors ({n_features} features)...")

    np.random.seed(42)
    X_demo = np.random.randn(5, n_features)

    results = engine.predict(X_demo)

    print(f"\n{'-' * 60}")
    for i, result in enumerate(results):
        icon = "🔴" if result["is_intrusion"] else "🟢"
        print(f"  Sample {i+1}: {icon} {result['class_name']} "
              f"(confidence: {result['confidence']:.3f})")
        print(f"           Base: RF={result['base_predictions']['rf']['class']}, "
              f"XGB={result['base_predictions']['xgb']['class']}, "
              f"DNN={result['base_predictions']['dnn']['class']}")
    print(f"{'-' * 60}")
    print("\n[OK] Demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Inference Engine")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    args = parser.parse_args()

    if args.demo:
        demo()
