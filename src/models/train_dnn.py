"""
Deep Neural Network (DNN) Classifier Training
Trains a Multi-Layer Perceptron using scikit-learn's MLPClassifier.

Usage:
    python -m src.models.train_dnn
    python -m src.models.train_dnn --test-mode
"""

import argparse
import time
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.data.feature_config import DNN_MODEL_FILE, NUM_CLASSES
from src.models.model_utils import (
    load_processed_data, evaluate_model, save_sklearn_model,
    plot_confusion_matrix,
)


def train_dnn(X_train, y_train, test_mode=False):
    """Train the DNN (MLP) model: 128 -> 64 -> 32 -> softmax."""
    print("\n-- Training Deep Neural Network (MLPClassifier) --")

    if test_mode:
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=20,
            random_state=42,
            verbose=True,
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,             # L2 regularization
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,     # Like Keras EarlyStopping
            n_iter_no_change=10,     # Patience
            validation_fraction=0.1,
            random_state=42,
            verbose=True,
        )

    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train DNN (MLPClassifier)")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    print("=" * 60)
    print("DNN Classifier Training (scikit-learn MLPClassifier)")
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

    # Train
    start = time.time()
    model = train_dnn(X_train, y_train, test_mode=args.test_mode)
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Iterations: {model.n_iter_}")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = evaluate_model(y_test, y_pred, y_prob, model_name="DNN (MLP)")

    # Save
    save_sklearn_model(model, DNN_MODEL_FILE)
    plot_confusion_matrix(y_test, y_pred, "DNN (MLP)", "models/dnn_confusion_matrix.png")

    # Training loss curve
    if hasattr(model, "loss_curve_"):
        print(f"\n  Final training loss: {model.loss_curve_[-1]:.4f}")

    print("\n[OK] DNN training complete!")


if __name__ == "__main__":
    main()
