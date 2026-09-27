"""Lesson 01: compute linear predictions for one observation and a batch.

Run from the repository root:
    python 01-foundations/01-vectors-and-predictions/example.py
"""

import numpy as np


def predict_one(features: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Return w · x + b for one observation with matching feature/weight shapes."""
    if features.ndim != 1 or weights.ndim != 1 or features.shape != weights.shape:
        raise ValueError("features and weights must be 1-D arrays of the same shape")
    return float(np.dot(weights, features) + bias)


def main() -> None:
    features = np.array([2.0, 3.0])
    weights = np.array([1.5, 0.5])
    bias = 2.0

    prediction = predict_one(features, weights, bias)
    assert np.isclose(prediction, 6.5)
    print(f"One observation: features={features}, shape={features.shape}")
    print(f"Weights: {weights}, shape={weights.shape}; bias={bias}")
    print(f"Prediction: {prediction:.1f}")

    batch = np.array([[2.0, 3.0], [4.0, 1.0]])
    if batch.ndim != 2 or batch.shape[1] != weights.shape[0]:
        raise ValueError("each batch row must have one value per weight")
    predictions = batch @ weights + bias
    assert np.allclose(predictions, np.array([6.5, 8.5]))
    print(f"Batch shape: {batch.shape}")
    print(f"Batch predictions: {predictions}, shape={predictions.shape}")


if __name__ == "__main__":
    main()
