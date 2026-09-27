"""Lesson 02: learn a slope and intercept with gradient descent.

Run from the repository root:
    python 01-foundations/02-learning-weights/example.py
"""

import numpy as np


def mse_and_gradients(
    x: np.ndarray, y: np.ndarray, weight: float, bias: float
) -> tuple[float, float, float]:
    """Return MSE and its derivatives with respect to weight and bias."""
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("x and y must be nonempty 1-D arrays of equal shape")
    residuals = weight * x + bias - y
    loss = float(np.mean(residuals**2))
    grad_weight = float(2.0 * np.mean(x * residuals))
    grad_bias = float(2.0 * np.mean(residuals))
    return loss, grad_weight, grad_bias


def main() -> None:
    x_train = np.array([0.0, 1.0, 2.0, 3.0])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    x_held_out = np.array([4.0])
    y_held_out = np.array([9.0])

    weight, bias = 0.0, 0.0
    learning_rate = 0.05
    steps = 2000

    initial_loss, initial_dw, initial_db = mse_and_gradients(
        x_train, y_train, weight, bias
    )
    assert np.isclose(initial_loss, 21.0)
    assert np.isclose(initial_dw, -17.0)
    assert np.isclose(initial_db, -8.0)
    print(f"Initial: MSE={initial_loss:.6f}, dw={initial_dw:.1f}, db={initial_db:.1f}")

    for step in range(1, steps + 1):
        _, grad_weight, grad_bias = mse_and_gradients(
            x_train, y_train, weight, bias
        )
        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias

        if step in (1, 10, 100, 1000, steps):
            loss, _, _ = mse_and_gradients(x_train, y_train, weight, bias)
            print(f"Step {step:4d}: w={weight:.6f}, b={bias:.6f}, MSE={loss:.6f}")

    assert np.isclose(weight, 2.0, atol=1e-3)
    assert np.isclose(bias, 1.0, atol=1e-3)

    train_predictions = weight * x_train + bias
    held_out_predictions = weight * x_held_out + bias
    print(f"Training predictions: {np.round(train_predictions, 4)}")
    print(f"Held-out input: {x_held_out}, prediction: {held_out_predictions[0]:.4f}")
    print(f"Held-out target (not used in training): {y_held_out[0]:.1f}")
    assert np.isclose(held_out_predictions[0], y_held_out[0], atol=1e-3)


if __name__ == "__main__":
    main()
