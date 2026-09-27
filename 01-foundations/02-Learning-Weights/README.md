# Lesson 02 — Learning Weights from Data

> **Stage 01 · Foundations** | Prerequisites: [Lesson 01](../01-vectors-and-predictions/README.md), Python loops and functions, basic NumPy arrays

In Lesson 01, we chose a model's weights by hand. Here, we ask a different question: **how can a model choose a weight and bias from examples?** We will fit the simplest possible linear model without scikit-learn or PyTorch.

## Learning goals

By the end of this lesson, you should be able to:

1. Distinguish features, targets, predictions, residuals, and parameters.
2. Calculate mean squared error (MSE) for a tiny dataset.
3. Follow the derivatives that update a weight and a bias.
4. Compare a model's behavior on training examples and a held-out example.
5. Explain why low prediction error does not prove that a fitted weight is a causal effect.

## 1. A tiny, invented dataset

Suppose `x` is the number of completed practice sets and `y` is a toy score. All numbers below are invented for teaching; they do not describe real students.

| `x` (practice sets) | `y` (toy score) | Use |
| ---: | ---: | --- |
| 0 | 1 | Train |
| 1 | 3 | Train |
| 2 | 5 | Train |
| 3 | 7 | Train |
| 4 | 9 | Held-out check |

The four training rows follow the rule $y=2x+1$; we do **not** give the learning algorithm that rule. The fifth row is reserved for a check after training. Because this dataset is perfectly regular, the check is educational, **not** evidence of generalization to real data.

We use one input feature, so our model is

$$
\hat y_i = wx_i + b.
$$

`w` is the slope (weight), and `b` is the intercept (bias). At first, we set both to zero. With `w = 0` and `b = 0`, every prediction is zero: a deliberately poor starting point.

## 2. Measure the error

For each row, the **residual** is $\hat y_i-y_i$. Square the residuals and average over the $n$ training rows:

$$
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}(wx_i+b-y_i)^2.
$$

This is the *mean squared error* (MSE). With the initial parameters, the training targets are `1, 3, 5, 7`, so

$$
L(0,0)=\frac{1^2+3^2+5^2+7^2}{4}=21.
$$

**Check by hand:** If we try `w = 1` and `b = 0`, the predictions are `0, 1, 2, 3`. The squared errors are `1, 4, 9, 16`, giving an MSE of `7.5`. The candidate is better than the initial one, but not perfect.

MSE is a choice of training objective; choosing a different objective can produce different fitted parameters.

## 3. Change the parameters deliberately

A *gradient* tells us how the loss changes for small changes in each parameter. Applying the chain rule to the squared residual gives

$$
\frac{\partial L}{\partial w}=\frac{2}{n}\sum_{i=1}^{n}x_i(wx_i+b-y_i),
\qquad
\frac{\partial L}{\partial b}=\frac{2}{n}\sum_{i=1}^{n}(wx_i+b-y_i).
$$

To see where the first expression comes from, take one term $(wx_i+b-y_i)^2$. Its derivative with respect to `w` is $2(wx_i+b-y_i)x_i$; then average the four terms. Differentiating with respect to `b` replaces $x_i$ by `1`.

At `w = 0, b = 0`, the residuals are `-1, -3, -5, -7`. Therefore:

$$
\frac{\partial L}{\partial w}
=\frac{2}{4}[0(-1)+1(-3)+2(-5)+3(-7)]=-17,
\qquad
\frac{\partial L}{\partial b}
=\frac{2}{4}(-1-3-5-7)=-8.
$$

Choose a **learning rate** $\eta=0.05$ and update both parameters using the gradients from the *same* current model:

$$
w_{\text{new}}=w-\eta\frac{\partial L}{\partial w}=0.85,
\qquad
b_{\text{new}}=b-\eta\frac{\partial L}{\partial b}=0.40.
$$

These are not the final values. Repeat the calculation until the loss becomes small. The numerical experiment in [`example.py`](example.py) uses NumPy for the sums but writes the gradient formulas explicitly; it does not call a fitting library.

## 4. Run the example

From the **repository root**, use the same NumPy dependency as Lesson 01:

```bash
python -m pip install -r requirements/foundations.txt
python 01-foundations/02-learning-weights/example.py
```

If your system uses `python3`, replace `python` in both commands. A virtual environment is recommended; see the [Python Packaging User Guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). The script checks the hand-calculated initial MSE and gradients, then prints a few training checkpoints, the fitted parameters, and predictions on both training and held-out inputs. No additional package is needed beyond the existing `requirements/foundations.txt`.

You should see the starting MSE of `21.000000`, an initial gradient of `dw=-17.0, db=-8.0`, and a fitted model with parameters close to `w=2, b=1`. The held-out input `x=4` should receive a prediction close to `9`. Floating-point results and the final displayed digits may vary; the assertions in the script check sensible tolerances rather than exact equality.

> **Training versus checking:** The fifth target is never passed into the optimizer. Inspecting one held-out row is a useful first habit, but it is not a statistically reliable estimate of performance on an unknown population. We will study proper data splits and evaluation later.

## 5. The XAI question

Once fitted, `w ≈ 2` describes this model's output: increasing `x` by one, with its parameters fixed, increases its predicted score by about two. On our synthetic data, this matches the rule we used to construct the scores.

But the program did not discover that real-world practice *causes* scores to rise by two. The data are invented, the relationship was built in, and even real observational data could contain confounding or correlated features. An interpretable equation makes the model's arithmetic visible; it does not automatically make its interpretation about the world correct.

## 6. Exercises

Try these before opening the answers.

1. **Hand calculation:** Confirm the initial MSE, both initial gradients, and the first parameter update shown above.
2. **Objective:** What is the MSE for `w=2, b=1` on the four training rows? Why is this the lowest possible MSE?
3. **Experiment:** In `example.py`, change the learning rate from `0.05` to `0.01`. After the same number of steps, did training progress faster or slower? Restore the original value when finished.
4. **Validation trap:** What goes wrong if you use `x=4, y=9` to choose hyperparameters repeatedly and then report that same row as an independent, untouched test?
5. **XAI:** Does a learned positive weight show a model-level response, a real-world intervention effect, or both?

<details>
<summary>Check your answers</summary>

1. `MSE=21`, `dw=-17`, `db=-8`; with learning rate `0.05`, the first update gives `w=0.85`, `b=0.40`.
2. `0`: each prediction equals its training target, and an average of squared real-valued errors cannot be negative.
3. Usually slower for this example over the same fixed number of steps; check the output rather than assuming this generalizes to all learning rates and models.
4. You have used the row to guide decisions, so it is no longer an untouched test. A fresh test set would be needed for an independent final evaluation.
5. It shows the model-level response for the specified equation. It does not, without additional causal assumptions and evidence, establish a real-world intervention effect.

</details>

## Sources and next step

- [NumPy: `mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) — the averaging operation used in the loss and gradients.
- [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) — a later reference for linear regression and its limitations; no scikit-learn code is needed here.
- [Christoph Molnar, *Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) — a broader discussion of model interpretation and its limits.

**Next:** This lesson used one feature and perfectly regular synthetic data. Later lessons will introduce multiple features, noisy data, appropriate evaluation, and parameter instability before moving to post-hoc explanations.
