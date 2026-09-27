# Lesson 01 — Feature Vectors and Linear Predictions

> **Stage 01 · Foundations** | Prerequisites: basic Python variables, lists, functions, and loops | Estimated scope: one short lesson, not a timed challenge

## Learning goals

By the end of this lesson, you should be able to:

1. Represent one observation as a feature vector and a set of model parameters as a weight vector.
2. Calculate a linear prediction by hand and with NumPy.
3. Explain what an array's shape means, and distinguish a single prediction from a batch of predictions.
4. State what model weights describe **and what they do not prove**.

## 1. From a Python list to a feature vector

Suppose we want a model to predict a **toy score** from two measurements:

- `x[0]`: hours of practice.
- `x[1]`: number of completed exercises.

We represent one observation as `x = [2.0, 3.0]`. It has **two features**. A Python list can store these numbers; a NumPy array also stores them in a form designed for numerical operations.

This example is deliberately invented. Its numbers are not observations from a real study, and the model has **not** been trained on data.

## 2. Make a prediction by hand

Choose weights `w = [1.5, 0.5]` and bias `b = 2.0`. The model computes

$$
\hat{y}=w^\top x+b=\sum_{j=1}^{2} w_jx_j+b.
$$

For our observation:

$$
\hat{y}=(1.5\times 2.0)+(0.5\times 3.0)+2.0=6.5.
$$

Here `x` holds **inputs**; `w` and `b` define the **model**; `ŷ` is its predicted score. The superscript `T` means transpose: in this expression, `wᵀx` is the dot product, obtained by multiplying matching entries and adding the products. No matrix-manipulation experience is required yet.

### NumPy version

Install the lesson's dependency from the repository root:

```bash
python -m pip install -r requirements/foundations.txt
python 01-foundations/01-vectors-and-predictions/example.py
```

If your computer uses `python3` rather than `python`, replace `python` in both commands. A project-specific virtual environment is recommended; see the [Python Packaging User Guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for setup instructions. Run commands from the **repository root** so the paths resolve correctly.

The example uses NumPy's dot product:

```python
prediction = np.dot(weights, features) + bias
```

For this example, the result is `6.5`. See the complete, executable [`example.py`](example.py) for shape checks and a second prediction.

## 3. Shapes and batches

In the script, `features.shape == (2,)` means one vector containing two numbers. For two observations, place one observation in each **row**:

```python
batch = np.array([[2.0, 3.0], [4.0, 1.0]])
```

Now `batch.shape == (2, 2)`: two observations, two features per observation. `batch @ weights + bias` produces two predictions, one per row. The second observation gives `(1.5 × 4.0) + (0.5 × 1.0) + 2.0 = 8.5`.

> **Shape rule:** If there are `n` observations and `d` features, the input matrix has shape `(n, d)` and the weights have shape `(d,)`; the predictions have shape `(n,)`.

## 4. What does a weight explain?

In this **fixed linear function**, if we increase `x[0]` by one unit while keeping `x[1]` and the parameters fixed, the predicted score increases by `1.5`. That is a statement about **this model's calculation**.

It does **not** show that one extra hour of practice causes a real person's score to increase by `1.5`. We invented the weights; even weights fitted to real observations would not by themselves establish a causal effect. Moreover, if two features tend to vary together in actual data, holding one fixed while changing the other may describe an unrealistic comparison.

This distinction—**a model-level description is not automatically a claim about the world**—will return throughout the repository.

## 5. Exercises

Try these **before** reading the answers.

1. **Hand calculation:** For `x = [4.0, 1.0]`, the same `w` and `b`, calculate the prediction without running Python.
2. **Edit and test:** Change the second weight from `0.5` to `1.0`. Predict the first observation's new output, then change `weights` in `example.py` and check your answer. The script's assertions refer to the original weights; update the expected values as part of the exercise, or restore the weights before rerunning the original script.
3. **Reasoning:** Does a positive weight for `hours of practice` prove that more practice increases real-world performance? Give one reason for your answer.
4. **Shapes:** What should the input and output shapes be for five observations with two features each?

<details>
<summary>Check your answers</summary>

1. `(1.5 × 4.0) + (0.5 × 1.0) + 2.0 = 8.5`.
2. `(1.5 × 2.0) + (1.0 × 3.0) + 2.0 = 8.0`. With the modified weight, the second observation would give `9.0`.
3. No. The number describes how this function changes when an input is changed with the others fixed. It is not evidence that intervening on practice hours causes the same real-world change; the model here was not even fitted to data.
4. Input `(5, 2)`, weights `(2,)`, output `(5,)`.

</details>

## Sources and next step

- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) — arrays and shapes.
- [NumPy `dot` reference](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) — vector dot products.
- [Christoph Molnar, *Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) — the broader distinction between models and explanations.

**Next:** How can a model obtain weights from data, and how do we check whether its predictions work on unseen examples? This question belongs to a future lesson; the current weights were chosen by hand.
