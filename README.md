# XAI: From Zero to Research

A structured, beginner-accessible path from basic Python to rigorous research in explainable artificial intelligence (XAI).

> **Project status: curriculum under development.** The roadmap below describes planned coverage, not completed lessons. A lesson will be linked when its explanations, exercises, and examples are ready. This is an independently developed learning project, not an accredited course or a claim that completing it makes someone a researcher.

## Who this is for

This repository is for readers who know basic Python syntax (variables, functions, loops, lists, and dictionaries) but have little or no machine-learning background. You do **not** need to know NumPy, PyTorch, or XAI before starting. Mathematical prerequisites are introduced as they become necessary.

The aim is to move beyond calling an explanation library: learn the mathematics, build and inspect models, implement small versions of methods, test their assumptions, reproduce results, and formulate questions that could support research.

**Start here:** Follow the stages in order. If no lesson is linked yet, the curriculum is still being written; do not mistake a planned topic for published material. Lesson numbers denote sequence, not a promise that each lesson takes one day.

## Learning roadmap

| Stage | Focus | What you should eventually be able to do | Status |
| --- | --- | --- | --- |
| 01 · Foundations | NumPy, pandas, linear algebra, probability and statistics, multivariable calculus, PyTorch basics | Represent data, follow mathematical derivations, compute gradients, and build reproducible small experiments | Planned |
| 02 · Machine learning | Interpretable and complex models; MLPs, CNNs, RNNs, Transformers | Train and evaluate models; distinguish predictions, model structure, and claims about the world | Planned |
| 03 · XAI methods | LIME, SHAP, Integrated Gradients, Grad-CAM, TCAV | Derive and implement simplified methods, compare them with libraries, and identify what each explanation actually measures | Planned |
| 04 · Research practice | Mechanistic interpretability, evaluation and formalization, causal XAI, paper reproduction | Reproduce a bounded result, run controls, investigate failure cases, and report limitations | Planned |

### Stage 01 — Mathematical and programming foundations

- **Scientific Python:** NumPy arrays, shapes, broadcasting; pandas tables; plotting and experimental hygiene.
- **Linear algebra:** vectors, matrix multiplication, projections, eigenvectors, and singular value decomposition (SVD).
- **Probability and statistics:** conditional probability, expectation, variance, sampling, and uncertainty.
- **Multivariable calculus:** partial derivatives, gradients, the chain rule, and gradient-based optimization.
- **PyTorch:** tensors, automatic differentiation, a small neural network, and access to intermediate activations and gradients.

**First planned lesson:** `01-foundations/01-vectors-and-predictions/` — represent features as vectors, compute a simple linear prediction by hand and in NumPy, and ask what model weights do *not* establish. Its path will become a link when the lesson is published.

### Stage 02 — Machine-learning models

- **Interpretable starting points:** linear regression, logistic regression, and small decision trees; study both what their parameters show and what they do not prove.
- **More complex models:** support-vector machines, random forests, and XGBoost; evaluate before attempting to explain them.
- **Deep learning:** multilayer perceptrons (MLPs), convolutional networks (CNNs), recurrent networks (RNNs), and Transformers, including self-attention.
- **Research habit:** establish data splits, metrics, and baseline models before comparing explanations. An attention weight or a model coefficient is not, by itself, a causal explanation.

### Stage 03 — Classical XAI methods

| Method | Central question | Planned exercise |
| --- | --- | --- |
| LIME | Can a simple model approximate a prediction locally? | Vary the perturbations and local neighborhood. |
| SHAP | How are contributions allocated under a specified value function and background/reference setup? | Compute exact Shapley values for a tiny feature set before using a library. |
| Integrated Gradients | How does an output change along a path from a baseline to an input? | Check the attribution sum and change the baseline. |
| Grad-CAM | Which regions of a convolutional model's feature maps influence a target output? | Compare heatmaps with controlled model changes. |
| TCAV | How sensitive is a model output to a user-defined concept in a representation space? | Construct concept and control examples, then test sensitivity. |

These methods answer different questions. Mathematical properties, attractive visualizations, and software outputs are not automatic evidence that an explanation is faithful or causal.

### Stage 04 — Research practice and frontiers

- **Mechanistic interpretability:** Transformers, superposition, sparse autoencoders (SAEs), feature hypotheses, circuits, and intervention-based tests. Do not assume each neuron or attention head stores one human-readable fact.
- **Evaluation and formalization:** randomization-based sanity checks, faithfulness, robustness, and explicit definitions of what an explanation is expected to explain.
- **Causal XAI:** distinguish changes to model inputs or internal activations from causal claims about the data-generating world; introduce causal inference and interventions with their assumptions.
- **Paper reproduction:** state the original claim, dataset, baseline, environment, metric, controls, deviations, negative results, and remaining uncertainty. Reproducing a result is a research exercise, not proof that an entire method works universally.

The research stage is an opportunity to practice research, **not** a certificate of research competence. Topics may evolve as the field and this project develop.

## How lessons will work

Each published lesson is intended to contain:

1. **Learning goals and prerequisites** — what you will be able to do and what you need first.
2. **Concept and mathematics** — intuition, definitions, worked examples, and derivations where appropriate.
3. **Implementation** — a small hand calculation or implementation, followed by a library comparison when useful.
4. **Checks and failure cases** — tests, counterexamples, assumptions, and limits of the result.
5. **Exercises and sources** — practice questions and links to original papers or documentation.

A paper-reproduction lesson additionally records what was reproduced and what was not. Code, data provenance, random seeds, dependency versions, and run instructions will be documented alongside experiments when those experiments are published.

## How to navigate

- Follow the stage order for a complete introduction; revisit earlier mathematics when a later method requires it.
- Look for a direct lesson link and a clear status before assuming material is available.
- When comparing methods, first ask: **Which model, which output, which data distribution or baseline, and which notion of explanation?**
- Use the original sources below as companions, not as substitutes for working through the examples.

## Core reading

- [Christoph Molnar, *Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) — broad introduction to interpretable models and explanation methods.
- [Ribeiro et al., *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier* (KDD 2016)](https://arxiv.org/abs/1602.04938) — LIME.
- [Lundberg and Lee, *A Unified Approach to Interpreting Model Predictions* (NeurIPS 2017)](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) — SHAP.
- [Sundararajan et al., *Axiomatic Attribution for Deep Networks* (ICML 2017)](https://arxiv.org/abs/1703.01365) — Integrated Gradients.
- [Kim et al., *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors* (ICML 2018)](https://arxiv.org/abs/1711.11279) — TCAV.
- [Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018)](https://papers.neurips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf) — evaluation by randomization tests.
- [Anthropic, *Towards Monosemanticity*](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) — dictionary learning and features in neural networks.
- [*Explainable AI needs formalization*](https://www.nature.com/articles/s44387-026-00095-1) — a perspective on defining explanation goals more precisely.

Reading-list inclusion does not imply that a paper has already been reproduced in this repository.

## Project principles

- **Teach before tooling:** explain the question and assumptions before importing a package.
- **Be precise about scope:** separate a mathematical guarantee, an empirical observation, a model-level intervention, and a real-world causal claim.
- **Publish verified work:** planned lessons stay labeled as planned; results are not described as reproduced without runnable evidence.
- **Make room for correction:** record counterexamples, limitations, and corrections openly.

## License and contributions

A project license and contribution guidelines will be added before accepting external contributions. Until a license is published, do not assume that the repository's original material has an open-source license. Linked papers, software, datasets, and figures retain their respective licenses.
