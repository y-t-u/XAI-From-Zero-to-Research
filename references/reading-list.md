# Reading List

A curated reading path for **XAI: From Zero to Research**. This is a syllabus, **not** a claim that the project author has read, endorsed every result in, or reproduced every item. Read alongside the corresponding lessons as they are published; do not try to finish this list before writing code.

**Starting point:** basic Python syntax; no prior machine-learning knowledge assumed.

## How to use this list

- **Core**: read the relevant parts while studying that stage. **Optional**: use when a specific question or project calls for it.
- Start with tutorials and small worked examples. Read original papers after you can identify their model, input, output, and evaluation question.
- A mathematical guarantee, an empirical finding, and a causal claim are different kinds of evidence. Note which kind each source provides.
- Entries link to the authors, official documentation, or primary publication pages where possible. Links are for reading; third-party materials are **not** relicensed by this repository.

## Stage 01 — Mathematics and programming foundations

| Priority | Resource | Read when / why |
| --- | --- | --- |
| Core | [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) | Start here: arrays, shapes, indexing, and basic numerical operations for feature vectors. |
| Core | [pandas: Getting started](https://pandas.pydata.org/docs/getting_started/index.html) | Once arrays make sense: read tables, select columns, and inspect data before modeling. |
| Core, selected chapters | [Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* (2020)](https://mml-book.github.io/) | Use the chapters on linear algebra, matrix decompositions, vector calculus, probability, and optimization **as needed**; it is a substantial textbook, not a prerequisite to Lesson 01. |
| Core, later in Stage 01 | [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | After NumPy and basic models: tensors, automatic differentiation, training, and saving models. |

**Checkpoint:** Can you hand-calculate a small linear prediction, reproduce it with arrays, and identify the shape and role of each quantity?

## Stage 02 — Machine learning and deep learning

| Priority | Resource | Read when / why |
| --- | --- | --- |
| Core | [scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html) | After the first model lesson: learn the fit/predict workflow, preprocessing, and model evaluation. Its documentation assumes some basic ML concepts, so use it alongside the lessons rather than as the only introduction. |
| Core, selected sections | [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) | Look up linear models, decision trees, support-vector machines, ensembles, and model selection as they appear in the curriculum. |
| Core for Transformer lessons | [Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)](https://papers.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | Read after a small attention implementation; study the Transformer design and its assumptions, not merely its diagrams. |
| Optional | [Jain & Wallace, *Attention is not Explanation* (NAACL 2019)](https://aclanthology.org/N19-1357/) | When interpreting attention weights: a useful challenge to treating attention scores as explanations by default. |

**Checkpoint:** Can you train and evaluate a simple model on held-out data and explain why a coefficient, feature importance, or attention score is not automatically a real-world causal effect?

## Stage 03 — Classical XAI methods

Begin with a broad overview, then read a paper when you reach its method. For each method, write down the **target output**, **reference or background choice**, **mathematical assumptions**, and **type of claim** its result supports.

| Priority | Resource | Read when / why |
| --- | --- | --- |
| Core overview | [Christoph Molnar, *Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) | Build a map of interpretable models, model-agnostic methods, local versus global explanations, and limitations. Read by topic; no need to finish it first. |
| Core: LIME | [Ribeiro et al., *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier* (KDD 2016)](https://arxiv.org/abs/1602.04938) | Understand perturbation-based local surrogate explanations and what happens when the locality definition changes. |
| Core: SHAP | [Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (NeurIPS 2017)](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | Work through a tiny Shapley-value calculation; identify the chosen value function and the scope of the paper's additive-attribution result. |
| Core: Integrated Gradients | [Sundararajan et al., *Axiomatic Attribution for Deep Networks* (ICML 2017)](https://arxiv.org/abs/1703.01365) | Study gradient-based attribution, the baseline/path, and the stated axioms; check the attribution sum on a small model. |
| Core: Grad-CAM | [Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization* (ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) | Learn how target gradients and convolutional feature maps produce a coarse localization map. |
| Core: TCAV | [Kim et al., *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors* (ICML 2018)](https://proceedings.mlr.press/v80/kim18d.html) | Explore sensitivity to user-defined concepts and the role of concept/control examples. |

**Checkpoint:** Can you implement a toy version of at least one method, check it against a library, and give a case in which the output is easy to misread?

## Stage 04 — Evaluation, mechanisms, and causal research

These are **research directions**, not required reading before starting the repository. Select a branch after completing relevant model and method lessons.

### Evaluation and formalization

| Priority | Resource | Read when / why |
| --- | --- | --- |
| Core | [Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | Reproduce a bounded model-parameter or data-randomization check. The finding concerns **some** methods under specific tests, not all heatmaps. |
| Core, advanced | [*Explainable AI needs formalization* (2026)](https://www.nature.com/articles/s44387-026-00095-1) | Ask what the explanation target and success criterion are before inventing or selecting a method. |

### Mechanistic interpretability

| Priority | Resource | Read when / why |
| --- | --- | --- |
| Core, advanced | [Anthropic, *Toy Models of Superposition* (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) | Study a controlled model of how more features than dimensions can be represented; test the assumptions in a small reproduction. |
| Core, advanced | [Anthropic, *Towards Monosemanticity* (2023)](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) | Study dictionary learning and sparse features after understanding model activations; distinguish a feature hypothesis from a validated circuit. |

### Causal inference and causal XAI

| Priority | Resource | Read when / why |
| --- | --- | --- |
| Core for this branch | [Hernán & Robins, *Causal Inference: What If*](https://miguelhernan.org/whatifbook) | Introduce causal questions, interventions, and assumptions before interpreting model perturbations as claims about the world. |
| Optional | [Pearl, Glymour & Jewell, *Causal Inference in Statistics: A Primer*](https://web.cs.ucla.edu/~kaoru/primer-complete-2019.pdf) | A complementary introduction to graphical causal models, interventions, and counterfactuals. Check the source's reuse terms before redistributing any content. |

**Checkpoint:** Can you state a precise research question, reproduce a limited result with documented settings and controls, and separate what the experiment shows from what it does not establish?

## Maintenance rules

1. Add a source only when it fills a named curriculum need; keep the list curated rather than exhaustive.
2. Prefer original papers and official documentation. Verify titles, publication years, and links before adding entries.
3. When a lesson is published, link it from this list or the lesson to show how the reading is used.
4. Do not label a source **reproduced** unless runnable code and a report are available in this repository.
