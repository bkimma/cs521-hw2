# CCS521 HW2 — Q4: Interval Analysis for a Neural Network

Interval analysis (Interval Bound Propagation, IBP) for a fully-connected MNIST
classifier, certifying local L∞ robustness.

## Contents
- [`mnist_interval_analysis.ipynb`](mnist_interval_analysis.ipynb) — the full solution
  (fully executed, with outputs and plot embedded).
- `gen_nb.py` — script that generates the notebook.

## What it does
1. Trains the provided fully-connected network (`Normalize → Linear(784→200) → ReLU →
   Linear(200→10) → softmax`) on MNIST for **15 epochs** and reports clean test accuracy.
2. Implements **interval analysis**: the input L∞ box `[x−ε, x+ε] ∩ [0,1]` is propagated
   layer by layer as elementwise lower/upper bounds (affine via center/radius with `|W|·r`,
   ReLU monotone). A folded **difference layer** computes `f_y − f_j` for every `j ≠ y`.
   The property is proven iff every difference's lower bound is ≥ 0. Bounds are computed on
   the pre-softmax logits (softmax is monotone, so the argmax/robustness property is identical).
3. Reports **verified accuracy** on the whole 10k test set for 10 radii evenly spaced in
   `[0.01, 0.1]`.

## Results
- Clean test accuracy (15 epochs): **93.43%**

| ε | verified accuracy |
|------|------|
| 0.01 | 70.55% |
| 0.02 | 23.68% |
| 0.03 | 4.56% |
| 0.04 | 0.58% |
| 0.05 | 0.02% |
| 0.06–0.10 | 0.00% |

Verified accuracy decreases monotonically with ε: larger boxes widen the propagated
intervals, so fewer `f_y − f_j` lower bounds stay non-negative. IBP is sound but loose,
so these are a conservative lower bound on the true robustness; the network uses standard
(non-robust) training, hence the fast fall-off.

## Running
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision matplotlib numpy jupyter
jupyter notebook mnist_interval_analysis.ipynb
```
