
# Self-Pruning Neural Network — Tredence AI Engineering Internship

> **Case Study:** Implementing a feed-forward neural network that learns to prune its own weights during training using learnable sigmoid gates and L1 sparsity regularization, evaluated on CIFAR-10.

---

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Why L1 + Sigmoid Encourages Sparsity](#why-l1--sigmoid-encourages-sparsity)
- [Key Design Decisions & Bug Fixes](#key-design-decisions--bug-fixes)
- [File Structure](#file-structure)

---

## Project Overview

Standard neural network pruning removes unimportant weights **after** training. This project implements **dynamic self-pruning**: the network learns *during* training which weights are unnecessary and eliminates them on the fly.

The core idea is a learnable **gate** parameter for every weight. If a gate value collapses to 0, that weight is effectively removed from the network — no separate pruning step needed.

**Task:** Image classification on CIFAR-10 (10 classes, 60,000 images, 32×32 RGB)

---

## How It Works

### The Gating Mechanism

Each weight `w` in a `PrunableLinear` layer has a corresponding learnable scalar `gate_score`. During the forward pass:

```
gate        = sigmoid(gate_score)          # ∈ (0, 1)
pruned_w    = weight × gate               # element-wise multiply
output      = pruned_w @ input + bias     # standard linear op
```

- If `gate_score → +∞`, then `gate → 1` → weight is **fully active**
- If `gate_score → -∞`, then `gate → 0` → weight is **effectively pruned**

Gradients flow through both `weight` and `gate_score` automatically via autograd.

### The Loss Function

```
Total Loss = CrossEntropyLoss(predictions, labels)
           + λ × SparsityLoss

SparsityLoss = mean of all gate values across all PrunableLinear layers
             = (Σ sigmoid(gate_score_i)) / total_gates
```

The L1 penalty on gate values creates a constant gradient pressure pushing all gates toward zero. The optimizer must balance this against the classification gradient, which needs some gates to stay open for accurate predictions.

---

## Architecture

```
Input (3×32×32 CIFAR-10 image)
       │
       ▼
   Flatten → 3072 features
       │
       ▼
PrunableLinear(3072 → 512) + ReLU + Dropout(0.3)   [~1.57M gated weights]
       │
       ▼
PrunableLinear(512 → 256)  + ReLU + Dropout(0.3)   [~131K gated weights]
       │
       ▼
PrunableLinear(256 → 128)  + ReLU                  [~32K gated weights]
       │
       ▼
  Linear(128 → 10)          [standard, not pruned]
       │
       ▼
  Class Scores (logits)
```

**Total gated weights:** 1,736,704 across 3 PrunableLinear layers

---

## Installation

```bash
# Run in Google Colab or local environment
pip install torch torchvision matplotlib numpy
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- CUDA GPU recommended (T4 or better on Colab)

---

## Usage

Open `AI_ENG_INTERN_Tredence.ipynb` in Google Colab and run blocks sequentially:

| Block | Description |
|-------|-------------|
| Block 1 | Install dependencies |
| Block 2 | Imports & device setup (auto-detects GPU) |
| Block 3 | `PrunableLinear` layer definition |
| Block 4 | `SelfPruningNet` model definition |
| Block 5 | Download & load CIFAR-10 dataset |
| Block 6 | Training & evaluation functions |
| Block 7 | Main training loop — 3 λ values × 25 epochs |
| Block 8 | Results table |
| Block 9 | Training curves plot |
| Block 10 | Gate distribution histogram |
| Block 11 | Markdown report |
| Block 12 | Final summary |

**Tip:** Use `Runtime → Change runtime type → T4 GPU` in Colab for ~5× speedup.

---

## Results

### Summary Table

| Lambda (λ) | Test Accuracy | Sparsity Level | Pruning Strength |
|-----------|--------------|----------------|-----------------|
| **0.5**   | 58.21%       | 37.70%         | Mild            |
| **1.5**   | 58.31%       | 73.66%         | Medium          |
| **4.0**   | **58.81%**   | **90.04%**     | Aggressive      |

### Key Observations

- **Pruning works with minimal accuracy cost.** The λ=4.0 model prunes 90% of weights yet achieves the *highest* test accuracy (58.81%) — stronger regularization here acts as a beneficial constraint, preventing overfitting.
- **Accuracy is remarkably stable across sparsity levels.** The difference between 37.7% and 90.0% sparsity is only 0.6% accuracy — demonstrating the network has massive redundancy.
- **Sparsity plateaus.** For λ=0.5, sparsity stabilizes at ~37.7% by epoch 20; for λ=4.0, it hits 90% by epoch 20 and holds steady — the network reaches an equilibrium where remaining gates are too important to prune further.

### Training Progression (λ = 4.0)

| Epoch | Classification Loss | Sparsity Loss | Test Accuracy | Sparsity | Mean Gate |
|-------|---------------------|---------------|--------------|----------|-----------|
| 1     | 1.7562              | 0.9069        | 45.3%        | 0.0%     | 0.891     |
| 5     | 1.3501              | 0.2610        | 52.0%        | 81.3%    | 0.576     |
| 10    | 1.1660              | 0.1598        | 55.9%        | 87.8%    | 0.483     |
| 15    | 1.0474              | 0.1333        | 58.1%        | 89.4%    | 0.445     |
| 20    | 0.9798              | 0.1246        | 58.6%        | 90.0%    | 0.431     |
| 25    | 0.9547              | 0.1229        | 58.8%        | 90.0%    | 0.428     |

### Gate Statistics — Best Model (λ = 4.0)

```
Total gates       :  1,736,704
Pruned (< 0.01)   :    769,185  (44.29%)
Active (> 0.5)    :    172,982   (9.96%)
Mean gate value   :      0.1229
Median gate value :      0.0130
```

The bimodal distribution — large spike near 0, small cluster near 1 — confirms successful self-pruning.

---

## Why L1 + Sigmoid Encourages Sparsity

**Total Loss = CrossEntropyLoss + λ × SparsityLoss**

where **SparsityLoss = mean of sigmoid(gate_score_i)** across all weights.

### Two Key Mechanisms

**1. Sigmoid constrains gates to (0, 1)**

Raw `gate_scores` are unconstrained real numbers. Applying `sigmoid` maps them to (0, 1), making them interpretable as "how much of this weight to keep." Since all values are positive, the L1 norm (sum of absolute values) equals the plain sum — every unit of gate activation incurs a direct penalty.

**2. L1 drives values to exactly zero**

Unlike L2 regularization, which shrinks values *toward* zero with diminishing force (gradient ∝ value), the L1 norm applies a **constant gradient** of ±1 everywhere. This constant pull means:

- Unimportant gates get pushed down with equal force regardless of their current value
- Many gates reach near-zero (gate_score → −∞, sigmoid → 0)
- The corresponding pruned weight ≈ `weight × 0 ≈ 0` — that connection is neutralized

Higher λ amplifies the penalty → more aggressive pruning → higher sparsity.

---

## Key Design Decisions & Bug Fixes

### Bug 1: Raw Sparsity Loss Was 857,000× Too Large

**Problem:** Summing all 1.7M gate values (each ~0.5) gives SparsityLoss ≈ 857,000. Even with λ=1e-3, the penalty dominated every gradient update but paradoxically never drove any individual gate to zero — Adam's adaptive learning rate neutralized the massive but uniform signal.

**Fix:** Normalize by the number of gates so SparsityLoss ∈ (0, 1), comparable to CrossEntropyLoss (~1.7).

```python
# Before (broken): sum of 1.7M values = ~857,000
sparse_loss = model.compute_sparsity_loss()

# After (fixed): mean = ~0.5, comparable to cls_loss
sparse_loss = model.compute_sparsity_loss() / total_gates
```

### Bug 2: Gate Initialization at sigmoid(0) = 0.5

**Problem:** Starting gate_scores at 0 means gates begin at 0.5 — right at the sigmoid's inflection point. To prune (gate < 0.01), the optimizer must push scores from 0 to below −4.6. The sigmoid gradient vanishes as scores go negative, making this push slower and slower.

**Fix:** Initialize gate_scores to +3 so gates start at sigmoid(+3) ≈ 0.95 (fully open). The optimizer now has a long, clear gradient highway to push scores into negative territory.

```python
# Before (broken): gates start at 0.5, ambiguous direction
nn.init.constant_(self.gate_scores, 0.0)

# After (fixed): gates start at 0.95, clear direction to push down
nn.init.constant_(self.gate_scores, 3.0)
```

### Design Decision: Separate Optimizer Groups

Gate parameters use a 5× higher learning rate than weights. This ensures gates can move toward zero faster than weights "learn around" them.

```python
optimizer = optim.Adam([
    {'params': other_params, 'lr': 1e-3,  'weight_decay': 1e-4},
    {'params': gate_params,  'lr': 5e-3,  'weight_decay': 0.0},
])
```

### Design Decision: Sparsity Threshold = 0.5

Using the natural sigmoid decision boundary (gate < 0.5 means gate_score < 0) rather than a tiny value like 1e-2 gives a meaningful measure of how many weights are "on" vs "off" from the optimizer's perspective.

---

## File Structure

```
AI_ENG_INTERN_Tredence.ipynb   ← Main notebook (all 12 blocks)
README.md                       ← This file
training_curves.png             ← Accuracy, sparsity & loss curves (saved by Block 9)
gate_distribution.png           ← Gate value histogram (saved by Block 10)
data/                           ← CIFAR-10 dataset (auto-downloaded by Block 5)
```

---

## Figure

<img width="469" height="358" alt="image" src="https://github.com/user-attachments/assets/948dfc20-3db2-4143-8a52-30b57b228487" />

<img width="726" height="472" alt="image" src="https://github.com/user-attachments/assets/1b226eea-e9df-44d5-824e-dc2d5bbe31c0" />



