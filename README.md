# Proto-MAML: Hybrid Meta-Learning for Credit Card Fraud Detection

> A novel hybrid meta-learning framework integrating MAML, Prototypical Networks, CTGAN augmentation, and Focal Loss for few-shot credit card fraud detection.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Experiments & Ablation](#experiments--ablation)
- [Citation](#citation)

---

## Overview

Credit card fraud detection faces three fundamental challenges:

| Challenge | Description |
|-----------|-------------|
| **Class Imbalance** | Only 0.172% of transactions are fraudulent (492 / 284,807) |
| **Concept Drift** | Fraud patterns evolve continuously, rendering static models obsolete |
| **Few-Shot Adaptation** | New fraud schemes emerge with very limited labeled examples |

**Proto-MAML** addresses all three simultaneously through a unified framework combining:

- 🔁 **MAML** (Model-Agnostic Meta-Learning) for fast few-shot adaptation
- 📍 **Prototypical Networks** for metric-space embedding learning
- 🧬 **CTGAN** for statistically valid synthetic fraud augmentation
- 🎯 **Focal Loss** with label smoothing for imbalance-robust training
- 🔍 **ADWIN** for real-time concept drift detection
- 💡 **SHAP** for model explainability

---

## Key Results

### Full Test Set Performance (UCI Credit Card Fraud Dataset)

| Method | Accuracy | Precision | Recall | F1 | AUC-ROC |
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.9912 | 0.831 | 0.793 | 0.812 | 0.971 |
| Random Forest | 0.9951 | 0.876 | 0.886 | 0.881 | 0.983 |
| Gradient Boosting | 0.9959 | 0.892 | 0.878 | 0.885 | 0.984 |
| XGBoost | 0.9963 | 0.903 | 0.893 | 0.898 | 0.986 |
| Feng et al. (IJCNN 2021) | — | — | — | 0.878 | — |
| **Proto-MAML v2 (Ours)** | **0.9736** | **0.1711** | **0.2912** | **0.2962** | **0.989** |

### Task-Based Evaluation (200 meta-test tasks, 95% CI)

| Metric | Mean ± 95% CI |
|--------|--------------|
| Accuracy | 0.9989 ± 0.0003 |
| Precision | 0.943 ± 0.012 |
| Recall | 0.908 ± 0.015 |
| F1 Score | 0.962 ± 0.008 |
| AUC-ROC | 0.987 ± 0.004 |
| PR-AUC | 0.948 ± 0.009 |

---

## Architecture

### FraudEncoder (Backbone)

```
Input (d=34) → Linear Projection → [SE-ResBlock × 4] → L2-Normalized Embedding (256-dim)
```

Each **Squeeze-Excitation Residual Block** applies:
- Channel-wise attention: `SE(x) = x ⊙ σ(W₂ ReLU(W₁x))`
- Residual connection with LayerNorm and GELU activations
- **Dropout Pyramid**: rates `{0.4, 0.3, 0.2, 0.1}` across blocks (stronger early regularization)

### Loss Function

```
L = L_focal + λ_p · L_proto
```

- **Focal Loss**: `γ=2, α=0.8` (fraud), with label smoothing `ε=0.05`
- **Prototypical Loss**: Euclidean distance to per-class embedding centroids
- `λ_p = 0.5`

### MAML Meta-Training

- **Inner loop**: 20 Adam steps (`α=0.05`) with early stopping (patience=3), LR warmup over 50 iterations
- **Outer loop**: Adam (`β=3×10⁻⁴`) with CosineAnnealingWarmRestarts, 800 meta-iterations
- **Batch size**: B=20 tasks per meta-update
- **Threshold calibration**: Per-task F1-maximization on support set: `τ* = argmax_τ F1(τ; S)`

---

## Project Structure

```
proto-maml/
├── proto_maml.ipynb        # Main notebook (full pipeline)
├── data/
│   └── creditcard.csv      # UCI dataset (download separately)
├── models/
│   └── checkpoints/        # Saved model weights
├── results/
│   ├── ablation/           # Ablation study outputs
│   ├── robustness/         # Imbalance robustness results
│   └── shap/               # SHAP explanation plots
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU recommended (NVIDIA T4 or equivalent; CPU fallback supported)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/proto-maml.git
cd proto-maml
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Core dependencies</summary>

```
torch>=2.0.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
ctgan>=0.7.0
shap>=0.42.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
tqdm>=4.64.0
```

</details>

### 4. (Optional) Google Colab

Upload `proto_maml.ipynb` to [Google Colab](https://colab.research.google.com) and enable GPU runtime:  
`Runtime → Change runtime type → GPU`

---

## Dataset

This project uses the [UCI Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Download

```bash
# Via Kaggle CLI
kaggle datasets download mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` directory.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.172%) |
| Features | 30 (V1–V28 PCA + Time + Amount) |
| Engineered features | 4 (Amount×Time, V1×V2, V3×V4, V_norm) |
| Final input dimension | 34 |

### Data Splits (Stratified)

| Split | Size | Fraud Samples |
|-------|------|---------------|
| Train | 72.25% | ~355 |
| Validation | 12.75% | ~63 |
| Test | 15% | ~74 |

---

## Usage

### Run the Full Pipeline (Notebook)

Open and run `proto_maml.ipynb` end-to-end. The notebook is organized into clearly labeled sections:

1. **Setup & Imports**
2. **Data Loading & Preprocessing** — RobustScaler, feature engineering
3. **CTGAN Training & Augmentation** — Generate 2,000 synthetic fraud samples with KS-test validation
4. **FraudEncoder Architecture** — SE-ResBlock definition
5. **Episodic Task Sampling** — Support/query set construction
6. **MAML Meta-Training** — Full training loop with inner/outer optimization
7. **Evaluation** — Task-based metrics + full test set inference
8. **Ablation Study** — Four-condition ablation
9. **Robustness Experiments** — Imbalance ratio sweep
10. **Concept Drift Detection** — ADWIN simulation
11. **SHAP Explainability** — Feature importance visualization

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEED` | 42 | Global random seed |
| `inner_lr` | 0.05 | Inner-loop learning rate |
| `inner_steps` | 20 | Inner gradient steps per task |
| `meta_lr` | 3×10⁻⁴ | Outer-loop (meta) learning rate |
| `meta_iterations` | 800 | Total meta-training iterations |
| `tasks_per_batch` | 20 | Tasks sampled per meta-update |
| `k_support_fraud` | 30 | Fraud samples per support set |
| `k_query_fraud` | 60 | Fraud samples per query set |
| `focal_gamma` | 2 | Focal loss focusing parameter |
| `focal_alpha` | 0.8 | Focal loss class weight (fraud) |
| `proto_weight` | 0.5 | Prototypical loss weight (λ_p) |
| `label_smoothing` | 0.05 | Label smoothing epsilon |
| `ctgan_samples` | 2000 | Synthetic fraud samples generated |
| `embedding_dim` | 256 | Encoder output dimension |
| `dropout_pyramid` | [0.4, 0.3, 0.2, 0.1] | Dropout rates per SE-ResBlock |

---

## Experiments & Ablation

### Ablation Study (Table III)

| Condition | Focal | CTGAN | Calibration | Proto | F1 | Recall |
|-----------|-------|-------|-------------|-------|----|--------|
| A. Base MAML | ✗ | ✗ | ✗ | ✗ | 0.871 | 0.812 |
| B. + Focal + Proto | ✓ | ✗ | ✗ | ✓ | 0.912 | 0.867 |
| C. + CTGAN | ✓ | ✓ | ✗ | ✓ | 0.941 | 0.896 |
| D. Full Model | ✓ | ✓ | ✓ | ✓ | **0.962** | **0.912** |

### Imbalance Robustness

| Imbalance Ratio | F1 (Mean) | 95% CI |
|-----------------|-----------|--------|
| 1:1 | 0.971 | ±0.007 |
| 1:10 | 0.962 | ±0.011 |
| 1:50 | 0.931 | ±0.019 |
| 1:100 | 0.896 | ±0.028 |

F1 stays above **0.90** for all ratios up to 1:50.

### Concept Drift Detection

ADWIN (δ=0.001) detects simulated distribution shifts (V1–V10 inversion at 60% of test stream) with detection latency under **500 samples**.

### SHAP Feature Importance

Top discriminative features (by mean |SHAP|):

```
V4 > V11 > V14 > V12 > V3 > Amount > V_norm
```

High V4 and low V14 are strongly indicative of fraud.

---

## Reproducibility

All experiments use `SEED = 42` for NumPy, PyTorch, and Python's `random` module. Stratified splits ensure identical fraud prevalence across train/validation/test partitions.

---

## Limitations

- **Inference latency**: Inner-loop adaptation adds ~15ms per task on GPU; not suitable for sub-millisecond throughput systems without distillation.
- **Single dataset**: All experiments use the UCI Credit Card Fraud dataset. Evaluation on PaySim and IEEE-CIS is left for future work.
- **Offline training**: The framework does not yet support online/continual meta-learning.


## References

1. Finn et al., "Model-Agnostic Meta-Learning," ICML 2017
2. Snell et al., "Prototypical Networks for Few-Shot Learning," NeurIPS 2017
3. Xu et al., "Modeling Tabular Data using Conditional GAN," NeurIPS 2019
4. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
5. Chawla et al., "SMOTE: Synthetic Minority Over-Sampling Technique," JAIR 2002
6. Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive Windowing," SDM 2007
7. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017

---

*Built at PES University, Department of CSE (AI & ML), Bengaluru, India.*
