# Latent Trajectory Pruning: Inference-Time Optimization via Hidden State Probing

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/BouajilaHamza/latent-trajectory-pruning)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
Large Reasoning Models (LRMs) like DeepSeek-R1 and OpenAI o1 have demonstrated remarkable performance on complex tasks through extended Chain-of-Thought (CoT) generation. However, this "test-time compute" comes at a significant cost, as models often spend thousands of tokens exploring logically doomed trajectories. We propose **Latent Trajectory Pruning (LTP)**, a mechanism that utilizes lightweight linear probes to monitor a model's internal hidden states during generation. Our preliminary results demonstrate that a model's final answer correctness can be predicted with extremely high accuracy (AUC-ROC > 0.99) from intermediate latent states, providing a foundation for active inference-time pruning and steering.

## 1. Introduction
The efficiency of inference-time scaling is strictly limited by the (N)$ cost of token generation. Current models lack a "metacognitive" layer to evaluate the quality of their own reasoning mid-trajectory. LTP aims to fill this gap by "reading" the model's internal conviction directly from its activation manifold.

## 2. Methodology

### 2.1 Trace Extraction
We hook the final Transformer layer of the target model (e.g., Qwen-2.5-1.5B-Instruct) to capture the hidden state vector $$h_t \in \mathbb{R}^d$ for each generated token in a Chain-of-Thought reasoning sequence.

### 2.2 Latent Process Reward Modeling (LPRM)
Unlike traditional Process Reward Models that evaluate explicit text, our **Latent Probe** operates in the activation space. We train a logistic regression classifier $$W_R$ to predict the final outcome $$y \in \{0, 1\}$ (correct/incorrect) based on the hidden state $$h_t$:
$$ C_t = \sigma(h_t^\top W_R + b) $$
where $$C_t$ is the confidence score at step $.

### 2.3 Evaluation Pipeline
- **Dataset:** GSM8K (Grade School Math 8K)
- **Model:** Qwen-2.5-0.5B-Instruct (initial validation) / Qwen-2.5-1.5B-Instruct
- **Ground Truth:** Automated extraction of numeric answers from `\boxed{}` delimiters.

## 3. Experimental Results (Phase 1)
In our initial validation sprint, we extracted **509 token states** from reasoning trajectories on GSM8K.

| Metric | Value |
| :--- | :--- |
| **Model** | Qwen-2.5-0.5B-Instruct |
| **Test Accuracy** | 96.08% |
| **AUC-ROC** | **0.9976** |
| **Precision (Success)** | 1.00 |
| **Recall (Success)** | 0.91 |

The high AUC-ROC indicates that the "correctness signal" is not just present but highly separable in the latent space long before the model reaches a final conclusion.

## 4. Implementation Status
- [x] **Phase 1: Empirical Validation** (Predicting outcomes from latents)
- [ ] **Phase 2: Inference Intervention** (Early exit/Trajectory steering)
- [ ] **Phase 3: Cross-Model Scaling** (Testing on 7B+ models)

## 5. Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Package manager)

### Installation
```bash
git clone https://github.com/BouajilaHamza/latent-trajectory-pruning.git
cd latent-trajectory-pruning
uv sync
```

### Running Experiments
```bash
# Extract latent traces from GSM8K
uv run python src/extractor.py

# Train and evaluate the linear probe
uv run python src/probe.py
```

## 6. Citation (BibTeX)
*This is a preliminary research project. A formal preprint is forthcoming.*

```bibtex
@article{limam2026latent,
  title={Latent Trajectory Pruning: Inference-Time Optimization via Hidden State Probing},
  author={Limam, Mahmoud and Bouajila, Hamza},
  journal={arXiv preprint (forthcoming)},
  year={2026}
}
```
