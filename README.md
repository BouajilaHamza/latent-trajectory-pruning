# Latent Trajectory Pruning

Research on optimizing Large Reasoning Model (LRM) inference by pruning doomed reasoning paths using latent hidden state probes.

## Core Hypothesis
The latent hidden states of a reasoning model (e.g., Qwen-2.5-Instruct) contain a decodable signal that predicts the final correctness of the reasoning trajectory long before the final answer is generated.

## Phase 1 Results (Validation Sprint)
Using **Qwen-2.5-0.5B-Instruct** on **GSM8K**:
- **Metric:** AUC-ROC in predicting final success from intermediate hidden states.
- **Result:** **0.9976** (initial validation on small subset).
- **Finding:** The model's "conviction" about the final answer is strongly encoded in the final layer's hidden states during the Chain-of-Thought process.

## Pipeline
1. **Trace Extraction:** Capture hidden states $h_t$ for every token during CoT.
2. **Evaluator:** Automatically grade solutions against ground truth.
3. **Probe:** Train a linear classifier to map $h_t \rightarrow \{0, 1\}$.
4. **Pruning (Upcoming):** Trigger early-exit or rethinking based on probe confidence.

## Usage
```bash
# 1. Extract traces
uv run python src/extractor.py

# 2. Train probe
uv run python src/probe.py
```
