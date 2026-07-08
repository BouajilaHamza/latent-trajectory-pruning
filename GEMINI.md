# Latent Trajectory Pruning: Project Intelligence

This document provides essential context and instructions for the Latent Trajectory Pruning (LTP) research project. LTP aims to optimize inference-time compute by pruning logically doomed reasoning trajectories using hidden state probes.

## Project Overview

- **Purpose:** Inference-time optimization via mechanistic interpretability. We predict final answer correctness from intermediate latent states to enable early exiting or trajectory steering.
- **Key Files:**
  - `src/extractor.py`: Hooks into Transformer layers to extract hidden states during CoT generation.
  - `src/probe.py`: Trains and evaluates linear classifiers (probes) on extracted latents.
  - `src/data.py`: Standardized loaders for GSM8K, MATH-500, and GPQA Diamond.
  - `src/evaluator.py`: Logic for extracting answers (e.g., from `\boxed{}`) and verifying correctness.
  - `docs/research/NORTH_STAR_PLAN.md`: The long-term research roadmap and publication strategy.

## Tech Stack

- **Language:** Python 3.13+
- **Package Manager:** `uv` (Required for all operations)
- **Libraries:** `transformers`, `datasets`, `torch`, `scikit-learn`, `pytest`

## Building and Running

Always use `uv run` to ensure the correct environment and dependencies.

### Installation
```bash
uv sync
```

### Execution
```bash
# Extract latent traces from datasets
uv run python src/extractor.py

# Train and evaluate the linear probe
uv run python src/probe.py

# Verify the data loading pipeline
uv run python src/verify_datasets.py
```

### Testing
Use `pytest` for all verifications. Add `PYTHONPATH=.` if necessary to resolve the `src` module.
```bash
PYTHONPATH=. uv run pytest
```

## Development Conventions

- **Tooling:** Prefer `uv run` for all command executions.
- **Testing:** Follow Test-Driven Development (TDD). Create failing tests in `tests/` before implementing features or fixes.
- **Surgical Edits:** Make precise, targeted changes. Respect existing architectural boundaries between data loading, trace extraction, and evaluation logic.
- **Data Standardization:** All data loaders in `src/data.py` must return a list of dictionaries in the format `[{"question": ..., "answer": ...}]`.
- **Reproducibility:** Use fixed random seeds (e.g., `42`) for dataset shuffling and model training to ensure consistent research results.
- **linting:** use uv and ruff for linting mypy too after any imeplementation to clean the code properly
## Roadmap (Phase 1 Status)
We are currently in **Phase 1: Ground Truth & Rigorous Probing**.
- [x] Task 1.1: Dataset Expansion (GSM8K, MATH-500, GPQA Diamond implemented)
- [ ] Task 1.2: Process-Level Annotation (Teacher-Model labeling)
- [ ] Task 1.3: Multi-Layer Probing
- [ ] Task 1.4: Baseline Implementation (SC, Text-PRM)
