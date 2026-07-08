# North Star Research Specification: Latent Trajectory Pruning (LTP)

**Target Venues:** ICLR 2025 / NeurIPS 2025 (Core A*)  
**Primary Track:** LLM Efficiency, Reasoning, and Mechanistic Interpretability  
**Drafting Template:** [NeurIPS/ICLR LaTeX Template](https://github.com/ICLR/iclr_latex_template)

---

## 1. Executive Summary & Research Question
Current Large Reasoning Models (LRMs) achieve high performance through "test-time compute" (Chain-of-Thought). However, this scaling is inefficient; models frequently generate hundreds of "doomed" tokens after a logical error. 

**Core Research Question:** Can we detect the internal "conviction" of correctness within a model's latent manifold *before* it generates the final answer, and can we use this signal to halt or steer inference to optimize the compute-performance Pareto frontier across complex reasoning domains (Math, Science, Logic)?

---

## 2. Technical Roadmap (The "No Shortcuts" Plan)

### Phase 1: Ground Truth & Rigorous Probing (Short-Term)
*Goal: Move from "shallow" token-labeling to "deep" process-labeling across diverse reasoning tasks.*
- **[Task 1.1] Dataset Expansion:** Move beyond GSM8K to **MATH-500** and **GPQA Diamond**. These datasets provide the long, complex trajectories where pruning is most valuable.
- **[Task 1.2] Process-Level Annotation (Teacher-Model):** Use a "Teacher" model (Qwen-2.5-72B-Instruct or GPT-4o) to annotate trajectories step-by-step. Identify the **Point of Failure (PoF)**—the first token where the reasoning goes irrecoverably wrong.
- **[Task 1.3] Multi-Layer Probing & Layer-wise Conviction:** Extract hidden states from *all* layers. Train probes to identify the earliest layer that reliably signals failure (e.g., middle vs. late).
- **[Task 1.4] Baseline Implementation:**
    - **Self-Consistency (SC) Baseline:** $N$ paths with voting.
    - **Text-PRM Baseline:** Implement a text-based Process Reward Model.
    - **Entropy-based Stopping:** Simple token-probability entropy as a baseline stopping criteria.

### Phase 2: Active Inference Intervention (Mid-Term)
*Goal: Implement the "Pruning" and verify the Efficiency-Accuracy Pareto Frontier.*
- **[Task 2.1] Early Stopping & Halting:** Develop a custom `StoppingCriteria` for Hugging Face `generate()`. If the probe's confidence $C_t < \tau$ for $k$ tokens, terminate the sequence.
- **[Task 2.2] Latent-Guided Search (Backtracking):** Implement a "Search-based Pruning" where a pruned branch triggers a "re-roll" from the last high-confidence latent state.
- **[Task 2.3] Pareto Analysis:** Rigorously map the exact reduction in FLOPs/tokens vs. the maintenance of accuracy. Prove we achieve a superior Pareto frontier compared to SC and Text-PRM.

### Phase 3: Scaling, Generalization & Interpretability (Long-Term)
*Goal: Prove universality and provide mechanistic insights.*
- **[Task 3.1] Cross-Model Generalization:** Test on **Llama-3-8B** and **Qwen-2.5-7B/14B**.
- **[Task 3.2] Zero-Shot Generalization:** Can a probe trained on MATH-500 detect failures in GPQA Diamond or coding tasks without retraining?
- **[Task 3.3] Mechanistic Analysis (SVD/LDA/Sparse Autoencoders):** Analyze the latent features the probe is using. Is it detecting "logical contradiction," "arithmetic confusion," or "hallucination"?
- **[Task 3.4] Drafting & ARR Submission:** Write the paper in the ICLR/NeurIPS LaTeX template, with a focus on reproducibility and the "compute-optimal reasoning" narrative.

---

## 3. Publication Strategy (ARR/Core A Standards)

### Essential Baselines (Must-Haves):
1. **Self-Consistency (SC):** Does LTP + Greedy beat SC with the same compute budget?
2. **Text-PRM:** Is latent probing faster/more accurate than reading the text?
3. **Logits-based Entropy:** Does our latent probe provide more information than raw token probabilities?

### Key Metrics for Core A:
- **AUC-ROC (Step-level):** Accuracy of predicting the *next* step's correctness.
- **Compute Savings (Tokens Saved %):** Average reduction in trajectory length.
- **Recovery Rate:** Percentage of "doomed" paths that were successfully steered to a correct answer via backtracking.
- **Latent Specificity:** Proof that the "conviction" signal is distinct from simple perplexity.

---

## 4. Execution Protocol
We will follow this plan **line-by-line**. No "shortcuts" (e.g., using only small models or simple datasets). Each task requires verification before proceeding.

**Current Status:** Phase 1 - *Transitioning from GSM8K only to MATH-500/GPQA and starting Task 1.2 (Process-Level Annotation).*
