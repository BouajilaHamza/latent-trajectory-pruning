# src/multi_layer_extractor.py
"""Multi-layer hidden state extractor for Task 1.3: Layer-wise Conviction Probing.

Hooks into configurable transformer layers (early, middle, late) during generation
and saves per-layer hidden states alongside correctness labels. This enables
training separate probes per layer to find the earliest reliable failure signal.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.data import load_math500_subset, load_gsm8k_subset, format_prompt
from src.evaluator import is_correct, is_correct_math


def get_layer_module(model, layer_idx: int):
    """Resolves the transformer layer module for a given index across architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError("Unsupported model architecture.")


def select_probe_layers(num_layers: int, num_probes: int = 8) -> list[int]:
    """Selects evenly-spaced layer indices across the model depth.

    Always includes layer 0 (earliest) and layer num_layers-1 (final).
    The remaining probes are evenly distributed in between.

    Args:
        num_layers: Total number of transformer layers in the model.
        num_probes: Number of layers to probe (default 8).

    Returns:
        Sorted list of unique layer indices.
    """
    if num_probes >= num_layers:
        return list(range(num_layers))
    indices = set()
    indices.add(0)
    indices.add(num_layers - 1)
    for i in range(1, num_probes - 1):
        idx = int(round(i * (num_layers - 1) / (num_probes - 1)))
        indices.add(idx)
    return sorted(indices)


def extract_multi_layer_traces(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name: str = "math500",
    num_samples: int = 10,
    num_probes: int = 8,
    max_new_tokens: int = 512,
    output_dir: str = "data",
):
    """Extracts hidden states from multiple layers during greedy generation.

    For each generated token, captures the hidden state at each probed layer.
    Saves a dict mapping layer_idx -> {"X": tensor, "y": tensor} to disk.

    Args:
        model_name: HuggingFace model identifier.
        dataset_name: One of "math500", "gsm8k".
        num_samples: Number of dataset questions to process.
        num_probes: Number of layers to probe.
        max_new_tokens: Max generation length per question.
        output_dir: Directory to save traces.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
        device
    )
    model.eval()

    # Select layers to probe
    num_layers = model.config.num_hidden_layers
    probe_layers = select_probe_layers(num_layers, num_probes)
    print(f"Model has {num_layers} layers. Probing layers: {probe_layers}")

    # Load dataset
    if dataset_name == "math500":
        dataset = load_math500_subset(num_samples=num_samples)
        correctness_fn = is_correct_math
    elif dataset_name == "gsm8k":
        dataset = load_gsm8k_subset(num_samples=num_samples)
        correctness_fn = is_correct
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Per-layer accumulators: layer_idx -> list of (states_tensor, label)
    layer_states: dict[int, list[torch.Tensor]] = {li: [] for li in probe_layers}
    layer_labels: dict[int, list[int]] = {li: [] for li in probe_layers}

    for item in tqdm(dataset, desc=f"Extracting ({dataset_name})"):
        prompt = format_prompt(tokenizer, item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[-1]

        # Per-layer captured states for this question
        captured: dict[int, list[torch.Tensor]] = {li: [] for li in probe_layers}

        # Register hooks on all probe layers
        handles = []
        for layer_idx in probe_layers:
            layer_module = get_layer_module(model, layer_idx)

            def make_hook(li):
                def hook_fn(module, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    # Capture only the last token's hidden state (autoregressive)
                    captured[li].append(hidden[:, -1:, :].detach().cpu().squeeze())

                return hook_fn

            handles.append(layer_module.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
        finally:
            for h in handles:
                h.remove()

        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        correct = correctness_fn(generated_text, item["answer"])
        label = 1 if correct else 0
        num_gen = len(generated_ids)

        # Trim captured states: drop the prompt forward-pass states
        for li in probe_layers:
            states = captured[li]
            if len(states) > num_gen:
                states = states[-num_gen:]
            if not states:
                continue
            trajectory_tensor = torch.stack(states)  # (num_gen, hidden_dim)
            layer_states[li].append(trajectory_tensor)
            layer_labels[li].extend([label] * len(trajectory_tensor))

    # Save per-layer traces
    result = {}
    for li in probe_layers:
        if layer_states[li]:
            X = torch.cat(layer_states[li], dim=0)
            y = torch.tensor(layer_labels[li], dtype=torch.long)
            result[li] = {"X": X, "y": y}
            print(
                f"  Layer {li:3d}: {X.shape[0]} tokens, "
                f"correct={y.sum().item()}, incorrect={len(y)-y.sum().item()}"
            )

    out_path = os.path.join(output_dir, "multi_layer_traces.pt")
    torch.save(result, out_path)
    print(f"\nSaved multi-layer traces to {out_path}")
    print(f"Layers saved: {sorted(result.keys())}")
    return result


if __name__ == "__main__":
    extract_multi_layer_traces(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_name="math500",
        num_samples=5,
        num_probes=8,
        max_new_tokens=256,
        output_dir="data",
    )
