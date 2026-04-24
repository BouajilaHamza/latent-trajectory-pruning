import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.data import load_gsm8k_subset, format_prompt
from src.evaluator import is_correct

def get_layer_module(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError("Unsupported model architecture.")

def extract_traces(model_name: str, num_samples: int, output_dir: str):
    """Runs the model on GSM8K, extracts hidden states, and saves traces."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()

    dataset = load_gsm8k_subset(num_samples=num_samples)
    
    # We will hook the final layer to capture hidden states
    final_layer_idx = model.config.num_hidden_layers - 1
    layer = get_layer_module(model, final_layer_idx)
    
    all_states = []
    all_labels = []
    
    for i, item in enumerate(tqdm(dataset, desc="Generating & Extracting")):
        prompt = format_prompt(tokenizer, item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[-1]
        
        captured_states = []
        
        def hook(module, inp, out):
            # out[0] is hidden_states of shape (batch, seq_len, hidden_size)
            hidden_states = out[0] if isinstance(out, tuple) else out
            # Detach and move to CPU immediately to save memory
            captured_states.append(hidden_states[:, -1:, :].detach().cpu().squeeze())

        handle = layer.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False, # Greedy decoding for determinism
                )
        finally:
            handle.remove()
            
        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Evaluate correctness
        correct = is_correct(generated_text, item["answer"])
        label = 1 if correct else 0
        
        # The first forward pass processes the whole prompt. We only care about states generated *after* the prompt.
        if len(captured_states) > len(generated_ids):
            # Drop the prompt forward pass state
            gen_states = captured_states[-(len(generated_ids)):]
        else:
            gen_states = captured_states
            
        if not gen_states:
            continue
            
        # Stack into tensor (num_gen_tokens, hidden_dim)
        trajectory_states = torch.stack(gen_states)
        all_states.append(trajectory_states)
        all_labels.extend([label] * len(trajectory_states))
    
    if all_states:
        # Concatenate all token states across all samples
        X = torch.cat(all_states, dim=0)
        y = torch.tensor(all_labels, dtype=torch.long)
        
        torch.save({"X": X, "y": y}, os.path.join(output_dir, "traces.pt"))
        print(f"\nSaved {X.shape[0]} token states to {output_dir}/traces.pt")
        print(f"Label distribution: Correct={sum(all_labels)}, Incorrect={len(all_labels)-sum(all_labels)}")
    else:
        print("No states captured.")

if __name__ == "__main__":
    # Small test run on 0.5B model to verify logic
    extract_traces("Qwen/Qwen2.5-0.5B-Instruct", num_samples=2, output_dir="data")
