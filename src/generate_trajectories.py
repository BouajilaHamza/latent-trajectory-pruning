# src/generate_trajectories.py
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from src.data import load_math500_subset, format_prompt
from src.evaluator import is_correct_math

def generate_trajectories(model_name: str, num_samples: int, hf_repo_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    dataset = load_math500_subset(num_samples=num_samples)
    records = []
    
    for item in tqdm(dataset, desc="Generating Trajectories"):
        prompt = format_prompt(tokenizer, item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        is_correct = is_correct_math(generated_text, item["answer"])
        
        records.append({
            "question": item["question"],
            "ground_truth": item["answer"],
            "prompt": prompt,
            "trajectory": generated_text,
            "is_correct": is_correct
        })
        
    hf_dataset = Dataset.from_list(records)
    print(f"Pushing dataset to HuggingFace: {hf_repo_id}")
    hf_dataset.push_to_hub(hf_repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repo", type=str, required=True, help="HF repo to push to (e.g., your-username/ltp-trajectories)")
    args = parser.parse_args()
    
    generate_trajectories(args.model, args.samples, args.repo)
