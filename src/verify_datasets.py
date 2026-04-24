# src/verify_datasets.py
from src.data import load_math500_subset, load_gpqa_subset, format_prompt, format_gpqa_prompt
from transformers import AutoTokenizer

def main():
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        print("Could not load real tokenizer, using dummy.")
        class DummyTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "\n".join([m["content"] for m in messages])
        tokenizer = DummyTokenizer()
        tokenizer.chat_template = "dummy"
    
    print("=== Verifying MATH-500 ===")
    math_data = load_math500_subset(num_samples=1)
    if math_data:
        print("MATH-500 Prompt:\n", format_prompt(tokenizer, math_data[0]["question"]))
        print("\nMATH-500 Expected Answer:\n", math_data[0]["answer"])
    
    print("\n=== Verifying GPQA ===")
    gpqa_data = load_gpqa_subset(num_samples=1)
    if gpqa_data:
        print("GPQA Prompt:\n", format_gpqa_prompt(tokenizer, gpqa_data[0]["question"]))
        print("\nGPQA Expected Answer:\n", gpqa_data[0]["answer"])

if __name__ == "__main__":
    main()
