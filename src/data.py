from datasets import load_dataset
from transformers import PreTrainedTokenizer

def load_gsm8k_subset(split: str = "train", num_samples: int = 1000) -> list[dict]:
    """Loads a subset of the GSM8K dataset."""
    dataset = load_dataset("gsm8k", "main", split=split)
    # Take a small subset for rapid validation
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    return [{"question": item["question"], "answer": item["answer"]} for item in dataset]

def format_prompt(tokenizer: PreTrainedTokenizer, question: str) -> str:
    """Formats the question into a zero-shot CoT prompt."""
    prompt = f"Question: {question}\n\nPlease reason step-by-step and then output your final answer enclosed in \\boxed{{}}."
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful mathematical reasoning assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    data = load_gsm8k_subset(num_samples=2)
    print(format_prompt(tokenizer, data[0]["question"]))
