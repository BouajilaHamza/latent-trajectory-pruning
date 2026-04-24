from datasets import load_dataset
from transformers import PreTrainedTokenizer
import random

def load_gsm8k_subset(split: str = "train", num_samples: int = 1000) -> list[dict]:
    """Loads a subset of the GSM8K dataset."""
    dataset = load_dataset("gsm8k", "main", split=split)
    # Take a small subset for rapid validation
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    return [{"question": item["question"], "answer": item["answer"]} for item in dataset]

def load_math500_subset(split: str = "test", num_samples: int = 500) -> list[dict]:
    """Loads a subset of the MATH-500 dataset."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", "default", split=split)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    return [{"question": item["problem"], "answer": item["solution"]} for item in dataset]


def load_gpqa_subset(split: str = "train", num_samples: int = 100) -> list[dict]:
    """Loads a subset of the GPQA Diamond dataset (non-gated mirror)."""
    # Wanfq/gpqa is non-gated and contains the diamond subset
    dataset = load_dataset("Wanfq/gpqa", "gpqa_diamond", split="train") 
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    formatted_data = []
    for item in dataset:
        options = [
            item['Incorrect Answer 1'],
            item['Incorrect Answer 2'],
            item['Incorrect Answer 3'],
            item['Correct Answer']
        ]
        random.seed(42) # For reproducibility
        random.shuffle(options)
        
        correct_idx = options.index(item['Correct Answer'])
        correct_letter = chr(65 + correct_idx) # 'A', 'B', 'C', or 'D'
        
        question_text = f"{item['Question']}\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}"
        formatted_data.append({
            "question": question_text,
            "answer": correct_letter
        })
    return formatted_data

def format_gpqa_prompt(tokenizer, question: str) -> str:
    """Formats the GPQA question into a multiple-choice prompt."""
    prompt = f"Question: {question}\n\nPlease reason step-by-step and then output your final choice as 'The correct option is (X)' where X is A, B, C, or D."
    if tokenizer and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful expert answering multiple choice questions."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt

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
