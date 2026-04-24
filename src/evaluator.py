import re

def extract_ground_truth(answer_str: str) -> str:
    """Extracts the final numeric answer from the GSM8K ground truth string (after ####)."""
    parts = answer_str.split("####")
    if len(parts) > 1:
        return parts[1].strip().replace(",", "")
    return ""

def extract_model_answer(text: str) -> str | None:
    """Extracts the content inside the last \boxed{} in the model's output."""
    # Find all occurrences of \boxed{...}
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip().replace(",", "")
    return None

def is_correct(model_text: str, ground_truth_str: str) -> bool:
    """Returns True if the extracted model answer matches the ground truth."""
    truth = extract_ground_truth(ground_truth_str)
    model_ans = extract_model_answer(model_text)
    if truth and model_ans:
        # Simple string equality for numbers; could be improved later
        return truth == model_ans
    return False

if __name__ == "__main__":
    gt = "The man has 10 apples. #### 10"
    out1 = "Let's think. He has 5+5. The answer is \\boxed{10}."
    out2 = "Let's think. The answer is \\boxed{12}."
    assert is_correct(out1, gt) == True
    assert is_correct(out2, gt) == False
    print("Evaluator tests passed.")
