# tests/test_evaluator.py
from src.evaluator import extract_ground_truth_math, is_correct_math

def test_is_correct_math():
    gt = "The final answer is \\boxed{42}."
    out = "Let's think. The answer is \\boxed{42}."
    assert is_correct_math(out, gt) == True
    
    gt_plain = "42"
    assert is_correct_math(out, gt_plain) == True

from src.evaluator import is_correct_gpqa

def test_is_correct_gpqa():
    gt = "B"
    out1 = "The correct option is (B)."
    out2 = "The correct option is (A)."
    assert is_correct_gpqa(out1, gt) == True
    assert is_correct_gpqa(out2, gt) == False

