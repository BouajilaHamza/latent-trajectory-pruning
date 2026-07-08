# tests/test_evaluator.py
from src.evaluator import is_correct_math, is_correct_gpqa


def test_is_correct_math():
    gt = "The final answer is \\boxed{42}."
    out = "Let's think. The answer is \\boxed{42}."
    assert is_correct_math(out, gt)

    gt_plain = "42"
    assert is_correct_math(out, gt_plain)


def test_is_correct_gpqa():
    gt = "B"
    out1 = "The correct option is (B)."
    out2 = "The correct option is (A)."
    assert is_correct_gpqa(out1, gt)
    assert not is_correct_gpqa(out2, gt)
