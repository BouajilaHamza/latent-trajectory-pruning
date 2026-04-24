# tests/test_data.py
import pytest
from src.data import load_math500_subset

def test_load_math500_subset():
    data = load_math500_subset(num_samples=2)
    assert len(data) == 2
    assert "question" in data[0]
    assert "answer" in data[0]

from src.data import load_gpqa_subset, format_gpqa_prompt

def test_load_gpqa_subset():
    data = load_gpqa_subset(num_samples=2)
    assert len(data) == 2
    assert "question" in data[0]
    assert "(A)" in data[0]["question"]
    assert data[0]["answer"] in ["A", "B", "C", "D"]

def test_format_gpqa_prompt():
    prompt = format_gpqa_prompt(None, "Test Question?")
    assert "Test Question?" in prompt
    assert "correct option is" in prompt

