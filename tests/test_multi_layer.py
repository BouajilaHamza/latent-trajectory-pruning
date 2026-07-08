"""Tests for the multi-layer extractor and probing modules."""

import os
import torch
import pytest
from src.multi_layer_extractor import select_probe_layers
from src.multi_layer_probe import train_multi_layer_probes


def test_select_probe_layers():
    # Test cases for layer selection spacing
    layers = select_probe_layers(num_layers=32, num_probes=8)
    assert len(layers) == 8
    assert layers[0] == 0
    assert layers[-1] == 31
    assert layers == sorted(list(set(layers)))  # check unique and sorted

    # Check boundary cases where num_probes >= num_layers
    layers_small = select_probe_layers(num_layers=5, num_probes=10)
    assert len(layers_small) == 5
    assert layers_small == [0, 1, 2, 3, 4]


def test_train_multi_layer_probes_file_not_found():
    with pytest.raises(FileNotFoundError):
        train_multi_layer_probes("nonexistent_traces_file.pt")
