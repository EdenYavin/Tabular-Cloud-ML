"""Integration tests for frozen T network loading and context extraction."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.utils.helpers import load_pretrained_t_network
from src.pipeline.frozen_t_context_extractor import FrozenTContextExtractor


def test_load_nonexistent_model():
    """Test that loading nonexistent model raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pretrained_t_network("/tmp/nonexistent_model.keras")


def test_frozen_t_context_extractor_init():
    """Test FrozenTContextExtractor initialization with mock model."""
    # This test is skipped if no pretrained model exists
    # It's a placeholder for when a real model is available
    pytest.skip("Requires pretrained T network model")

    # When implemented with real model:
    # model_path = "path/to/pretrained/model.keras"
    # extractor = FrozenTContextExtractor(
    #     model_path,
    #     n_anchors=5,
    #     raw_dim=64,
    #     emb_dim=768
    # )
    # assert extractor is not None
    # assert extractor.n_anchors == 5
    # assert extractor.raw_dim == 64
    # assert extractor.emb_dim == 768


def test_context_extraction_shapes():
    """Test that extracted context has correct shape (batch_size, 128)."""
    # Skip if no model available
    pytest.skip("Requires pretrained T network model")

    # When implemented with real model:
    # model_path = "path/to/pretrained/model.keras"
    # extractor = FrozenTContextExtractor(
    #     model_path,
    #     n_anchors=5,
    #     raw_dim=64,
    #     emb_dim=768
    # )
    #
    # batch_size = 10
    # p_x = np.random.randn(batch_size, 64)
    # p_i = np.random.randn(batch_size, 5 * 64)
    # q_i = np.random.randn(batch_size, 5 * 768)
    #
    # context = extractor.extract_context(p_x, p_i, q_i)
    # assert context.shape == (batch_size, 128)
    # assert isinstance(context, np.ndarray)


def test_weights_frozen():
    """Test that all T network weights are frozen after loading."""
    pytest.skip("Requires pretrained T network model")

    # When implemented with real model:
    # model_path = "path/to/pretrained/model.keras"
    # extractor = FrozenTContextExtractor(
    #     model_path,
    #     n_anchors=5,
    #     raw_dim=64,
    #     emb_dim=768
    # )
    # assert extractor.verify_frozen() == True
    #
    # # Verify no trainable parameters
    # trainable_params = sum(
    #     np.prod(var.shape) for var in extractor.t_network.trainable_variables
    # )
    # assert trainable_params == 0
