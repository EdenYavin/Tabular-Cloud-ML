"""
Frozen T-Network Context Extractor.

This module provides functionality to extract T_context embeddings from a
frozen, pretrained T network. Used for feature ablation studies to isolate
the contribution of T network representations without retraining.
"""

import pathlib
import numpy as np
import tensorflow as tf
from loguru import logger

from src.utils.helpers import load_pretrained_t_network


class FrozenTContextExtractor:
    """
    Extracts T_context embeddings from a frozen pretrained T network.

    The T network encoder processes anchor pairs (p_i, q_i) and sample p_x
    to produce a 128-dimensional context embedding that captures learned
    representations from the anchor set.

    This class is used in feature ablation studies to provide T_context
    as a feature without retraining the T network.
    """

    def __init__(
        self,
        t_network_path: str | pathlib.Path,
        n_anchors: int,
        raw_dim: int,
        emb_dim: int
    ):
        """
        Load frozen T network and store configuration.

        Args:
            t_network_path: Path to saved T network .keras file
            n_anchors: Number of anchor pairs used in the T network
            raw_dim: Dimension of raw embeddings (p_x, p_i)
            emb_dim: Dimension of encrypted embeddings (q_i)

        Raises:
            FileNotFoundError: If t_network_path does not exist
            ValueError: If loaded model is not a valid T network
        """
        self.n_anchors = n_anchors
        self.raw_dim = raw_dim
        self.emb_dim = emb_dim

        # Load the pretrained T network with frozen weights
        try:
            self.t_network = load_pretrained_t_network(
                t_network_path,
                freeze_weights=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load T network from {t_network_path}: {e}"
            ) from e

        # Extract the encoder submodel (TNetworkOnlyIIM exposes self.encoder)
        if not hasattr(self.t_network, 'encoder'):
            raise ValueError(
                f"Loaded T network does not have 'encoder' attribute. "
                f"Model type: {type(self.t_network).__name__}. "
                f"Ensure you're loading a TNetworkOnlyIIM model with encoder submodel."
            )

        self.encoder = self.t_network.encoder

        logger.info(
            f"FrozenTContextExtractor initialized: "
            f"n_anchors={n_anchors}, raw_dim={raw_dim}, emb_dim={emb_dim}"
        )
        logger.info(f"Encoder output shape: {self.encoder.output_shape}")

    def extract_context(
        self,
        p_x: np.ndarray,  # (batch_size, raw_dim)
        p_i: np.ndarray,  # (batch_size, n_anchors * raw_dim)
        q_i: np.ndarray   # (batch_size, n_anchors * emb_dim)
    ) -> np.ndarray:
        """
        Extract T_context embeddings for a batch of samples.

        Args:
            p_x: Raw sample embeddings (batch_size, raw_dim)
            p_i: Flattened raw anchor embeddings (batch_size, n_anchors * raw_dim)
            q_i: Flattened encrypted anchor embeddings (batch_size, n_anchors * emb_dim)

        Returns:
            T_context embeddings (batch_size, 128)

        Raises:
            ValueError: If input shapes don't match expected dimensions
        """
        batch_size = p_x.shape[0]

        # Validate input shapes
        if p_x.shape[1] != self.raw_dim:
            raise ValueError(
                f"p_x dimension mismatch: expected {self.raw_dim}, got {p_x.shape[1]}"
            )
        if p_i.shape[1] != self.n_anchors * self.raw_dim:
            raise ValueError(
                f"p_i dimension mismatch: expected {self.n_anchors * self.raw_dim}, "
                f"got {p_i.shape[1]}"
            )
        if q_i.shape[1] != self.n_anchors * self.emb_dim:
            raise ValueError(
                f"q_i dimension mismatch: expected {self.n_anchors * self.emb_dim}, "
                f"got {q_i.shape[1]}"
            )

        # Reshape p_i from (batch, n_anchors * raw_dim) to (batch, n_anchors, raw_dim)
        p_i_reshaped = p_i.reshape(batch_size, self.n_anchors, self.raw_dim)

        # Reshape q_i from (batch, n_anchors * emb_dim) to (batch, n_anchors, emb_dim)
        q_i_reshaped = q_i.reshape(batch_size, self.n_anchors, self.emb_dim)

        # Concatenate p_i and q_i along last axis to form anchor pairs
        # Shape: (batch, n_anchors, raw_dim + emb_dim)
        anchor_pairs = np.concatenate([p_i_reshaped, q_i_reshaped], axis=-1)

        # Convert to TensorFlow tensors for encoder call
        anchor_pairs_tf = tf.constant(anchor_pairs, dtype=tf.float32)
        p_x_tf = tf.constant(p_x, dtype=tf.float32)

        # Call encoder with [anchor_pairs, p_x] inputs as list
        context_tf = self.encoder([anchor_pairs_tf, p_x_tf])

        # Convert output back to numpy
        context = context_tf.numpy()

        # Verify output shape
        if context.shape != (batch_size, 128):
            logger.warning(
                f"Unexpected context shape: {context.shape}, "
                f"expected ({batch_size}, 128)"
            )

        logger.debug(f"Extracted context embeddings: shape={context.shape}")

        return context

    def verify_frozen(self) -> bool:
        """
        Verify all T network weights are frozen (trainable=False).

        Returns:
            True if all layers are frozen, False otherwise
        """
        all_frozen = True
        trainable_layers = []

        for layer in self.t_network.layers:
            if layer.trainable:
                all_frozen = False
                trainable_layers.append(layer.name)

        if not all_frozen:
            logger.warning(
                f"Found {len(trainable_layers)} trainable layers: {trainable_layers}"
            )
        else:
            logger.info(
                f"Verified: All {len(self.t_network.layers)} layers are frozen"
            )

        return all_frozen
