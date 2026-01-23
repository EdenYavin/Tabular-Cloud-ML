"""
Key Encoder for learning functional embeddings of encryption keys.

Uses contrastive learning (InfoNCE) to learn embeddings where:
- Same key (different calibration samples) -> close embeddings
- Different keys -> far embeddings
"""

import tensorflow as tf
from keras.src.models import Model
from keras.src import losses
import numpy as np
from loguru import logger

from src.meta_learning.set_transformer import SetTransformer


class InfoNCELoss(tf.keras.losses.Loss):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    Given anchor embeddings and positive embeddings from the same key,
    pulls them together while pushing away negative samples from different keys.

    Loss = -log( exp(sim(anchor, positive) / tau) / sum_i(exp(sim(anchor, negative_i) / tau)) )
    """

    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, anchors, positives):
        """
        Args:
            anchors: Embeddings from context A (batch_size, embedding_dim)
            positives: Embeddings from context B (same keys) (batch_size, embedding_dim)

        Returns:
            InfoNCE loss value
        """
        # Normalize embeddings
        anchors = tf.nn.l2_normalize(anchors, axis=1)
        positives = tf.nn.l2_normalize(positives, axis=1)

        # Compute similarity matrix (batch_size, batch_size)
        # similarity[i, j] = cosine_similarity(anchor_i, positive_j)
        similarity_matrix = tf.matmul(anchors, positives, transpose_b=True)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Labels: diagonal elements are positives (same key)
        batch_size = tf.shape(anchors)[0]
        labels = tf.range(batch_size)

        # InfoNCE is categorical cross-entropy where positive is on diagonal
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels,
            similarity_matrix,
            from_logits=True
        )

        return tf.reduce_mean(loss)


class KeyEncoder(Model):
    """
    Key Encoder model that wraps SetTransformer and adds contrastive training.

    Architecture:
    1. SetTransformer processes calibration pairs
    2. Outputs functional embedding (z_k) representing the encryption key
    3. Trained with InfoNCE to cluster same keys, separate different keys
    """

    def __init__(self,
                 calibration_dim,
                 encrypted_dim,
                 embedding_dim=256,
                 temperature=0.07,
                 **kwargs):
        super().__init__(**kwargs)

        self.calibration_dim = calibration_dim
        self.encrypted_dim = encrypted_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # Core SetTransformer
        self.set_transformer = SetTransformer(
            pair_dim=calibration_dim + encrypted_dim,
            hidden_dim=128,
            embedding_dim=embedding_dim,
            num_inducing_points=32,
            num_isab_blocks=2,
            num_heads=4
        )

        # Contrastive loss
        self.contrastive_loss = InfoNCELoss(temperature=temperature)

        # Metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.alignment_tracker = tf.keras.metrics.Mean(name="alignment")
        self.uniformity_tracker = tf.keras.metrics.Mean(name="uniformity")

    def call(self, calibration_pairs, training=None):
        """
        Forward pass through SetTransformer.

        Args:
            calibration_pairs: (batch_size, num_pairs, pair_dim)

        Returns:
            functional_embedding: (batch_size, embedding_dim)
        """
        return self.set_transformer(calibration_pairs, training=training)

    def train_step(self, data):
        """
        Custom training step for contrastive learning.

        Args:
            data: Tuple of (context_A, context_B) where both are (batch, n_pairs, pair_dim)
                 Context A and B are different calibration samples from the SAME keys.
        """
        context_A, context_B = data

        with tf.GradientTape() as tape:
            # Encode both contexts
            embeddings_A = self(context_A, training=True)
            embeddings_B = self(context_B, training=True)

            # Contrastive loss
            loss = self.contrastive_loss(embeddings_A, embeddings_B)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)

        # Compute alignment (positive pair similarity)
        alignment = self._compute_alignment(embeddings_A, embeddings_B)
        self.alignment_tracker.update_state(alignment)

        # Compute uniformity (distribution spread)
        uniformity = self._compute_uniformity(embeddings_A)
        self.uniformity_tracker.update_state(uniformity)

        return {
            "loss": self.loss_tracker.result(),
            "alignment": self.alignment_tracker.result(),
            "uniformity": self.uniformity_tracker.result()
        }

    def test_step(self, data):
        """
        Validation step.
        """
        context_A, context_B = data

        # Encode both contexts
        embeddings_A = self(context_A, training=False)
        embeddings_B = self(context_B, training=False)

        # Contrastive loss
        loss = self.contrastive_loss(embeddings_A, embeddings_B)

        # Update metrics
        self.loss_tracker.update_state(loss)

        alignment = self._compute_alignment(embeddings_A, embeddings_B)
        self.alignment_tracker.update_state(alignment)

        uniformity = self._compute_uniformity(embeddings_A)
        self.uniformity_tracker.update_state(uniformity)

        return {
            "loss": self.loss_tracker.result(),
            "alignment": self.alignment_tracker.result(),
            "uniformity": self.uniformity_tracker.result()
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.alignment_tracker, self.uniformity_tracker]

    def _compute_alignment(self, embeddings_A, embeddings_B):
        """
        Compute alignment: average cosine similarity between positive pairs.
        Should be close to 1 (embeddings from same key should be similar).
        """
        # Normalize
        emb_A_norm = tf.nn.l2_normalize(embeddings_A, axis=1)
        emb_B_norm = tf.nn.l2_normalize(embeddings_B, axis=1)

        # Cosine similarity for positive pairs (diagonal)
        similarity = tf.reduce_sum(emb_A_norm * emb_B_norm, axis=1)

        return tf.reduce_mean(similarity)

    def _compute_uniformity(self, embeddings):
        """
        Compute uniformity: measures how uniformly embeddings are distributed.
        Lower uniformity = embeddings are more spread out (desirable).
        """
        # Normalize embeddings
        embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)

        # Pairwise similarity matrix
        similarity_matrix = tf.matmul(embeddings_norm, embeddings_norm, transpose_b=True)

        # Exclude diagonal (self-similarity)
        batch_size = tf.shape(embeddings)[0]
        mask = 1.0 - tf.eye(batch_size)

        # Average off-diagonal similarities (should be low for good uniformity)
        off_diagonal = similarity_matrix * mask
        uniformity = tf.reduce_sum(off_diagonal) / tf.reduce_sum(mask)

        return uniformity

    def encode_calibration_pairs(self, calibration_pairs):
        """
        Convenience method to encode calibration pairs to functional embeddings.

        Args:
            calibration_pairs: (batch_size, num_pairs, pair_dim) or single (num_pairs, pair_dim)

        Returns:
            functional_embedding: (batch_size, embedding_dim) or (embedding_dim,)
        """
        # Add batch dimension if needed
        if len(calibration_pairs.shape) == 2:
            calibration_pairs = np.expand_dims(calibration_pairs, axis=0)
            squeeze_output = True
        else:
            squeeze_output = False

        embeddings = self(calibration_pairs, training=False)

        if squeeze_output:
            embeddings = tf.squeeze(embeddings, axis=0)

        return embeddings.numpy()

    def get_config(self):
        return {
            'calibration_dim': self.calibration_dim,
            'encrypted_dim': self.encrypted_dim,
            'embedding_dim': self.embedding_dim,
            'temperature': self.temperature
        }


def create_key_encoder(calibration_dim, encrypted_dim, embedding_dim=256):
    """
    Factory function to create a KeyEncoder.

    Args:
        calibration_dim: Dimension of calibration vectors
        encrypted_dim: Dimension of encrypted outputs
        embedding_dim: Dimension of functional embeddings

    Returns:
        KeyEncoder model
    """
    encoder = KeyEncoder(
        calibration_dim=calibration_dim,
        encrypted_dim=encrypted_dim,
        embedding_dim=embedding_dim,
        temperature=0.07
    )

    return encoder
