"""
Set Transformer implementation for permutation-invariant processing of calibration pairs.

Based on "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
(Lee et al., 2019)

Key components:
- ISAB: Induced Set Attention Block for efficient O(nm) attention
- PMA: Pooling by Multihead Attention for set aggregation
- SetTransformer: Full architecture combining ISAB + PMA
"""

import tensorflow as tf
from keras.src.layers import Layer, Dense, LayerNormalization, MultiHeadAttention
from keras.src.models import Model
from keras.src.layers import Input
import numpy as np


class MultiheadAttentionBlock(Layer):
    """
    Multihead Attention Block (MAB) as defined in Set Transformer paper.
    MAB(X, Y) = LayerNorm(H + rFF(H)) where H = LayerNorm(X + Attention(X, Y))
    """

    def __init__(self, dim, num_heads=4, ff_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or dim * 4

        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

        self.ff = tf.keras.Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dense(dim)
        ])

    def call(self, X, Y=None):
        """
        Args:
            X: Query matrix (batch_size, n, dim)
            Y: Key/Value matrix (batch_size, m, dim). If None, Y = X (self-attention)
        """
        if Y is None:
            Y = X

        # Attention with residual
        H = self.attention(query=X, key=Y, value=Y)
        H = self.layernorm1(X + H)

        # Feed-forward with residual
        out = self.ff(H)
        out = self.layernorm2(H + out)

        return out


class ISAB(Layer):
    """
    Induced Set Attention Block (ISAB).

    Reduces complexity from O(n^2) to O(nm) where m << n using learned inducing points.
    ISAB(X) = MAB(X, H) where H = MAB(I, X) and I are m learnable inducing points.
    """

    def __init__(self, dim, num_inducing_points=32, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_inducing_points = num_inducing_points
        self.num_heads = num_heads

        # Learnable inducing points
        self.inducing_points = self.add_weight(
            shape=(1, num_inducing_points, dim),
            initializer='glorot_uniform',
            trainable=True,
            name='inducing_points'
        )

        self.mab1 = MultiheadAttentionBlock(dim, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads)

    def call(self, X):
        """
        Args:
            X: Input set (batch_size, n, dim)
        Returns:
            Output set (batch_size, n, dim)
        """
        batch_size = tf.shape(X)[0]

        # Expand inducing points for batch
        I = tf.tile(self.inducing_points, [batch_size, 1, 1])

        # H = MAB(I, X) - inducing points attend to input
        H = self.mab1(I, X)

        # Output = MAB(X, H) - input attends to induced representations
        out = self.mab2(X, H)

        return out


class PMA(Layer):
    """
    Pooling by Multihead Attention (PMA).

    Aggregates a set into a fixed number of seed vectors using attention.
    PMA(X) = MAB(S, X) where S are k learnable seed vectors.
    """

    def __init__(self, dim, num_seeds=1, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads

        # Learnable seed vectors for pooling
        self.seed_vectors = self.add_weight(
            shape=(1, num_seeds, dim),
            initializer='glorot_uniform',
            trainable=True,
            name='seed_vectors'
        )

        self.mab = MultiheadAttentionBlock(dim, num_heads)

    def call(self, X):
        """
        Args:
            X: Input set (batch_size, n, dim)
        Returns:
            Pooled output (batch_size, num_seeds, dim)
        """
        batch_size = tf.shape(X)[0]

        # Expand seed vectors for batch
        S = tf.tile(self.seed_vectors, [batch_size, 1, 1])

        # Seeds attend to input set
        out = self.mab(S, X)

        return out


class SetTransformer(Model):
    """
    Complete Set Transformer for processing calibration pairs.

    Architecture:
    Input: Set of (x_cal, y_enc) pairs -> shape (batch, n_pairs, pair_dim)
    1. Point-wise encoder MLP
    2. ISAB x 2 for relational reasoning
    3. PMA for set pooling
    4. Output MLP for functional embedding
    """

    def __init__(self,
                 pair_dim,
                 hidden_dim=128,
                 embedding_dim=256,
                 num_inducing_points=32,
                 num_isab_blocks=2,
                 num_heads=4,
                 **kwargs):
        super().__init__(**kwargs)

        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Point-wise encoder
        self.point_encoder = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            LayerNormalization(),
            Dense(hidden_dim, activation='relu'),
            LayerNormalization()
        ])

        # Stack of ISAB blocks
        self.isab_blocks = [
            ISAB(hidden_dim, num_inducing_points, num_heads)
            for _ in range(num_isab_blocks)
        ]

        # Pooling layer
        self.pma = PMA(hidden_dim, num_seeds=1, num_heads=num_heads)

        # Output projection
        self.output_mlp = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(embedding_dim)
        ])

    def call(self, calibration_pairs):
        """
        Args:
            calibration_pairs: (batch_size, n_pairs, pair_dim)
                               where pair_dim = x_cal_dim + y_enc_dim
        Returns:
            functional_embedding: (batch_size, embedding_dim)
        """
        # Point-wise encoding
        x = self.point_encoder(calibration_pairs)  # (batch, n_pairs, hidden_dim)

        # Relational reasoning through ISAB blocks
        for isab in self.isab_blocks:
            x = isab(x)  # (batch, n_pairs, hidden_dim)

        # Pool to single vector per set
        x = self.pma(x)  # (batch, 1, hidden_dim)

        # Remove sequence dimension
        x = tf.squeeze(x, axis=1)  # (batch, hidden_dim)

        # Project to embedding space
        embedding = self.output_mlp(x)  # (batch, embedding_dim)

        return embedding

    def get_config(self):
        return {
            'pair_dim': self.pair_dim,
            'hidden_dim': self.hidden_dim,
            'embedding_dim': self.embedding_dim
        }


def create_set_transformer(calibration_dim, encrypted_dim, embedding_dim=256):
    """
    Factory function to create a SetTransformer for calibration pair processing.

    Args:
        calibration_dim: Dimension of calibration vectors
        encrypted_dim: Dimension of encrypted calibration vectors
        embedding_dim: Output functional embedding dimension

    Returns:
        SetTransformer model
    """
    pair_dim = calibration_dim + encrypted_dim

    model = SetTransformer(
        pair_dim=pair_dim,
        hidden_dim=128,
        embedding_dim=embedding_dim,
        num_inducing_points=32,
        num_isab_blocks=2,
        num_heads=4
    )

    return model
