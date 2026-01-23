"""
Meta-Learning components for learning encryption key distributions.

This module implements Set Transformers and Key Encoders for offline meta-learning
of functional embeddings from encryption keys.
"""

from src.meta_learning.set_transformer import SetTransformer, ISAB, PMA
from src.meta_learning.key_encoder import KeyEncoder, create_key_encoder
from src.meta_learning.keyspace_generator import KeyspaceDataGenerator, generate_keyspace_dataset

__all__ = [
    "SetTransformer",
    "ISAB",
    "PMA",
    "KeyEncoder",
    "create_key_encoder",
    "KeyspaceDataGenerator",
    "generate_keyspace_dataset"
]
