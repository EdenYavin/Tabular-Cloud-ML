"""
Synthetic Keyspace Data Generator for Offline Meta-Learning.

Generates (Key, Calibration_Pairs) tuples for training the Key Encoder.
Since encryption keys are random initializations, we can generate infinite data.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict
from loguru import logger
from tqdm import tqdm

from src.encryptor.base import BaseEncryptor
from src.utils.helpers import generate_calibration_vectors


class KeyspaceDataGenerator:
    """
    Generates synthetic training data for learning the encryption key distribution.

    For each synthetic key:
    1. Initialize encryptor with random seed
    2. Generate N calibration vectors
    3. Encrypt them to get input-output pairs
    4. Store pairs with unique key ID
    """

    def __init__(self,
                 encryptor_class: type,
                 dataset_name: str,
                 embedding_dim: int,
                 num_calibration_pairs: int = 50,
                 calibration_distributions: List[str] = None):
        """
        Args:
            encryptor_class: Class of encryptor to use (e.g., DenseEncryptor)
            dataset_name: Name of dataset (for encryptor initialization)
            embedding_dim: Dimension of embeddings to encrypt
            num_calibration_pairs: Number of calibration pairs per key
            calibration_distributions: List of distributions for calibration vectors
        """
        self.encryptor_class = encryptor_class
        self.dataset_name = dataset_name
        self.embedding_dim = embedding_dim
        self.num_calibration_pairs = num_calibration_pairs

        # Default to uniform and gaussian distributions
        self.calibration_distributions = calibration_distributions or ["uniform", "gaussian"]

    def generate_single_key_data(self,
                                  key_seed: int,
                                  separate_contexts: bool = True) -> Dict:
        """
        Generate data for a single encryption key.

        Args:
            key_seed: Random seed for key initialization
            separate_contexts: If True, generate two separate context sets A and B
                             (for contrastive learning). If False, generate single context.

        Returns:
            Dictionary containing:
            - key_id: Unique identifier for this key
            - context_A: List of (cal_vec, encrypted) tuples
            - context_B: List of (cal_vec, encrypted) tuples (if separate_contexts=True)
            - encrypted_dim: Dimension of encrypted outputs
        """
        # Initialize encryptor with specific seed
        encryptor = self.encryptor_class(dataset_name=self.dataset_name)

        # Generate calibration vectors
        num_per_context = self.num_calibration_pairs // 2 if separate_contexts else self.num_calibration_pairs

        # Context A
        calibration_A = self._generate_random_calibrations(num_per_context, seed=key_seed * 1000)
        encrypted_A = []

        for cal_vec in calibration_A:
            # Build generator with specific seed for this key
            if encryptor.model is None:
                # build_generator signature varies by encryptor type
                # Some encryptors (DCEncryptor) support seed, others (DenseEncryptor) don't
                try:
                    encryptor.model = encryptor.build_generator(
                        input_shape=(1, self.embedding_dim),
                        output_shape=(1, self.embedding_dim),
                        seed=key_seed
                    )
                except TypeError:
                    # Fallback for encryptors that don't support seed parameter
                    encryptor.model = encryptor.build_generator(
                        input_shape=(1, self.embedding_dim),
                        output_shape=(1, self.embedding_dim)
                    )

            # Add batch dimension if needed
            cal_vec_batched = np.expand_dims(cal_vec, axis=0) if cal_vec.ndim == 2 else cal_vec
            enc_output = encryptor.model(cal_vec_batched).numpy()
            encrypted_A.append(enc_output)

        context_A_pairs = list(zip(calibration_A, encrypted_A))

        result = {
            'key_id': key_seed,
            'context_A': context_A_pairs,
            'encrypted_dim': encrypted_A[0].shape[1]
        }

        # Context B (different calibration vectors, same key)
        if separate_contexts:
            calibration_B = self._generate_random_calibrations(num_per_context, seed=key_seed * 1000 + 1)
            encrypted_B = []

            for cal_vec in calibration_B:
                # Add batch dimension if needed
                cal_vec_batched = np.expand_dims(cal_vec, axis=0) if cal_vec.ndim == 2 else cal_vec
                enc_output = encryptor.model(cal_vec_batched).numpy()
                encrypted_B.append(enc_output)

            context_B_pairs = list(zip(calibration_B, encrypted_B))
            result['context_B'] = context_B_pairs

        # Clean up
        del encryptor

        return result

    def generate_batch(self,
                       num_keys: int,
                       starting_seed: int = 0,
                       separate_contexts: bool = True) -> Dict:
        """
        Generate data for multiple keys.

        Args:
            num_keys: Number of different keys to generate
            starting_seed: Starting seed value
            separate_contexts: Whether to generate A/B contexts for contrastive learning

        Returns:
            Dictionary with:
            - keys: List of key IDs
            - contexts_A: List of context A for each key
            - contexts_B: List of context B for each key (if separate_contexts)
            - calibration_dim: Dimension of calibration vectors
            - encrypted_dim: Dimension of encrypted outputs
        """
        logger.info(f"Generating keyspace data for {num_keys} keys...")

        keys = []
        contexts_A = []
        contexts_B = [] if separate_contexts else None

        for i in tqdm(range(num_keys), desc="Generating keys"):
            seed = starting_seed + i
            key_data = self.generate_single_key_data(seed, separate_contexts)

            keys.append(key_data['key_id'])
            contexts_A.append(key_data['context_A'])

            if separate_contexts:
                contexts_B.append(key_data['context_B'])

        result = {
            'keys': keys,
            'contexts_A': contexts_A,
            'calibration_dim': self.embedding_dim,
            'encrypted_dim': key_data['encrypted_dim']
        }

        if separate_contexts:
            result['contexts_B'] = contexts_B

        logger.info(f"Generated {num_keys} synthetic keys successfully")

        return result

    def _generate_random_calibrations(self,
                                      num_vectors: int,
                                      seed: int) -> List[np.ndarray]:
        """
        Generate random calibration vectors from different distributions.

        Args:
            num_vectors: Number of vectors to generate
            seed: Random seed for reproducibility

        Returns:
            List of calibration vectors (each shaped (1, embedding_dim))
        """
        rng = np.random.default_rng(seed=seed)
        calibrations = []

        # Cycle through distributions
        for i in range(num_vectors):
            dist_type = self.calibration_distributions[i % len(self.calibration_distributions)]

            if dist_type == "uniform":
                vec = rng.uniform(0, 1, size=(1, self.embedding_dim))
            elif dist_type in ["gaussian", "normal"]:
                vec = rng.normal(loc=0.5, scale=0.2, size=(1, self.embedding_dim))
                vec = np.clip(vec, 0, 1)
            else:
                # Default uniform
                vec = rng.uniform(0, 1, size=(1, self.embedding_dim))

            calibrations.append(vec.astype(np.float32))

        return calibrations

    @staticmethod
    def format_for_set_transformer(contexts: List[List[Tuple]]) -> np.ndarray:
        """
        Convert list of calibration pair contexts to SetTransformer input format.

        Args:
            contexts: List of contexts, where each context is a list of (cal_vec, enc_vec) tuples

        Returns:
            Array of shape (num_contexts, num_pairs, pair_dim)
            where pair_dim = cal_dim + enc_dim
        """
        formatted = []

        for context in contexts:
            # Each context is a list of (cal, enc) pairs
            pairs = []
            for cal_vec, enc_vec in context:
                # Concatenate calibration and encrypted vectors
                pair = np.concatenate([cal_vec.flatten(), enc_vec.flatten()])
                pairs.append(pair)

            formatted.append(np.array(pairs))

        # Stack to (num_contexts, num_pairs, pair_dim)
        return np.array(formatted, dtype=np.float32)


def generate_keyspace_dataset(encryptor_class: type,
                               dataset_name: str,
                               embedding_dim: int,
                               num_keys: int = 1000,
                               num_calibration_pairs: int = 50,
                               save_path: str = None) -> Dict:
    """
    Convenience function to generate a full keyspace dataset.

    Args:
        encryptor_class: Encryptor class to use
        dataset_name: Dataset name for encryptor
        embedding_dim: Embedding dimension
        num_keys: Number of unique keys to generate
        num_calibration_pairs: Calibration pairs per key
        save_path: Optional path to save the dataset

    Returns:
        Dictionary with formatted data ready for training
    """
    generator = KeyspaceDataGenerator(
        encryptor_class=encryptor_class,
        dataset_name=dataset_name,
        embedding_dim=embedding_dim,
        num_calibration_pairs=num_calibration_pairs
    )

    # Generate batch with separate contexts for contrastive learning
    batch = generator.generate_batch(num_keys=num_keys, separate_contexts=True)

    # Format for SetTransformer
    X_A = KeyspaceDataGenerator.format_for_set_transformer(batch['contexts_A'])
    X_B = KeyspaceDataGenerator.format_for_set_transformer(batch['contexts_B'])

    dataset = {
        'X_A': X_A,
        'X_B': X_B,
        'key_ids': np.array(batch['keys']),
        'calibration_dim': batch['calibration_dim'],
        'encrypted_dim': batch['encrypted_dim']
    }

    if save_path:
        np.savez(save_path, **dataset)
        logger.info(f"Saved keyspace dataset to {save_path}")

    return dataset
