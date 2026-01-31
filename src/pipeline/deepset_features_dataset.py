from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from loguru import logger


class DeepSetFeatureEngineering(FeatureEngineeringPipeline):
    """
    Produces dataset specifically for Deep Sets + Reconstruction IIM.

    Output Vector Structure (NEW - DeepSets on Anchor Pairs):
    [ Plaintext X Emb (p_x) | Plaintext Anchors Emb ({p_i}) | Encrypted Anchors Emb ({q_i}) | Cloud Predictions | Encrypted X Emb (q_x_target) ]

    Architecture:
    - Encoder (T) input: Anchor pairs (p_i, q_i) for each anchor i, plus p_x
    - Decoder target: Reconstruct q_x (encrypted X embedding)

    All embeddings are DINO/CLIP embeddings (embedding_dim = 768 or 512).
    """

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata=None):
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)

        if config.cloud_config.names:
            logger.info(f"Cloud models flag is ON, using: {config.cloud_config.names} Models")

        # New architecture: DeepSets on anchor pairs (p_i, q_i), reconstruct q_x
        logger.info("### USING DEEP SET FEATURES (Anchor Pairs + q_x Reconstruction Target) ###")

    def _get_features(self, X, embeddings, y, is_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # 1. Prepare Triangulation Samples (Anchors) - raw tabular vectors
        triangulation_samples = self._get_triangulation_samples(
            embeddings, y,
            how_to_choose=config.experiment_config.triangulation_choosing,
            n_samples=config.experiment_config.n_triangulation_samples
        )

        # 2. Compute Plaintext Anchor Embeddings (p_i) BEFORE encryption
        # Shape: (N_anchors, embedding_dim)
        p_i_emb = self.triangulation_embedding.forward(triangulation_samples)
        flat_plaintext_anchor_emb = p_i_emb.flatten()

        predictions_for_baseline = np.array(list())
        observations, new_y = [], []

        # Initialize Cloud Context
        cloud = self.cloud_model_manager.__enter__()

        with tqdm(total=len(embeddings), leave=True, position=0, desc="DeepSets Pipeline") as pbar:
            with tf.device(GPU_DEVICE):

                for x, x_emb, label in zip(X, embeddings, y):
                    pbar.update(1)

                    # --- PLAINTEXT EMBEDDING (BEFORE ENCRYPTION) ---
                    # 1. Compute Plaintext X Embedding (p_x)
                    # Shape: (1, embedding_dim)
                    p_x_emb = self.triangulation_embedding.forward(x_emb.reshape(1, -1))
                    flat_plaintext_sample_emb = p_x_emb.flatten()

                    # --- ENCRYPTION & SCALING ---
                    # 2. Encrypt Sample (for q_x target)
                    x_tag = self.encryptor.encode(x_emb.reshape(1, -1))
                    x_tag = x_tag / config.experiment_config.scaling_factor
                    x_tag = np.clip(x_tag, 0.0, 1.0)

                    # 3. Encrypt Anchors (q_i) using the SAME key
                    y_tag = self.encryptor.encode(triangulation_samples)
                    y_tag = y_tag / config.experiment_config.scaling_factor
                    y_tag = np.clip(y_tag, 0.0, 1.0)

                    # --- EMBEDDING OF ENCRYPTED DATA ---
                    # 4. Embed Encrypted Anchors (q_i)
                    # Shape: (N_anchors, embedding_dim)
                    q_i_emb = self.triangulation_embedding.forward(y_tag)
                    flat_encrypted_anchor_emb = q_i_emb.flatten()

                    # 5. Embed Encrypted Sample (q_x) - THIS IS THE TARGET
                    # Shape: (1, embedding_dim)
                    q_x_emb = self.triangulation_embedding.forward(np.vstack(x_tag))
                    flat_encrypted_sample_emb = q_x_emb.flatten()

                    # --- CONSTRUCT OBSERVATION ---
                    # NEW Structure: [ p_x | p_i | q_i | cloud | q_x_target ]
                    observation_parts = [
                        flat_plaintext_sample_emb,      # p_x: plaintext X embedding
                        flat_plaintext_anchor_emb,       # p_i: plaintext anchor embeddings
                        flat_encrypted_anchor_emb,       # q_i: encrypted anchor embeddings
                    ]

                    # --- CLOUD PREDICTIONS ---
                    if config.cloud_config.names:
                        predictions = []
                        for cloud_model in config.cloud_config.names:
                            # Predict on encrypted sample image
                            pred = cloud.predict(model_name=cloud_model, batch=x_tag)
                            predictions.append(pred.flatten())

                        # Add cloud predictions to parts
                        observation_parts.append(np.hstack(predictions))

                    # --- APPEND q_x TARGET ---
                    # q_x (encrypted X embedding) is the reconstruction target
                    observation_parts.append(flat_encrypted_sample_emb)

                    # Final stack
                    final_vector = np.hstack(observation_parts)

                    observations.append(final_vector)
                    new_y.append(label)

                    # Switch Key for next sample
                    if config.encoder_config.rotating_key:
                        self.encryptor.switch_key()

        cloud.__exit__(None, None, None)
        return np.vstack(observations), np.vstack(new_y), predictions_for_baseline