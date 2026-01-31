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

    Output Vector Structure:
    [ p_x | p_i | q_i | cloud | q_x_target ]

    Where:
    - p_x: RAW tabular sample data (before encryption) - shape: (raw_dim,)
    - p_i: RAW tabular anchor data (before encryption) - shape: (n_anchors * raw_dim,)
    - q_i: Encrypted anchors → DINO/CLIP embedding - shape: (n_anchors * emb_dim,)
    - cloud: Cloud model predictions (optional) - shape: (1000 * num_cloud_models,)
    - q_x_target: Encrypted sample → DINO/CLIP embedding (reconstruction target) - shape: (emb_dim,)

    Architecture:
    - Encoder (T) input: Anchor pairs (p_i, q_i) for each anchor i, plus p_x
    - Decoder target: Reconstruct q_x (encrypted X embedding)
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

        # ================================================================
        # p_i = RAW tabular anchor data (BEFORE encryption, no embedding)
        # This is the plaintext representation
        # ================================================================
        flat_plaintext_anchors = triangulation_samples.flatten()  # Shape: (n_anchors * raw_dim,)

        predictions_for_baseline = np.array(list())
        observations, new_y = [], []

        # Initialize Cloud Context
        cloud = self.cloud_model_manager.__enter__()

        with tqdm(total=len(embeddings), leave=True, position=0, desc="DeepSets Pipeline") as pbar:
            with tf.device(GPU_DEVICE):

                for idx, (x, x_emb, label) in enumerate(zip(X, embeddings, y)):
                    pbar.update(1)

                    # ================================================================
                    # p_x = RAW tabular sample data (BEFORE encryption, no embedding)
                    # ================================================================
                    flat_plaintext_sample = x_emb.flatten()  # Shape: (raw_dim,)

                    # --- ENCRYPTION WITH CURRENT KEY ---
                    # 1. Encrypt Sample with current key
                    x_tag = self.encryptor.encode(x_emb.reshape(1, -1))
                    x_tag = x_tag / config.experiment_config.scaling_factor
                    x_tag = np.clip(x_tag, 0.0, 1.0)

                    # 2. Encrypt Anchors with current key
                    y_tag = self.encryptor.encode(triangulation_samples)
                    y_tag = y_tag / config.experiment_config.scaling_factor
                    y_tag = np.clip(y_tag, 0.0, 1.0)

                    # ================================================================
                    # q_i = Encrypted anchors → DINO/CLIP embedding
                    # q_x = Encrypted sample → DINO/CLIP embedding (RECONSTRUCTION TARGET)
                    # ================================================================
                    q_i_emb = self.triangulation_embedding.forward(y_tag)
                    flat_encrypted_anchor_emb = q_i_emb.flatten()  # Shape: (n_anchors * emb_dim,)

                    q_x_emb = self.triangulation_embedding.forward(x_tag)
                    flat_encrypted_sample_emb = q_x_emb.flatten()  # Shape: (emb_dim,)

                    # --- CONSTRUCT OBSERVATION ---
                    # Structure: [ p_x | p_i | q_i | cloud | q_x_target ]
                    # p_x, p_i = RAW tabular data (before encryption)
                    # q_i, q_x = encrypted → embedded vectors
                    observation_parts = [
                        flat_plaintext_sample,           # p_x: raw sample (raw_dim)
                        flat_plaintext_anchors,          # p_i: raw anchors (n_anchors * raw_dim)
                        flat_encrypted_anchor_emb,       # q_i: encrypted anchor embeddings (n_anchors * emb_dim)
                    ]

                    # --- CLOUD PREDICTIONS ---
                    if config.cloud_config.names:
                        predictions = []
                        for cloud_model in config.cloud_config.names:
                            pred = cloud.predict(model_name=cloud_model, batch=x_tag)
                            predictions.append(pred.flatten())
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