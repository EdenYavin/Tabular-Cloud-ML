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
    [ Encrypted Sample (q_x) | Encrypted Anchors ({q_i}) | Cloud Predictions | Plaintext Anchors ({p_i}) ]

    1. Encrypted Sample (q_x): Input for classification.
    2. Encrypted Anchors ({q_i}): Input for Deep Sets (to learn context 'c').
    3. Cloud Predictions: Input for classification (optional).
    4. Plaintext Anchors ({p_i}): TARGETS for reconstruction loss (not used for inference).
    """

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata=None):
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)

        if config.cloud_config.names:
            logger.info(f"Cloud models flag is ON, using: {config.cloud_config.names} Models")

        # Ensure we are using 'concat' logic implicitly by passing raw vectors
        logger.info("### USING DEEP SET FEATURES (Raw Encrypted Anchors + Plaintext Targets) ###")

    def _get_features(self, X, embeddings, y, is_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # 1. Prepare Triangulation Samples (Anchors)
        # These are the FIXED plaintext anchors selected from the training set
        triangulation_samples = self._get_triangulation_samples(
            embeddings, y,
            how_to_choose=config.experiment_config.triangulation_choosing,
            n_samples=config.experiment_config.n_triangulation_samples
        )

        # 2. Compute Plaintext Anchor Embeddings (Targets for Reconstruction)
        # We compute this ONCE as they are fixed.
        # Shape: (N_anchors, embedding_dim)
        p_anchors_emb = self.triangulation_embedding.forward(triangulation_samples)

        # Flatten them to append to the end of every sample
        # Shape: (N_anchors * embedding_dim, )
        flat_plaintext_anchors = p_anchors_emb.flatten()

        predictions_for_baseline = np.array(list())
        observations, new_y = [], []

        # Initialize Cloud Context
        cloud = self.cloud_model_manager.__enter__()

        with tqdm(total=len(embeddings), leave=True, position=0, desc="DeepSets Pipeline") as pbar:
            with tf.device(GPU_DEVICE):

                for x, x_emb, label in zip(X, embeddings, y):
                    pbar.update(1)

                    # --- ENCRYPTION & SCALING ---
                    # 1. Encrypt Sample (q_x)
                    x_tag = self.encryptor.encode(x_emb.reshape(1, -1))
                    x_tag = x_tag / config.experiment_config.scaling_factor
                    x_tag = np.clip(x_tag, 0.0, 1.0)

                    # 2. Encrypt Anchors (q_i) using the SAME key
                    # Note: x_tag and y_tag are encrypted with the current random key
                    y_tag = self.encryptor.encode(triangulation_samples)
                    y_tag = y_tag / config.experiment_config.scaling_factor
                    y_tag = np.clip(y_tag, 0.0, 1.0)

                    # --- EMBEDDING ---
                    # 3. Embed Encrypted Anchors (q_i)
                    # Shape: (N_anchors, embedding_dim)
                    y_tag_emb = self.triangulation_embedding.forward(y_tag)
                    flat_encrypted_anchors = y_tag_emb.flatten()

                    # 4. Embed Encrypted Sample (q_x)
                    # Shape: (1, embedding_dim)
                    x_tag_emb = self.triangulation_embedding.forward(np.vstack(x_tag))
                    flat_encrypted_sample = x_tag_emb.flatten()

                    # --- CONSTRUCT OBSERVATION ---
                    # Base: [ q_x | q_i ]
                    observation_parts = [flat_encrypted_sample, flat_encrypted_anchors]

                    # --- CLOUD PREDICTIONS ---
                    if config.cloud_config.names:
                        predictions = []
                        for cloud_model in config.cloud_config.names:
                            # Predict on encrypted sample image
                            pred = cloud.predict(model_name=cloud_model, batch=x_tag)
                            predictions.append(pred.flatten())

                        # Add cloud predictions to parts
                        # Structure: [ q_x | q_i | cloud ]
                        observation_parts.append(np.hstack(predictions))

                    # --- APPEND PLAINTEXT TARGETS ---
                    # This is crucial for the reconstruction loss
                    # Structure: [ q_x | q_i | cloud | p_i (targets) ]
                    observation_parts.append(flat_plaintext_anchors)

                    # Final stack
                    final_vector = np.hstack(observation_parts)

                    # Store
                    # Handle horizontal cloud appending logic if strictly needed,
                    # but typically Deep Sets handles one vector per sample.
                    observations.append(final_vector)
                    new_y.append(label)

                    # Switch Key for next sample
                    if config.encoder_config.rotating_key:
                        self.encryptor.switch_key()

        cloud.__exit__(None, None, None)
        return np.vstack(observations), np.vstack(new_y), predictions_for_baseline