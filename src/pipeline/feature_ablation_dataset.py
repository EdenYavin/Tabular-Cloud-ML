"""
Feature Ablation Pipeline for Frozen T Network Studies.

Generates 4 different feature combinations for ablation experiments:
- baseline_no_cloud: [p_x, q_x, T_context] - 960 dims (64 + 768 + 128)
- no_raw_embedding: [q_x, T_context] - 896 dims (768 + 128)
- full_features: [p_x, q_x, T_context, cloud] - 960+cloud dims
- cloud_no_raw: [q_x, T_context, cloud] - 896+cloud dims

Where:
- p_x: Raw sparse autoencoder sample embedding (64 dims)
- q_x: Encrypted sample → DINO/CLIP embedding (768 dims for DINO, 512 for CLIP)
- T_context: Frozen T network encoder output (128 dims)
- cloud: Cloud model predictions (1000 * num_cloud_models)
"""

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE, FEATURE_COMBINATIONS
from src.pipeline.base import FeatureEngineeringPipeline
from src.pipeline.frozen_t_context_extractor import FrozenTContextExtractor
from src.utils.config import config
from src.utils.helpers import get_t_network_model_path
from loguru import logger


class FeatureAblationPipeline(FeatureEngineeringPipeline):
    """
    Generates 4 feature combinations for frozen T network ablation study.

    This pipeline extracts features from a frozen, pretrained T network and combines
    them with raw embeddings and/or cloud predictions based on the selected combination.

    The feature_combination is specified via config.experiment_config.feature_combination
    and must be one of: baseline_no_cloud, no_raw_embedding, full_features, or cloud_no_raw.
    """

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata=None):
        """
        Initialize Feature Ablation Pipeline.

        Args:
            dataset_name: Name of dataset being processed
            encryptor: Encryptor model for generating encrypted embeddings
            embeddings_model: Embedding model for raw features
            metadata: Optional dataset metadata

        Raises:
            ValueError: If feature_combination not specified in config
            FileNotFoundError: If pretrained T network path not found
        """
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)

        # Validate feature combination is specified
        if not config.experiment_config.feature_combination:
            raise ValueError(
                "feature_combination must be specified in config. "
                "Use --feature-combination baseline_no_cloud|no_raw_embedding|full_features|cloud_no_raw"
            )

        self.feature_combination = config.experiment_config.feature_combination

        # Validate it's a valid combination
        try:
            FEATURE_COMBINATIONS(self.feature_combination)
        except ValueError:
            raise ValueError(
                f"Invalid feature_combination: {self.feature_combination}. "
                f"Must be one of: baseline_no_cloud, no_raw_embedding, full_features, cloud_no_raw"
            )

        logger.info(
            f"### FEATURE ABLATION PIPELINE - {self.feature_combination.upper()} ###"
        )

        # Determine T network path (use explicit path or generate default)
        if config.experiment_config.pretrained_t_network_path:
            t_network_path = Path(config.experiment_config.pretrained_t_network_path)
            logger.info(f"Using explicit T network path: {t_network_path}")
        else:
            # Generate default path based on dataset and config
            t_network_path = get_t_network_model_path(
                dataset_name=dataset_name,
                ensure_exists=False  # We'll check existence below
            )
            logger.info(f"Using default T network path: {t_network_path}")

        # Verify the T network model exists
        if not t_network_path.exists():
            raise FileNotFoundError(
                f"Pretrained T network not found at: {t_network_path}\n"
                f"Either:\n"
                f"  1. Train a T network first: python main.py --experiment-to-run t_network_training --datasets {dataset_name}\n"
                f"  2. Provide explicit path: --pretrained-t-network /path/to/model.keras"
            )

        # Initialize Frozen T Context Extractor
        # Configuration: n_anchors, raw_dim (sparse AE: 64), emb_dim (DINO: 768, CLIP: 512)
        n_anchors = config.experiment_config.n_triangulation_samples
        raw_dim = 64  # Sparse autoencoder embedding dimension

        # Determine encrypted embedding dimension based on triangulation embedding
        if config.encoder_config.embedding == "dino":
            emb_dim = 768
        elif config.encoder_config.embedding == "clip":
            emb_dim = 512
        else:
            raise ValueError(
                f"Unsupported triangulation embedding: {config.encoder_config.embedding}"
            )

        logger.info(
            f"Initializing FrozenTContextExtractor: "
            f"n_anchors={n_anchors}, raw_dim={raw_dim}, emb_dim={emb_dim}"
        )

        self.t_context_extractor = FrozenTContextExtractor(
            t_network_path=t_network_path,
            n_anchors=n_anchors,
            raw_dim=raw_dim,
            emb_dim=emb_dim
        )

        # Verify T network is frozen
        is_frozen = self.t_context_extractor.verify_frozen()
        if not is_frozen:
            logger.warning("T network has trainable layers - should be fully frozen")

        # Log combination details
        self._log_combination_info()

        # Check if cloud models needed
        if self.feature_combination in ["full_features", "cloud_no_raw"]:
            if not config.cloud_config.names:
                raise ValueError(
                    f"{self.feature_combination} requires cloud models. "
                    f"Specify --use-cloud-models [model1] [model2] ..."
                )
            logger.info(
                f"Cloud models enabled for {self.feature_combination}: "
                f"{config.cloud_config.names}"
            )

    def _log_combination_info(self):
        """Log information about the selected feature combination."""
        combo_info = {
            "baseline_no_cloud": "[p_x(64), q_x(768/512), T_context(128)] → 960/704 dims",
            "no_raw_embedding": "[q_x(768/512), T_context(128)] → 896/640 dims",
            "full_features": "[p_x(64), q_x(768/512), T_context(128), cloud(N)] → 960/704+N dims",
            "cloud_no_raw": "[q_x(768/512), T_context(128), cloud(N)] → 896/640+N dims"
        }

        logger.info(f"Feature combination: {combo_info[self.feature_combination]}")

    def _get_features(self, X, embeddings, y, is_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features based on selected combination.

        For each sample:
        1. Extract p_x (raw sample embedding, 64 dims)
        2. Extract p_i (raw anchor embeddings, n_anchors * 64 dims)
        3. Encrypt sample and anchors with current key
        4. Extract q_x (encrypted sample → DINO/CLIP embedding)
        5. Extract q_i (encrypted anchors → DINO/CLIP embeddings)
        6. Call FrozenTContextExtractor.extract_context(p_x, p_i, q_i) → T_context (128 dims)
        7. Generate cloud predictions if needed (full_features/cloud_no_raw)
        8. Build observation based on feature_combination

        Args:
            X: Raw data samples
            embeddings: Sparse autoencoder embeddings
            y: Labels
            is_test: Whether this is test set

        Returns:
            Tuple of (observations, labels, baseline_predictions)
        """

        # 1. Prepare Triangulation Samples (Anchors) - sparse AE embeddings
        triangulation_samples = self._get_triangulation_samples(
            embeddings, y,
            how_to_choose=config.experiment_config.triangulation_choosing,
            n_samples=config.experiment_config.n_triangulation_samples
        )

        # ================================================================
        # p_i = Sparse autoencoder anchor embeddings (BEFORE encryption)
        # This is the plaintext representation (64-dim embeddings per anchor)
        # ================================================================
        flat_plaintext_anchors = triangulation_samples.flatten()  # Shape: (n_anchors * 64,)

        predictions_for_baseline = np.array(list())
        observations, new_y = [], []

        # Initialize Cloud Context
        cloud = self.cloud_model_manager.__enter__()

        # Batch collection for T_context extraction (more efficient than per-sample)
        batch_p_x = []
        batch_p_i = []
        batch_q_i = []

        with tqdm(total=len(embeddings), leave=True, position=0,
                  desc=f"FeatureAblation-{self.feature_combination}") as pbar:
            with tf.device(GPU_DEVICE):

                for idx, (x, x_emb, label) in enumerate(zip(X, embeddings, y)):
                    pbar.update(1)

                    # ================================================================
                    # p_x = Sparse autoencoder sample embedding (BEFORE encryption)
                    # ================================================================
                    flat_plaintext_sample = x_emb.flatten()  # Shape: (64,)

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
                    # q_x = Encrypted sample → DINO/CLIP embedding
                    # ================================================================
                    q_i_emb = self.triangulation_embedding.forward(y_tag)
                    flat_encrypted_anchor_emb = q_i_emb.flatten()  # Shape: (n_anchors * emb_dim,)

                    q_x_emb = self.triangulation_embedding.forward(x_tag)
                    flat_encrypted_sample_emb = q_x_emb.flatten()  # Shape: (emb_dim,)

                    # ================================================================
                    # T_context = Frozen T network encoder output (128 dims)
                    # Extract context from (p_x, p_i, q_i)
                    # ================================================================
                    T_context = self.t_context_extractor.extract_context(
                        p_x=flat_plaintext_sample.reshape(1, -1),
                        p_i=flat_plaintext_anchors.reshape(1, -1),
                        q_i=flat_encrypted_anchor_emb.reshape(1, -1)
                    )
                    flat_T_context = T_context.flatten()  # Shape: (128,)

                    # Log first few samples for debugging
                    if idx < 3:
                        logger.debug(
                            f"Sample {idx}: p_x={flat_plaintext_sample.shape}, "
                            f"q_x={flat_encrypted_sample_emb.shape}, "
                            f"T_context={flat_T_context.shape}"
                        )

                    # --- CONSTRUCT OBSERVATION BASED ON COMBINATION ---
                    observation_parts = []

                    if self.feature_combination == "baseline_no_cloud":
                        # [p_x, q_x, T_context]
                        observation_parts = [
                            flat_plaintext_sample,       # p_x: 64
                            flat_encrypted_sample_emb,   # q_x: 768/512
                            flat_T_context               # T_context: 128
                        ]

                    elif self.feature_combination == "no_raw_embedding":
                        # [q_x, T_context]
                        observation_parts = [
                            flat_encrypted_sample_emb,   # q_x: 768/512
                            flat_T_context               # T_context: 128
                        ]

                    elif self.feature_combination == "full_features":
                        # [p_x, q_x, T_context, cloud]
                        observation_parts = [
                            flat_plaintext_sample,       # p_x: 64
                            flat_encrypted_sample_emb,   # q_x: 768/512
                            flat_T_context               # T_context: 128
                        ]

                        # Add cloud predictions
                        predictions = []
                        for cloud_model in config.cloud_config.names:
                            pred = cloud.predict(model_name=cloud_model, batch=x_tag)
                            predictions.append(pred.flatten())
                        observation_parts.append(np.hstack(predictions))

                    elif self.feature_combination == "cloud_no_raw":
                        # [q_x, T_context, cloud]
                        observation_parts = [
                            flat_encrypted_sample_emb,   # q_x: 768/512
                            flat_T_context               # T_context: 128
                        ]

                        # Add cloud predictions
                        predictions = []
                        for cloud_model in config.cloud_config.names:
                            pred = cloud.predict(model_name=cloud_model, batch=x_tag)
                            predictions.append(pred.flatten())
                        observation_parts.append(np.hstack(predictions))

                    # Final stack
                    final_vector = np.hstack(observation_parts)

                    observations.append(final_vector)
                    new_y.append(label)

                    # Switch Key for next sample
                    if config.encoder_config.rotating_key:
                        self.encryptor.switch_key()

        cloud.__exit__(None, None, None)

        # Log final observation shape
        observations_array = np.vstack(observations)
        logger.info(
            f"Generated {len(observations)} observations with shape: "
            f"{observations_array.shape} for {self.feature_combination}"
        )

        return observations_array, np.vstack(new_y), predictions_for_baseline
