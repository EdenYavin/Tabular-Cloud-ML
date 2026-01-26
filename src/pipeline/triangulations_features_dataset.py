from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE, KEY_ENCODER_MODEL_PATH
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from src.utils.traingulations import TriangulationTransformer
from src.utils.helpers import generate_calibration_vectors
from loguru import logger


class TriangulationFeatureEngineering(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata=None):
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)

        # Load key encoder model if enabled
        self.key_encoder_model = None
        use_key_encoder = getattr(config, 'experiment_use_key_encoder', False)
        if use_key_encoder:
            try:
                if Path(KEY_ENCODER_MODEL_PATH).exists():
                    self.key_encoder_model = tf.keras.models.load_model(KEY_ENCODER_MODEL_PATH)
                    logger.info(f"Loaded key encoder model from {KEY_ENCODER_MODEL_PATH}")
                else:
                    logger.warning(f"Key encoder model not found at {KEY_ENCODER_MODEL_PATH}, skipping")
            except Exception as e:
                logger.error(f"Failed to load key encoder model: {e}")

        if config.cloud_config.names:
            logger.info(f"Cloud models flag is ON, using: {config.cloud_config.names} Models")

    def _get_features(self, X, embeddings, y, is_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info(
            f"### USING SAE FEATURES TRIANGULATION WITH {config.experiment_config.triangulation_mode} TRIANGULATION MODE"
            f" {'AND CALIBRATION VECTOR' if config.experiment_config.use_calibration_vector else ''}"
            f" {'WITH KEY ENCODER' if config.experiment_config.use_key_encoder else ''} ###")

        # 1. Prepare Triangulation Samples (Anchors)
        triangulation_samples = self._get_triangulation_samples(embeddings, y,
                                                                how_to_choose=config.experiment_config.triangulation_choosing,
                                                                n_samples=config.experiment_config.n_triangulation_samples
                                                                )

        # 2. Prepare Calibration Vectors (Multiple distributions for richer key fingerprint)
        # Using a fixed seed ensures the "noise pattern" is identical for Train and Test sets
        calibration_vectors = generate_calibration_vectors(
            embedding_dim=embeddings.shape[1],
            distributions=config.experiment_config.calibration_distributions,
            seed=42
        )

        predictions_for_baseline = np.array(list())
        observations, new_y = [], []
        cloud = self.cloud_model_manager.__enter__()

        with tqdm(total=len(embeddings), leave=True, position=0, desc="Encrypting, Embedding, Predicting") as pbar:
            with tf.device(GPU_DEVICE):
                logger.debug(f"Running ON GPU device: {GPU_DEVICE}")

                for x, x_emb, label in zip(X, embeddings, y):
                    pbar.update(1)

                    # --- ENCRYPTION & SCALING ---
                    # FIX 2: Apply Scaling to X and Y (same as C) to preserve relative geometry

                    # Encrypt Sample
                    x_tag = self.encryptor.encode(x_emb.reshape(1, -1))
                    x_tag = x_tag / config.experiment_config.scaling_factor
                    x_tag = np.clip(x_tag, 0.0, 1.0)

                    # Encrypt Anchors
                    y_tag = self.encryptor.encode(triangulation_samples)
                    y_tag = y_tag / config.experiment_config.scaling_factor
                    y_tag = np.clip(y_tag, 0.0, 1.0)

                    # --- EMBEDDING ---
                    y_tag_emb = self.triangulation_embedding.forward(y_tag)
                    x_tag_emb = self.triangulation_embedding.forward(np.vstack(x_tag))

                    # --- FIX 3: LOGIC FOR DIFF / COS / CONCAT ---
                    if config.experiment_config.triangulation_mode == "diff":
                        triangulation_features = TriangulationTransformer.compute_differential(
                            target_embedding=x_tag_emb,
                            anchor_embeddings=y_tag_emb
                        )
                    elif config.experiment_config.triangulation_mode == "cos":
                        triangulation_features = TriangulationTransformer.compute_cosine_distances(
                            target_embedding=x_tag_emb,
                            anchor_embeddings=y_tag_emb
                        )
                    else:
                        # Default: Concat
                        triangulation_features = TriangulationTransformer.compute_concatenation(
                            target_embedding=x_tag_emb,
                            anchor_embeddings=y_tag_emb
                        )
                    # ------------------------------------------------

                    # Construct Observation
                    if config.experiment_config.use_embedding:
                        # Stack raw 'x' (the sample) + the triangulation features
                        observation = np.hstack([x, triangulation_features])
                    else:
                        observation = np.hstack([triangulation_features])

                    # --- CALIBRATION VECTORS (Multi-distribution key fingerprint) ---
                    if config.experiment_config.use_calibration_vector:
                        calib_embeddings = []
                        for calib_vec in calibration_vectors:
                            c_tag = self.encryptor.encode(calib_vec)
                            c_tag = c_tag / config.experiment_config.scaling_factor
                            c_tag = np.clip(c_tag, 0.0, 1.0)

                            # Use key encoder if available, otherwise use triangulation embedding
                            if self.key_encoder_model is not None:
                                # Key encoder expects batch dimension
                                c_tag_batched = np.expand_dims(c_tag, axis=0) if c_tag.ndim == 2 else c_tag
                                c_tag_emb = self.key_encoder_model(c_tag_batched, training=False).numpy()
                                calib_embeddings.append(c_tag_emb.flatten())
                            else:
                                c_tag_emb = self.triangulation_embedding.forward(c_tag)
                                calib_embeddings.append(c_tag_emb.flatten())

                        # Concatenate all calibration embeddings into a single fingerprint
                        calibration_fingerprint = np.hstack(calib_embeddings)
                        observation = np.hstack([observation, calibration_fingerprint])

                    # --- CLOUD PREDICTIONS ---
                    if config.cloud_config.names:
                        predictions = []
                        for cloud_model in config.cloud_config.names:
                            # Note: We use x_tag (scaled/clipped) which is safe for the cloud model
                            predictions.append(cloud.predict(model_name=cloud_model, batch=x_tag))

                        predictions = [p.flatten() for p in predictions]

                        if config.cloud_config.horizontal_append:
                            observation = np.hstack([observation, np.hstack(predictions)])
                            observations.append(observation)
                            new_y.append(label)
                        else:
                            for p in predictions:
                                observations.append(np.hstack([observation, p]))
                                new_y.append(label)
                        del predictions

                    else:
                        observations.append(np.hstack(observation))
                        new_y.append(label)

                    if config.encoder_config.rotating_key:
                        self.encryptor.switch_key()

                del x_tag, x_tag_emb, y_tag, y_tag_emb

        cloud.__exit__(None, None, None)
        return np.vstack(observations), np.vstack(new_y), predictions_for_baseline