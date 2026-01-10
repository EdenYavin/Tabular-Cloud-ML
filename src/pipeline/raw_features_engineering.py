
from tqdm import tqdm
import numpy as np
from loguru import logger
from src.embeddings.model import tf

from src.encryptor.base import BaseEncryptor
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.constansts import GPU_DEVICE
from src.utils.config import config
from src.utils.traingulations import TriangulationTransformer

class RawFeaturesEngineering(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata=None):
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)
        if config.cloud_config.names:
            logger.info(f"Cloud models flag is ON, using: {config.cloud_config.names} Models")

    def _get_features(self, X, embeddings, y, is_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get features from embeddings for training or testing.

        This function processes embeddings through multiple cloud models, performs various
        data augmentations for training if necessary, and generates processed observations
        and corresponding labels as output. Predictions for the baseline model can also be
        retrieved if applicable.

        Parameters:
        embeddings : ndarray
            A numpy array containing the embeddings to be processed.
        y : ndarray
            A numpy array containing the corresponding labels for the embeddings.
        is_test : bool
            A flag that determines if the data is being processed in testing mode.

        Returns:
        tuple
            A tuple containing the following elements:
            - A numpy array of processed observations, combining features and optionally
              embeddings from triangulation.
            - A numpy array of expanded labels corresponding to the processed observations.
            - A numpy array containing predictions for the baseline model if applicable.

        Raises:
        None
        """
        logger.info(f"### USING RAW FEATURES TRIANGULATION WITH {config.experiment_config.triangulation_mode} TRIANGULATION MODE "
                   f" {'AND CALIBRATION VECTOR' if config.experiment_config.use_calibration_vector else ''} ###")
        # Add the new triangulation samples' embedding as well
        triangulation_samples = self._get_triangulation_samples(X, y,
         how_to_choose=config.experiment_config.triangulation_choosing,
         n_samples=config.experiment_config.n_triangulation_samples
        )

        #Define the Calibration Vector (C) ###
        # We create a vector of ones with the same shape as a single embedding input (latent dim)
        # This acts as our "Perfect Compass" before distortion.
        rng = np.random.default_rng(seed=42)
        calibration_vector = rng.normal(loc=0.5, scale=0.1, size=(1, X.shape[1]))
        calibration_vector = np.clip(calibration_vector, 0, 1)  # Ensure input is valid


        # For test data we won't duplicate but encrypt it only once
        predictions_for_baseline = np.array(list())  # Will be used for the baseline, TODO: If needed use it
        cloud = self.cloud_model_manager.__enter__()

        observations = []

        with tqdm(total=len(X), leave=True, position=0, desc="Encrypting, Embedding, Predicting") as pbar:
            with tf.device(GPU_DEVICE):  # Run the models on the GPU
                logger.debug(f"Running ON GPU device: {GPU_DEVICE}")

                for x, label in zip(X, y):
                    pbar.update(1)

                    # Triangulation features vector = X', Y_1', Y_2',...
                    # --- 1. Encrypt Sample (X) ---
                    x_tag = self.encryptor.encode(x.reshape(1, -1))
                    x_tag = x_tag / config.experiment_config.scaling_factor  # Preserves relative signal
                    x_tag = np.clip(x_tag, 0.0, 1.0)  # Prevents crash

                    # --- 2 Encrypt Anchors (Y) ---
                    y_tag = self.encryptor.encode(triangulation_samples)
                    y_tag = y_tag / config.experiment_config.scaling_factor  # Apply SAME scaling
                    y_tag = np.clip(y_tag, 0.0, 1.0)  # Prevents crash

                    # 3. Embed the encryption
                    y_tag_emb = self.triangulation_embedding.forward(y_tag)

                    # Embedding for triangulation using image embedding, those are the new features
                    x_tag_emb = self.triangulation_embedding.forward(np.vstack(x_tag))


                    # 3. Apply Triangulation Strategy based on Config
                    if config.experiment_config.triangulation_mode == "diff":
                        # New Strategy: Relative Differentials
                        triangulation_features = TriangulationTransformer.compute_differential(
                            target_embedding=x_tag_emb,
                            anchor_embeddings=y_tag_emb
                        )
                    # 3. Apply Triangulation Strategy based on Config
                    elif config.experiment_config.triangulation_mode == "cos":
                        # New Strategy: Relative Differentials
                        triangulation_features = TriangulationTransformer.compute_cosine_distances(
                            target_embedding=x_tag_emb,
                            anchor_embeddings=y_tag_emb
                        )

                    else:
                        # Default Strategy: Concatenation (Original)
                        triangulation_features = TriangulationTransformer.compute_concatenation(
                            target_embedding=x_tag_emb,
                            anchor_embeddings=y_tag_emb
                        )

                    # If 'use_raw' is enabled, we prepend the raw data 'x' (usually False in TEP-KD)
                    # Note: Original code had logic for `config.experiment_config.use_embedding`
                    # which effectively meant "append x".
                    if config.experiment_config.use_embedding:  # This naming in original code implied "use raw x + embedding"
                        observation = np.hstack([x, triangulation_features])
                    else:
                        observation = np.hstack([triangulation_features])

                    if config.experiment_config.use_calibration_vector:
                        # ### Encrypt & Embed the Calibration Vector ###
                        # We encrypt C using the CURRENT key (same as x_tag and y_tag)
                        # The IIM will see how this 'all-ones' vector got twisted.
                        # -----------------------------------------------------
                        c_tag = self.encryptor.encode(calibration_vector)
                        # Step 1: Scale down to preserve the "peaks"
                        c_tag = c_tag / config.experiment_config.scaling_factor
                        # Step 2: Clip only as a final safety net for extreme outliers (infinity/NaN/huge spikes)
                        c_tag = np.clip(c_tag, 0.0, 1.0)

                        c_tag_emb = self.triangulation_embedding.forward(c_tag)
                        observation = np.hstack([observation, c_tag_emb.flatten()])


                    # Add the cloud predictions as features if needed:
                    if config.cloud_config.names:
                        predictions = []
                        for cloud_model in config.cloud_config.names:
                            predictions.append(cloud.predict(model_name=cloud_model, batch=x_tag))

                        # Flatten prediction to be stacked correctly
                        predictions = [p.flatten() for p in predictions]

                        if config.cloud_config.horizontal_append:
                            observation = np.hstack([observation, np.hstack(predictions)])
                            observations.append(observation)


                        del predictions

                    else:
                        # No cloud models need to be used, just use the features up until now
                        observations.append(np.hstack(observation))


                    if config.encoder_config.rotating_key:
                        # Switch key for the next example
                        self.encryptor.switch_key()

                del x_tag, x_tag_emb, y_tag, y_tag_emb

        cloud.__exit__(None, None, None)
        return np.vstack(observations), y, predictions_for_baseline