"""
OTP (One-Time Pad) Feature Engineering Pipeline.

For every raw sparse-autoencoder vector x:
  1. Sample a fresh uniform random OTP vector (same dimension as x).
  2. Additively mix:  x_mixed = clip((x + otp) / 2, 0, 1)
  3. Zero-pad x_mixed into the cloud model's expected image shape using
     helpers.expand_matrix_to_img_size.
  4. Query the cloud model and collect its output vector.
  5. That cloud-response vector becomes the IIM feature for this sample.

The class inherits `create()` from FeatureEngineeringPipeline so dataset
caching, train/test splitting, embedding-baseline construction, and
IIMDataset packaging are all handled automatically.
"""

import numpy as np
from tqdm import tqdm
from loguru import logger

from src.encryptor.base import BaseEncryptor
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.helpers import expand_matrix_to_img_size
from src.utils.config import config
from src.cloud import CLOUD_MODELS, DEFAULT_CLOUD_OUTPUT_SHAPE


class OTPFeatureEngineering(FeatureEngineeringPipeline):
    """
    Feature engineering pipeline based on OTP (One-Time Pad) masking.

    Each raw sample is additively blended with a fresh random vector
    (the 'OTP') before being sent to the cloud. The cloud's response
    is used as the IIM training feature, making it impossible for the
    cloud to learn the original sample distribution.

    Args:
        dataset_name:     Name of the dataset (used for caching paths).
        encryptor:        Encryptor model instance (unused in OTP path,
                          kept for interface compatibility with base class).
        embeddings_model: Embedding model instance (used by base class
                          to compute the embedding baseline).
        metadata:         Optional raw dataset metadata dict.
        seed:             RNG seed for reproducibility. Each sample gets
                          a deterministic-but-unique OTP derived from
                          this seed + sample index.
    """

    def __init__(
        self,
        dataset_name: str,
        encryptor: BaseEncryptor,
        embeddings_model,
        metadata=None,
        seed: int = 42,
    ):
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)
        self.seed = seed

        if config.cloud_config.names:
            logger.info(
                f"[OTPFeatureEngineering] Cloud models: {config.cloud_config.names}"
            )
        else:
            logger.warning(
                "[OTPFeatureEngineering] No cloud models configured. "
                "OTP features will be empty — the IIM will have no signal."
            )

        # Determine the target image shape from the first configured cloud model
        if config.cloud_config.names:
            self._cloud_input_shape = CLOUD_MODELS[
                config.cloud_config.names[0]
            ].input_shape
        else:
            self._cloud_input_shape = DEFAULT_CLOUD_OUTPUT_SHAPE

        logger.info(
            f"[OTPFeatureEngineering] Cloud input shape: {self._cloud_input_shape}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_otp(self, dim: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a uniform random OTP vector in [0, 1]^dim."""
        return rng.uniform(0.0, 1.0, size=(dim,))

    def _mix(self, x: np.ndarray, otp: np.ndarray) -> np.ndarray:
        """
        Additively blend sample x with otp and clip to [0, 1].

        Formula:  x_mixed = clip((x + otp) / 2, 0, 1)

        This keeps the result in valid image-pixel range regardless of x's
        original distribution.
        """
        return np.clip((x + otp) / 2.0, 0.0, 1.0)

    def _to_cloud_image(self, x_mixed_flat: np.ndarray) -> np.ndarray:
        """
        Zero-pad a flat 1-D mixed vector into the cloud model's image tensor.

        The cloud model expects shape (1, H, W, C).  We reshape x_mixed_flat
        into a 2-D matrix (1, dim) and pass it through expand_matrix_to_img_size
        which distributes the values into (H*W, C) space and pads with zeros.

        Args:
            x_mixed_flat: 1-D array of shape (d,)

        Returns:
            4-D numpy array of shape (1, H, W, 3) ready for the cloud model.
        """
        dim = x_mixed_flat.shape[0]

        # expand_matrix_to_img_size expects a 2-D matrix (rows, cols)
        # We treat the vector as a single row: (1, dim)
        matrix_2d = x_mixed_flat.reshape(1, dim)

        # Target shape for the 2-D pad is (H, W) taken from the cloud input
        # _cloud_input_shape is the full input_shape tuple, e.g. (224, 224, 3)
        H, W = self._cloud_input_shape[0], self._cloud_input_shape[1]
        target_2d = (H, W)  # expand_matrix handles the 3 channel stacking

        # Clamp if vector is larger than the target (shouldn't happen for
        # typical tabular datasets, but guard just in case)
        if dim > target_2d[0] * target_2d[1]:
            logger.warning(
                f"[OTPFeatureEngineering] Sample dim {dim} exceeds target "
                f"{target_2d} — truncating before padding."
            )
            matrix_2d = matrix_2d[:, : target_2d[0] * target_2d[1]]

        image_tensor = expand_matrix_to_img_size(matrix_2d, target_2d)
        # image_tensor shape: (1, H, W, 3)
        return image_tensor

    # ------------------------------------------------------------------
    # Core feature extraction (called by base.create())
    # ------------------------------------------------------------------

    def _get_features(
        self,
        X: np.ndarray,
        embeddings: np.ndarray,
        y: np.ndarray,
        is_test: bool,
    ):
        """
        Build OTP-masked cloud features for every sample in X.

        For each sample x_i:
          1. Generate a fresh OTP vector.
          2. Mix: x_mixed = clip((x_i + otp) / 2, 0, 1).
          3. Zero-pad x_mixed → cloud image tensor.
          4. Query cloud → collect prediction vector as IIM feature.

        Args:
            X:          Raw feature matrix, shape (N, d).
            embeddings: Pre-computed embeddings (not used for OTP features,
                        but available if future variants need them).
            y:          One-hot label matrix, shape (N, n_classes).
            is_test:    Whether this is the test split (for logging).

        Returns:
            Tuple of (observations, y, empty_predictions_array)
            matching the FeatureEngineeringPipeline._get_features contract.
        """
        split_label = "TEST" if is_test else "TRAIN"
        logger.info(
            f"[OTPFeatureEngineering] Building OTP features for {split_label} "
            f"split — {len(X)} samples, dim={X.shape[1]}"
        )

        rng = np.random.default_rng(seed=self.seed)
        observations = []
        predictions_for_baseline = np.array([])

        cloud = self.cloud_model_manager.__enter__()

        with tqdm(
            total=len(X),
            desc=f"OTP Masking + Cloud [{split_label}]",
            unit="sample",
            leave=True,
            position=0,
        ) as pbar:
            for idx, x in enumerate(X):
                pbar.update(1)

                # 1. Generate fresh OTP
                otp = self._generate_otp(x.shape[0], rng)

                # 2. Mix
                x_mixed = self._mix(x, otp)

                # 3. Zero-pad → image tensor (1, H, W, 3)
                image_tensor = self._to_cloud_image(x_mixed)

                # 4. Query cloud models and concatenate predictions
                sample_features = []
                for cloud_model_name in config.cloud_config.names:
                    pred = cloud.predict(
                        model_name=cloud_model_name, batch=image_tensor
                    )
                    sample_features.append(pred.flatten())

                if sample_features:
                    observation = np.hstack(sample_features)
                else:
                    # Fallback if no cloud model configured: use x_mixed itself
                    if idx == 0:
                        logger.warning(
                            f"[OTPFeatureEngineering] No cloud predictions — "
                            f"falling back to raw mixed vectors for all samples."
                        )
                    observation = x_mixed

                observations.append(observation)

        cloud.__exit__(None, None, None)

        logger.info(
            f"[OTPFeatureEngineering] {split_label} features built — "
            f"shape: {np.vstack(observations).shape}"
        )
        return np.vstack(observations), y, predictions_for_baseline
