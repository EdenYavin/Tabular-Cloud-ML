
from tqdm import tqdm
import numpy as np
from loguru import logger
import tensorflow as tf

from src.encryptor.base import BaseEncryptor
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.constansts import GPU_DEVICE
from src.utils.config import config


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
        # Add the new triangulation samples' embedding as well

        # For test data we won't duplicate but encrypt it only once
        predictions_for_baseline = np.array(list())  # Will be used for the baseline, TODO: If needed use it

        if config.cloud_config.names:
            cloud = self.cloud_model_manager.__enter__()
            observations = []

            with tqdm(total=len(embeddings), leave=True, position=0, desc="Encrypting, Embedding, Predicting") as pbar:
                with tf.device(GPU_DEVICE):  # Run the models on the GPU
                    logger.debug(f"Running ON GPU device: {GPU_DEVICE}")

                    for x, x_emb, label in zip(X, embeddings, y):
                        pbar.update(1)

                        # Triangulation features vector = X', Y_1', Y_2',...
                        x_tag = self.encryptor.encode(x_emb.reshape(1, -1))

                        observation = np.hstack([x])

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

                    del x_tag

            cloud.__exit__(None, None, None)

        else:
            observations = X

        return np.vstack(observations), y, predictions_for_baseline