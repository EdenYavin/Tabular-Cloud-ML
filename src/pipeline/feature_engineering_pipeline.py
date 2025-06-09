
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.cloud import CloudModelManager
from src.domain.dataset import Batch
from src.embeddings import ClipEmbedding
from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from loguru import logger


class DatasetCreation(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 n_pred_vectors, metadata = None):
        super().__init__(dataset_name, encryptor, embeddings_model, n_pred_vectors, metadata)
        self.triangulation_embedding = None
        if config.encoder_config.rotating_key:
            logger.info(f"Triangulation model is on, using {ClipEmbedding.name}")
            self.triangulation_embedding = ClipEmbedding()

        if config.experiment_config.use_preds:
            logger.info(f"Cloud models flag is ON, using: {config.cloud_config.names} Models")

    def _get_features(self, embeddings, y, is_test):
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
        observations = []
        predictions_for_baseline = [] # Will be used for the baseline
        new_y = []

        triangulation_samples = embeddings[:config.experiment_config.n_triangulation_samples]

        # Start processing the data using batches. We will do it for each cloud model
        # separately and use the cloud cache to save processed file to save up memory (no need to
        # load the cloud model) and speed up the runtime.
        batch = Batch(X=embeddings, y=y, size=config.dataset_config.batch_size)
        number_of_new_samples = (
            self.n_pred_vectors if not is_test
            else 1
        )

        with self.cloud_model_manager as cloud:
            # Use context manager to control the Cloud models

            for mini_batch, labels in batch:

                for idx, cloud_model in enumerate(config.cloud_config.names):

                    for _ in range(number_of_new_samples):

                        observation = []

                        if config.experiment_config.use_embedding:
                            observation.append(mini_batch)

                        images = self.encryptor.encode(mini_batch)

                        with tf.device(GPU_DEVICE):  # Run the models on the GPU

                            if config.encoder_config.rotating_key:
                                # embed the encrypted samples
                                x_tag = self.triangulation_embedding.forward(images)
                                observation.append(x_tag)

                                # Add the new triangulation samples' embedding as well:
                                # 1. Encrypt them
                                y_tag = self.encryptor.encode(triangulation_samples)
                                # 2. Embed the encryption
                                y_tag = self.triangulation_embedding(y_tag)
                                observation.append(
                                    np.vstack([np.hstack([x, y_tag.flatten()]) for x in
                                                              x_tag])
                                )  # Triangulation features vector = X', Y_1', Y_2',...

                            if config.experiment_config.use_preds:
                                    predictions = cloud.predict(model_name=cloud_model, batch=images)
                                    observations.append(np.hstack([*observation, predictions]))
                                    predictions_for_baseline.append(predictions)

                            else:
                                observations.append(np.hstack(observation))

                        # Add the labels accordingly
                        new_y.extend(labels)

                    del images, y_tag, x_tag

                if config.encoder_config.rotating_key:
                    # Rotate the key for the next sample to be encoded by a new key
                    self.encryptor.switch_key()

        if len(predictions_for_baseline) > 0:
            predictions_for_baseline = np.vstack(predictions_for_baseline)
        else:
            predictions_for_baseline = np.array(list())

        return np.vstack(observations), np.vstack(new_y), predictions_for_baseline