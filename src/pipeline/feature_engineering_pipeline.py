
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

from src.utils.helpers import batching


class DatasetCreation(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 n_pred_vectors, metadata = None):
        super().__init__(dataset_name, encryptor, embeddings_model, n_pred_vectors, metadata)
        self.triangulation_embedding = None
        if config.encoder_config.rotating_key:
            logger.info(f"Triangulation model is on, using {ClipEmbedding.name}")
            self.triangulation_embedding = ClipEmbedding()

        if config.cloud_config.names:
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
        predictions_for_baseline = [] # Will be used for the baseline

        triangulation_samples = embeddings[:config.experiment_config.n_triangulation_samples]

        # Add the new triangulation samples' embedding as well:
        # 1. Encrypt them
        y_tag = self.encryptor.encode(triangulation_samples)
        # 2. Embed the encryption
        y_tag = self.triangulation_embedding(y_tag)

        # For test data we won't duplicate but encrypt it only once
        number_of_new_samples = (
            self.n_pred_vectors if not is_test
            else 1
        )
        encrypted, observations,new_y = [], [], []
        with tf.device(GPU_DEVICE):  # Run the models on the GPU
            for x, label in tqdm(zip(embeddings, y),total=len(embeddings), desc="Encrypting" ,leave=True, position=0):
                for _ in range(number_of_new_samples):
                    # Encrypt and switch encryption key for each new sample
                    encrypted.append(
                        self.encryptor.encode(x.reshape(1, -1))
                    )
                    if config.encoder_config.rotating_key:
                        self.encryptor.switch_key()

                    new_y.append(label) # Duplicate the labels as the number of new samples

            for x_tags in tqdm(batching(encrypted, config.dataset_config.batch_size), desc="Embedding X_tag"):
                x_tags =  self.triangulation_embedding.forward(np.vstack(x_tags))
                observations.append(
                    np.vstack([np.hstack([x, y_tag.flatten()]) for x in
                               x_tags])
                )  # Triangulation features vector = X', Y_1', Y_2',...
            observations = np.vstack(observations)


        del embeddings, y_tag # No need for the embeddings & y_tag anymore

        if config.cloud_config.names:

            with self.cloud_model_manager as cloud:
                # Add the cloud models prediction to the overall features (X_tag, Y_tag) if needed by the config
                progress_bar = tqdm(total=len(encrypted), desc="Processing Cloud Models")


                for x_tags in batching(encrypted, config.dataset_config.batch_size):
                        cloud_predictions = []

                        with tf.device(GPU_DEVICE):  # Run the models on the GPU
                                for cloud_model in config.cloud_config.names:
                                    predictions = cloud.predict(model_name=cloud_model, batch=np.vstack(x_tags))
                                    cloud_predictions.append(predictions)
                                    predictions_for_baseline.append(predictions)

                                progress_bar.update(config.dataset_config.batch_size)


                        del x_tags

            # Add the cloud prediction features as well
            observations = np.hstack([observations, np.vstack(cloud_predictions)])
            del cloud_predictions

        if len(predictions_for_baseline) > 0:
            predictions_for_baseline = np.vstack(predictions_for_baseline)
        else:
            predictions_for_baseline = np.array(list())

        return observations, np.vstack(new_y), predictions_for_baseline