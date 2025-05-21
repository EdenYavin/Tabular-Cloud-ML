
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.domain.dataset import Batch
from src.embeddings import ClipEmbedding
from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from loguru import logger


class NoStackingFeatureEngineeringPipeline(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 n_pred_vectors, metadata = None):
        super().__init__(dataset_name, encryptor, embeddings_model, n_pred_vectors, metadata)
        self.triangulation_embedding = None
        if config.encoder_config.rotating_key:
            logger.info(f"Triangulation model is on, using {ClipEmbedding.name}")
            self.triangulation_embedding = ClipEmbedding()


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
        num_of_cloud_models = len(config.cloud_config.names)
        observations = []
        predictions_for_baseline = [] # Will be used for the baseline
        new_y = []

        triangulation_samples = embeddings[:config.experiment_config.n_triangulation_samples]

        with tqdm(total=num_of_cloud_models, desc="Processing models") as pbar:
            # Start processing the data using batches. We will do it for each cloud model
            # separately and use the cloud cache to save processed file to save up memory (no need to
            # load the cloud model) and speed up the runtime.
            batch = Batch(X=embeddings, y=y, size=config.dataset_config.batch_size)
            for mini_batch, labels in batch:

                    # Because we are potentially expanding X_train we should also expand y_train
                    # with the same number of labels. We do so by duplicate the labels for each new augmentation.
                    # For example if we encode x_1 to 3 new samples x_enc_1_1, x_enc_1_2, x_enc_1_2 and the original
                    # label for x_1 was 1, than we add [1,1,1] to y_train.
                    # For y_test, we just duplicate it based on the number of encryptors
                    new_samples = (
                        len(config.cloud_config.names) if config.experiment_config.use_preds
                        else 1
                    )

                    for label in labels:
                        [new_y.append(label) for _ in range(new_samples)]

                    observation = []

                    if config.experiment_config.use_embedding:
                        observation.append(mini_batch)

                    images = self.encryptor.encode(mini_batch)

                    with tf.device(GPU_DEVICE):  # Run the models on the GPU
                        if config.experiment_config.use_preds:
                            for idx, cloud_model in enumerate(config.cloud_config.names):
                                # Update the description with the current cloud model name
                                pbar.set_description(f"Processing {cloud_model}")
                                pbar.update(1)

                                predictions = self.cloud_db.get_predictions(cloud_model, images, batch.progress.n, is_test)
                                observation.append(predictions)
                                predictions_for_baseline.append(predictions)

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

                            # Rotate the key for the next sample to be encoded by a new key
                            self.encryptor.switch_key()

                    observations.append(np.hstack(observation))


        if len(predictions_for_baseline) > 0:
            predictions_for_baseline = np.vstack(predictions_for_baseline)
        else:
            predictions_for_baseline = np.array(list())

        return np.vstack(observations), np.vstack(new_y), predictions_for_baseline