import pickle

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.embeddings import ClipEmbedding
from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from loguru import logger

class DatasetCreation(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 n_pred_vectors, metadata = None):
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)
        self.triangulation_embedding = None
        if config.encoder_config.rotating_key:
            logger.info(f"Triangulation model is on, using {ClipEmbedding.name}")
            self.triangulation_embedding = ClipEmbedding()

        if config.cloud_config.names:
            logger.info(f"Cloud models flag is ON, using: {config.cloud_config.names} Models")

    
    def _get_features(self, embeddings, y, is_test) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        triangulation_samples = embeddings[:config.experiment_config.n_triangulation_samples]

        # For test data we won't duplicate but encrypt it only once

        predictions_for_baseline = np.array(list())  # Will be used for the baseline, TODO: If needed use it
        observations, new_y =  [], []
        cloud = self.cloud_model_manager.__enter__()
        with tqdm(total=len(embeddings), leave=True, position=0, desc="Encrypting, Embedding, Predicting") as pbar:
            with tf.device(GPU_DEVICE):  # Run the models on the GPU
                logger.debug(f"Running ON GPU device: {GPU_DEVICE}")

                for x, label in zip(embeddings, y):
                    pbar.update(1)

                    # Triangulation features vector = X', Y_1', Y_2',...
                    x_tag = self.encryptor.encode(x.reshape(1, -1))
                    # 1. Encrypt them using the new key
                    y_tag = self.encryptor.encode(triangulation_samples)
                    # 2. Embed the encryption
                    y_tag_emb = self.triangulation_embedding.forward(y_tag)

                    # Embedding fore triangulation using CLIP, those are the new features
                    x_tag_emb = self.triangulation_embedding.forward(np.vstack(x_tag))
                    observation = [x_tag_emb.flatten(), y_tag_emb.flatten()]

                    # Add embedding as features if needed
                    if config.experiment_config.use_embedding:
                        observation.append(x)
                    # Add the cloud predictions as features if needed:
                    if config.cloud_config.names:
                        for cloud_model in config.cloud_config.names:
                            predictions = cloud.predict(model_name=cloud_model, batch=x_tag)
                            observations.append(np.hstack([np.hstack(observation), predictions.flatten()]))
                            # Duplicate the labels
                            new_y.append(label)
                            del predictions
                    else:
                        # No cloud models need to be used, just use the features up until now
                        observations.append(np.hstack(observation))
                        # Duplicate the labels
                        new_y.append(label)

                    if config.encoder_config.rotating_key:
                        # Switch key for the next example
                        self.encryptor.switch_key()


                del x_tag, x_tag_emb, y_tag, y_tag_emb

        cloud.__exit__(None, None, None)
        return np.vstack(observations), np.vstack(new_y), predictions_for_baseline