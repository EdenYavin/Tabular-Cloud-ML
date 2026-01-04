from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.embeddings import ClipEmbedding
from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from loguru import logger

class TriangulationFeatureEngineering(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata = None):
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
        logger.info(f"### USING SAE FEATURES TRIANGULATION WITH {config.experiment_config.triangulation_mode} TRIANGULATION MODE"
                    f" {'AND CALIBRATION VECTOR' if config.experiment_config.use_calibration_vector else ''} ###")


        # Add the new triangulation samples' embedding as well
        triangulation_samples = self._get_triangulation_samples(embeddings, y,
         how_to_choose=config.experiment_config.triangulation_choosing,
         n_samples=config.experiment_config.n_triangulation_samples
        )
        #Define the Calibration Vector (C) ###
        # We create a vector of ones with the same shape as a single embedding input (latent dim)
        # This acts as our "Perfect Compass" before distortion.
        calibration_vector = np.ones((1, embeddings.shape[1]))
        # -----------------------------------------------

        predictions_for_baseline = np.array(list())  # Will be used for the baseline, TODO: If needed use it
        observations, new_y =  [], []
        cloud = self.cloud_model_manager.__enter__()

        with tqdm(total=len(embeddings), leave=True, position=0, desc="Encrypting, Embedding, Predicting") as pbar:
            with tf.device(GPU_DEVICE):  # Run the models on the GPU
                logger.debug(f"Running ON GPU device: {GPU_DEVICE}")

                for x, x_emb, label in zip(X, embeddings, y):
                    pbar.update(1)

                    # Triangulation features vector = X', Y_1', Y_2',...
                    x_tag = self.encryptor.encode(x_emb.reshape(1, -1))
                    # 1. Encrypt them using the new key
                    y_tag = self.encryptor.encode(triangulation_samples)
                    # 2. Embed the encryption
                    y_tag_emb = self.triangulation_embedding.forward(y_tag)

                    # Embedding for triangulation using CLIP, those are the new features
                    x_tag_emb = self.triangulation_embedding.forward(np.vstack(x_tag))



                    if config.experiment_config.use_embedding:
                        observation = np.hstack([x, x_tag_emb.flatten(), y_tag_emb.flatten()])
                    else:
                        observation = np.hstack([x_tag_emb.flatten(), y_tag_emb.flatten()])

                    # ### Encrypt & Embed the Calibration Vector ###
                    # We encrypt C using the CURRENT key (same as x_tag and y_tag)
                    # The IIM will see how this 'all-ones' vector got twisted.
                    if config.experiment_config.use_calibration_vector:
                        c_tag = self.encryptor.encode(calibration_vector)
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
                            new_y.append(label)
                        else:
                            for p in predictions:
                                observations.append(np.hstack([observation, p]))
                                # Duplicate the labels as we duplicate each sample
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