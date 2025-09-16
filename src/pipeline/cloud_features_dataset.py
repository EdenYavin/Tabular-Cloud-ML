
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.encryptor.base import BaseEncryptor
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from loguru import logger

class CloudFeatureEngineering(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata = None):
        super().__init__(dataset_name, encryptor, embeddings_model, metadata)
        logger.info(f"Cloud models flag is ON, using: {config.cloud_config.names} Models")

    
    def _get_features(self, X, embeddings, y, is_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        observations = []
        cloud = self.cloud_model_manager.__enter__()

        with tqdm(total=len(embeddings), leave=True, position=0, desc="Encrypting, Embedding, Predicting") as pbar:
            with tf.device(GPU_DEVICE):  # Run the models on the GPU
                logger.debug(f"Running ON GPU device: {GPU_DEVICE}")

                for x, x_emb, label in zip(X, embeddings, y):
                    pbar.update(1)

                    # Triangulation features vector = X', Y_1', Y_2',...
                    x_tag = self.encryptor.encode(x_emb.reshape(1, -1))

                    # Add the cloud predictions as features if needed:
                    predictions = []
                    for cloud_model in config.cloud_config.names:
                        predictions.append(cloud.predict(model_name=cloud_model, batch=x_tag))

                        # Flatten prediction to be stacked correctly
                    predictions = [p.flatten() for p in predictions]
                    observations.append(np.hstack(predictions))

                    del predictions

                    if config.encoder_config.rotating_key:
                        # Switch key for the next example
                        self.encryptor.switch_key()


                del x_tag

        cloud.__exit__(None, None, None)
        return np.vstack(observations), y, np.array([])