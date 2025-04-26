
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.domain.dataset import Batch
from src.embeddings import ClipEmbedding
from src.encryptor.base import Encryptors
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config
from loguru import logger


class NoStackingFeatureEngineeringPipeline(FeatureEngineeringPipeline):

    def __init__(self, dataset_name, encryptor: Encryptors, embeddings_model,
                 n_pred_vectors, metadata = None):
        super().__init__(dataset_name, encryptor, embeddings_model, n_pred_vectors, metadata)
        self.triangulation_embedding = None
        if config.encoder_config.rotating_key:
            logger.info(f"Triangulation model is on, using {ClipEmbedding.name}")
            self.triangulation_embedding = ClipEmbedding()

    def _get_features(self, embeddings, y, is_test):
        num_of_cloud_models = len(config.cloud_config.names)
        observations = []
        predictions_for_baseline = [] # Will be used for the baseline
        new_y = []

        triangulation_samples = embeddings[:config.experiment_config.n_triangulation_samples]

        with tqdm(total=num_of_cloud_models, desc="Processing models") as pbar:
            for idx, cloud_model in enumerate(config.cloud_config.names):
                # Update the description with the current cloud model name
                pbar.set_description(f"Processing {cloud_model}")
                pbar.update(1)

                # Start processing the data using batches. We will do it for each cloud model
                # separately and use the cloud cache to save processed file to save up memory (no need to
                # load the cloud model) and speed up the runtime.
                batch = Batch(X=embeddings,y=y, size=config.dataset_config.batch_size)
                for mini_batch, labels in batch:

                    # For the training set creation we can create multiple encoded images for each input
                    # and by doing so augmenting the training set beyond the original size.
                    # For the testing, we can't create new samples
                    if not is_test:
                        number_of_new_samples = (
                            self.n_pred_vectors * self.encryptor.number_of_encryptors_to_init
                        )
                        number_of_samples_encoding = self.n_pred_vectors
                    else:
                        number_of_new_samples = 1
                        number_of_samples_encoding = 1

                    # Because we are potentially expanding X_train we should also expand y_train
                    # with the same number of labels. We do so by duplicate the labels for each new augmentation.
                    # For example if we encode x_1 to 3 new samples x_enc_1_1, x_enc_1_2, x_enc_1_2 and the original
                    # label for x_1 was 1, than we add [1,1,1] to y_train.
                    # For y_test, we just duplicate it based on the number of encryptors
                    for label in labels:
                        [new_y.append(label) for _ in range(number_of_new_samples)]

                    with tf.device(GPU_DEVICE):

                        observation = []

                        if config.experiment_config.use_embedding:
                            observation.append(mini_batch)


                        # Run the models on the GPU
                        # We are encrypting each sample N times, where N is the number of prediction vectors we want to use as features
                        images = self.encryptor.encode(mini_batch, number_of_samples_encoding)
                        if config.experiment_config.use_preds:
                            predictions = self.cloud_db.get_predictions(cloud_model, images, batch.progress.n, is_test)
                            observation.append(predictions)

                        if config.encoder_config.rotating_key:

                            # embed the encrypted samples
                            x_tag = self.triangulation_embedding.forward(images)
                            observation.append(x_tag)

                            # Add the new triangulation samples' embedding as well:
                            # 1. Encrypt them
                            y_tag = self.encryptor.encode(triangulation_samples)
                            # 2. Embed the encryption
                            y_tag = self.triangulation_embedding(y_tag)
                            observation.append(np.vstack([np.hstack([x, y_tag.flatten()]) for x in x_tag])) # Triangulation features vector = X', Y_1', Y_2',...

                            # Rotate the key for the next sample to be encoded by a new key
                            self.encryptor.switch_key()

                    observations.append(np.hstack(observation))
                    predictions_for_baseline.append(predictions)

        return np.vstack(observations), np.vstack(new_y), np.vstack(predictions_for_baseline)