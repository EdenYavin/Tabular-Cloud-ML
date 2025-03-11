import numpy as np
import tensorflow as tf
from datasets import tqdm

from src.domain.dataset import Batch
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.constansts import GPU_DEVICE
from src.utils.config import config



class StackingFeatureEngineeringPipeline(FeatureEngineeringPipeline):


    def _get_features(self, embeddings, y, is_test):
        num_of_cloud_models = len(config.cloud_config.names)
        datasets = [[] for _ in range(num_of_cloud_models)]
        predictions_for_baseline = [] # Will be used for the baseline
        embeddings_for_baseline = [] # Will be used for embedding baseline
        new_y = []

        with tqdm(total=num_of_cloud_models, desc="Processing models") as pbar:
            for idx, cloud_model in enumerate(config.cloud_config.names):
                # Update the description with the current cloud model name
                pbar.set_description(f"Processing {cloud_model}")
                pbar.update(1)

                batch = Batch(X=embeddings,y=y, size=config.dataset_config.batch_size)
                for mini_batch, labels in batch:

                    # For the training set creation we can create multiple encoded images for each input
                    # and by doing so augmenting the training set beyond the original size.
                    # For the testing, we can't create new samples
                    number_of_new_samples = (
                        self.n_pred_vectors if not is_test
                        else 1
                    )

                    embeddings_samples = np.vstack([mini_batch for _ in range(
                        number_of_new_samples)])  # Duplicate the embeddings as the number of predictions
                    embeddings_for_baseline.append(embeddings_samples)

                    with tf.device(GPU_DEVICE):

                        images = self.encryptor.encode(mini_batch,number_of_new_samples)

                        # We are then creating a prediction vector for each new encoded sample (image)
                        predictions = self.cloud_db.get_predictions(cloud_model, images, batch.progress.n, is_test)

                        predictions = np.vstack(list(predictions)) # Create one feature vector of all concatenated predictions

                        datasets[idx].append(np.hstack([predictions, embeddings_samples]))  # Shape - |CMLS|, (1,|Embedding| * Number of noise samples)

                        predictions_for_baseline.append(predictions)

                    # Because we are potentially expanding X_train we should also expand y_train
                    # with the same number of labels. We do so by duplicate the labels for each new augmentation.
                    # For example if we encode x_1 to 3 new samples x_enc_1_1, x_enc_1_2, x_enc_1_2 and the original
                    # label for x_1 was 1, than we add [1,1,1] to y_train.
                    # So we use the number of encryptos to know how much new samples we be generated.
                    # For y_test, we just duplicate it based on the number of encryptors alone
                    for label in labels:
                        if not is_test:
                            [new_y.append(label) for _ in range(number_of_new_samples * self.encryptor.number_of_encryptors_to_init)]
                        else:
                            [new_y.append(label) for _ in range(self.encryptor.number_of_encryptors_to_init)]

        return [np.vstack(d) for d in datasets], np.vstack(new_y), np.vstack(predictions_for_baseline)