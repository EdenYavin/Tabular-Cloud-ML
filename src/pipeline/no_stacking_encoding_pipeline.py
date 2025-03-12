
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.domain.dataset import Batch
from src.utils.constansts import GPU_DEVICE
from src.pipeline.base import FeatureEngineeringPipeline
from src.utils.config import config



class NoStackingFeatureEngineeringPipeline(FeatureEngineeringPipeline):


    def _get_features(self, embeddings, y, is_test):
        num_of_cloud_models = len(config.cloud_config.names)
        observations = []
        predictions_for_baseline = [] # Will be used for the baseline
        new_y = []

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
                    number_of_new_samples = (
                        self.n_pred_vectors if not is_test
                        else 1
                    )
                    # Because we are potentially expanding X_train we should also expand y_train
                    # with the same number of labels. We do so by duplicate the labels for each new augmentation.
                    # For example if we encode x_1 to 3 new samples x_enc_1_1, x_enc_1_2, x_enc_1_2 and the original
                    # label for x_1 was 1, than we add [1,1,1] to y_train.
                    # For y_test, we just duplicate it based on the number of encryptors
                    for label in labels:
                        if not is_test:
                            [new_y.append(label) for _ in range(number_of_new_samples * self.encryptor.number_of_encryptors_to_init)]
                        else:
                            [new_y.append(label) for _ in range(self.encryptor.number_of_encryptors_to_init)]


                    with tf.device(GPU_DEVICE):
                        # Run the models on the GPU
                        images = self.encryptor.encode(mini_batch, number_of_new_samples) # We are encrypting each sample N times, where N is the number of prediction vectors we want to use as features
                        predictions = self.cloud_db.get_predictions(cloud_model, images, batch.progress.n, is_test)


                    # We are then creating a prediction vector for each new encoded sample (image)
                    predictions = np.vstack(predictions) # Create one feature vector of all concatenated predictions
                    observation =  np.hstack([predictions, mini_batch])

                    observations.append(observation)
                    predictions_for_baseline.append(predictions)

        return np.vstack(observations), np.vstack(new_y), np.vstack(predictions_for_baseline)