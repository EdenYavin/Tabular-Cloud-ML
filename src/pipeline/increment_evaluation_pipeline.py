import pandas as pd
from keras.src.utils import to_categorical
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from src.domain.dataset import Batch, EmbeddingBaselineDataset, EmbeddingBaselineFeatures, PredictionBaselineDataset, \
    PredictionBaselineFeatures, IIMDataset, IIMFeatures
from src.embeddings import RawDataEmbedding
from src.encryptor.base import Encryptors
from src.utils.constansts import GPU_DEVICE
from src.utils.config import config
from src.utils.db import CloudPredictionDataDatabase


class IncrementEvalFeatureEngineeringPipeline:

    def __init__(self, dataset_name, encryptor: Encryptors, embeddings_model, n_pred_vectors=1):
        self.name = dataset_name
        self.use_embedding = config.experiment_config.use_embedding
        self.use_predictions = config.experiment_config.use_preds
        self.embedding_model = embeddings_model
        self.encryptor = encryptor
        self.cloud_db = CloudPredictionDataDatabase(dataset_name)
        self.n_pred_vectors = n_pred_vectors

    def create(self, X_train, y_train, X_test, y_test) -> tuple[list[IIMDataset] | IIMDataset, EmbeddingBaselineDataset, PredictionBaselineDataset]:

        X_emb_train = self._get_embeddings(X_train)
        X_emb_test = self._get_embeddings(X_test)

        Xs_train, new_y_train, X_pred_train = self._get_features(X_emb_train, y_train, is_test=False)
        Xs_test, new_y_test, X_pred_test = self._get_features(X_emb_test, y_test, is_test=True)

        # One hot encode the labels
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes=num_classes)
        new_y_train = to_categorical(new_y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)
        new_y_test = to_categorical(new_y_test, num_classes=num_classes)

        embeddings_baseline = EmbeddingBaselineDataset(
            train=EmbeddingBaselineFeatures(embeddings=X_emb_train, labels=y_train),
            test=EmbeddingBaselineFeatures(embeddings=X_emb_test, labels=y_test)
        )

        pred_baseline = PredictionBaselineDataset(
            train=PredictionBaselineFeatures(predictions=X_pred_train, labels=new_y_train),
            test=PredictionBaselineFeatures(predictions=X_pred_test, labels=new_y_test)
        )

        if config.experiment_config.stacking:
            # For stacking we create a new dataset for each stack model
            dataset = [
                        IIMDataset(
                        train=IIMFeatures(features=x_train, labels=y_train),
                        test=IIMFeatures(features=x_test, labels=y_test)
                    )
                for x_train, x_test in zip(Xs_train, Xs_test)
                ]
        else:
            # For non stacking we create only one dataset for one internal model
            dataset = IIMDataset(train=IIMFeatures(features=Xs_train, labels=new_y_train),
                                 test=IIMFeatures(features=Xs_test, labels=new_y_test))

        return dataset, embeddings_baseline, pred_baseline

    def _get_embeddings(self, X):
        X = pd.DataFrame(X)

        embeddings = []
        for idx, row in tqdm(X.iterrows(), total=len(X), position=0, leave=True, desc="Embedding Dataset"):

            if type(self.embedding_model) is RawDataEmbedding:
                # Special case to not waste resources
                embedding = self.embedding_model(pd.DataFrame(row).T)
            else:
                embedding = self.embedding_model(pd.DataFrame(row).T.values.reshape(1, -1))

            embeddings.append(embedding)

        return np.vstack(embeddings)

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
                        # Run the models on the GPU
                        images = self.encryptor.encode(mini_batch, number_of_samples_encoding) # We are encrypting each sample N times, where N is the number of prediction vectors we want to use as features
                        predictions = self.cloud_db.get_predictions(cloud_model, images, batch.progress.n, is_test)


                    # We are then creating a prediction vector for each new encoded sample (image)
                    predictions = np.vstack(predictions) # Create one feature vector of all concatenated predictions
                    observation =  np.hstack([predictions, mini_batch])

                    observations.append(observation)
                    predictions_for_baseline.append(predictions)

        return np.vstack(observations), np.vstack(new_y), np.vstack(predictions_for_baseline)