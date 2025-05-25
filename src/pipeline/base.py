from abc import ABC, abstractmethod
import numpy as np
from keras.src.utils import to_categorical
from loguru import logger
from src.domain.dataset import IIMFeatures, IIMDataset, PredictionBaselineFeatures, PredictionBaselineDataset, \
    EmbeddingBaselineFeatures, EmbeddingBaselineDataset
from src.encryptor.base import BaseEncryptor
from src.utils.config import config
from src.utils.db import EmbeddingDBFactory, CloudPredictionDataDatabase, EncryptionDatasetDB
from src.utils.helpers import get_number_of_samples_to_make


class FeatureEngineeringPipeline(ABC):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 n_pred_vectors, metadata = None):

        self.n_pred_vectors = n_pred_vectors
        self.name = dataset_name
        self.use_embedding = config.experiment_config.use_embedding
        self.use_predictions = config.experiment_config.use_preds
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.embedding_db = EmbeddingDBFactory.get_db(dataset_name, self.embeddings_model)
        self.encryptor = encryptor
        self.cloud_db = CloudPredictionDataDatabase(dataset_name)
        self.encrypted_db = EncryptionDatasetDB(dataset_name)

    def _calculate_number_of_columns_to_append(self, embeddings):
        """
        Calculates the number of columns to append to the database based on the embeddings
        and existing data configuration.

        Summary:
        This private method determines the number of additional columns required for storing
        new data. It computes this value by evaluating the current database column structure,
        the embeddings provided, and specific experimental configuration parameters.

        Example: If the dataset columns size is 2600 and the embedding shape is 64 features, than:
        1000 + 64 out of the 2600 are for cloud + embedding features.
        The rest 1536 are 3 triangulations - 1536 / 512 = 3

        Args:
            embeddings (numpy.ndarray): A 2D array containing the new embeddings to be
                appended. The second dimension corresponds to the embedding size.

        Returns:
            int: The difference between the configured number of triangulation samples
            and the number of triangulations already stored in the database.
        """
        saved_data_cols_num = self.encrypted_db.get_shape()[1]
        embed_cols = embeddings.shape[1]
        emb_and_cloud_cols_indexes = embed_cols + 1000

        triangulations_indexes = saved_data_cols_num - emb_and_cloud_cols_indexes
        number_of_triangulation_used = triangulations_indexes // 512 # 512 is the clip embedding
        new_number_of_cols = config.experiment_config.n_triangulation_samples

        return new_number_of_cols - number_of_triangulation_used

    def create(self, X_train, y_train, X_test, y_test) -> tuple[list[IIMDataset] | IIMDataset, EmbeddingBaselineDataset]:#, PredictionBaselineDataset]:

        X_emb_train = self._get_embeddings(X_train, is_test=False)
        X_emb_test = self._get_embeddings(X_test, is_test=True)


        # One hot encode the labels
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        embeddings_baseline = EmbeddingBaselineDataset(
            train=EmbeddingBaselineFeatures(embeddings=X_emb_train, labels=y_train),
            test=EmbeddingBaselineFeatures(embeddings=X_emb_test, labels=y_test)
        )

        # For the training set creation we can create multiple encoded images for each input
        # and by doing so augmenting the training set beyond the original size.
        # For the testing, we can't create new samples
        original_num_samples = X_emb_train.shape[0]
        if self.encrypted_db.is_db_exists():
            current_num_samples = self.encrypted_db.get_shape()[0]
        else:
            current_num_samples = 0
        # Number of new samples:
        number_of_new_samples_to_make = get_number_of_samples_to_make(original_num_samples)

        if 0 < number_of_new_samples_to_make <= current_num_samples:
            logger.warning(f"No need to create new dataset, the cache has a dataset with size: {current_num_samples}, "
                           f"while the new dataset size is: {number_of_new_samples_to_make}")
            return self.encrypted_db.get_dataset(), embeddings_baseline

        logger.info(f"Intend to create dataset with size: {number_of_new_samples_to_make},"
                    f" as oppose to the current size: {current_num_samples}")

        # To make the dataset increase - we can duplicate the embeddings N times.
        # For example, the dataset is 1000 samples, and we want 3000 samples, we can duplicate the embedding size to
        # be 3000. The first 1000 are already cached and will be retrieved instantly without going to the cloud
        how_much_to_duplicate = number_of_new_samples_to_make // current_num_samples if current_num_samples else 1
        new_train_embeddings = np.repeat(X_emb_train, how_much_to_duplicate, axis=0)
        new_y_train = np.repeat(y_train, how_much_to_duplicate, axis=0)

        Xs_train, new_y_train, X_pred_train = self._get_features(new_train_embeddings, new_y_train, is_test=False)
        Xs_test, new_y_test, X_pred_test = self._get_features(X_emb_test, y_test, is_test=True)

        # pred_baseline = PredictionBaselineDataset(
        #     train=PredictionBaselineFeatures(predictions=X_pred_train, labels=new_y_train),
        #     test=PredictionBaselineFeatures(predictions=X_pred_test, labels=new_y_test)
        # )

        if config.iim_config.stacking:
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

        self.embedding_db.save()
        return self.encrypted_db.append(dataset), embeddings_baseline#, pred_baseline

    def _get_embeddings(self, X, is_test=False):
        embeddings = self.embedding_db.get_embedding(X, is_test)
        return np.vstack(embeddings)

    @abstractmethod
    def _get_features(self, embeddings, y, is_test):
        pass

