import json, pickle
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

from src.cloud import CloudModel, CLOUD_MODELS
from src.dataset.base import RawDataset
from src.utils.config import config
from src.embeddings.model import RawDataEmbedding
from src.utils.constansts import (DATA_CACHE_PATH, DB_EMBEDDING_TOKEN, DB_LABEL_TOKEN, DB_RAW_FEATURES_TOKEN,
                                  DB_TRAIN_INDEX_TOKEN, DB_TEST_INDEX_TOKEN, DB_IMM_TRAIN_INDEX_TOKEN,
                                  CLOUD_PRED_CACHE_DIR_NAME, DATASETS
                                  )


class RawDataExperimentDatabase:
    """
    This will enable consistent experiments - Each split of dataset will be save, i.e. the indexes will be saved.
    Thus changes to the data will not result in changes to the split
    """
    def __init__(self, dataset: RawDataset):
        self.dataset = dataset
        db_path = os.path.join(DATA_CACHE_PATH, dataset.name)
        os.makedirs(db_path, exist_ok=True)
        self.db_path = os.path.join(db_path, f"{dataset.name}_dataset.json")
        self.key = str(config.dataset_config.split_ratio)
        if os.path.exists(self.db_path):
            self.db = json.load(open(self.db_path, "r"))
            self.empty = False if self.key in self.db else True
        else:
            self.db = {config.dataset_config.split_ratio: {}}
            self.empty = True


    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.db, f)

    def get_split(self):

        X, y = self.dataset.get_dataset()

        if self.empty:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y,
                                                                random_state=42)

            if config.dataset_config.split_ratio == 1:
                # Use the entire Train set as the sample
                X_sample, y_sample = X_train, y_train

            else:
                _, X_sample, _, y_sample = train_test_split(X_train, y_train,
                                                        test_size=config.dataset_config.split_ratio, stratify=y_train,
                                                        random_state=42)

            new_data = {}
            new_data[DB_TRAIN_INDEX_TOKEN] = X_train.index.tolist()
            new_data[DB_IMM_TRAIN_INDEX_TOKEN] = X_sample.index.tolist()
            new_data[DB_TEST_INDEX_TOKEN] = X_test.index.tolist()

            self.db[self.key] = new_data

            logger.info(f"#### CREATED NEW INDEX FOR {config.dataset_config.split_ratio} - INDEX SIZE {len(X_sample)}")

            self._save()

        else:
            indexes = self.db[self.key]
            logger.info(f"LOAD INDEX {config.dataset_config.split_ratio} - INDEX SIZE {len(DB_IMM_TRAIN_INDEX_TOKEN)}")
            # Get the existing indices and create new dataframes
            X_train = pd.DataFrame(X.loc[indexes[DB_TRAIN_INDEX_TOKEN]])
            y_train = pd.Series(y.loc[indexes[DB_TRAIN_INDEX_TOKEN]])
            X_sample = pd.DataFrame(X.loc[indexes[DB_IMM_TRAIN_INDEX_TOKEN]])
            y_sample = pd.Series(y.loc[indexes[DB_IMM_TRAIN_INDEX_TOKEN]])
            X_test = pd.DataFrame(X.loc[indexes[DB_TEST_INDEX_TOKEN]])
            y_test = pd.Series(y.loc[indexes[DB_TEST_INDEX_TOKEN]])

        return X_train.values, X_test.values, X_sample.values, y_train.values, y_test.values, y_sample.values


class ExperimentDatabase:
    """
    To enable faster experiments we need to have a cache with a specific use case of the cloud dataset.
    This class will be tasked with caching the data after the pipeline, and with smart fetching.
    This class will be a singelton so it can be called from different processes and parts of the code.
    """

    def __init__(self, dataset_name, embedding_model):
        self.dataset_name = dataset_name
        self.embedding_model = embedding_model

        folder_path = os.path.join(DATA_CACHE_PATH, dataset_name)
        os.makedirs(folder_path, exist_ok=True)

        cache_file_name = f"{self.dataset_name}_{self.embedding_model.name}_db.pkl"
        self.cache_file_path = os.path.join(folder_path, cache_file_name)
        self.db = {}

    def save(self):
        try:
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.db, f)

            del self.db
            self.db = {}

        except Exception as e:
            logger.error(str(e))
            logger.warning("Skipping saving")

    def load(self):

        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, 'rb') as f:
                return pickle.load(f)

        else:
            return {}

    def get_embedding(self, samples):

        if type(self.embedding_model) is RawDataEmbedding:
            # Special case to not waste resources
            return self.embedding_model(samples)

        if not self.db:
            self.db = self.load()

        embeddings = []
        for i, sample in samples.iterrows():
            embedding = self.db.get(i, {}).get(DB_EMBEDDING_TOKEN, None)
            if embedding is None:
                embedding = self.embedding_model(sample.values.reshape(1, -1))
                self.set_embedding(i, embedding)

            embeddings.append(embedding)

        embeddings = np.vstack(embeddings)

        return embeddings

    def set_label(self, idx, value):
        self.db[idx][DB_LABEL_TOKEN] = value

    def set_feature(self, idx, value):
        self.db[idx][DB_RAW_FEATURES_TOKEN] = value

    def set_embedding(self, idx, value):
        self.db.setdefault(idx, {}).setdefault(DB_EMBEDDING_TOKEN, value)


class CloudPredictionDataDatabase:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        path = Path(DATA_CACHE_PATH) / dataset_name / CLOUD_PRED_CACHE_DIR_NAME
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.cloud_model: CloudModel | None = None

    def get_predictions(self, cloud_model_name: str, batch: np.ndarray, index: int, is_test: bool):
        train_test_dir = "train" if not is_test else "test"
        cache_dir = self.path / cloud_model_name / train_test_dir
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = cache_dir / f"{index}.npy"
        if os.path.exists(cache_file):
            # Load the cached processed batch from disk if it exists.
            return np.load(cache_file)
        else:
            # Lazy loading, we only load the cloud model if have not seen it before and
            # use it for the entire batches. We switch models once we see a new model
            if not self.cloud_model:
                self.cloud_model = CLOUD_MODELS[cloud_model_name]()
            elif cloud_model_name != self.cloud_model.name:
                del self.cloud_model # Unload the previous model to free up memory
                self.cloud_model = CLOUD_MODELS[cloud_model_name]()

            # Process the batch and save the result to disk.
            processed_batch = self.cloud_model.predict(batch)
            np.save(cache_file, processed_batch)
            return processed_batch


class EmbeddingDBFactory:
    dbs = {}

    @staticmethod
    def get_db(dataset_name, embedding_model):
        if dataset_name in EmbeddingDBFactory.dbs:
            return EmbeddingDBFactory.dbs[dataset_name]

        EmbeddingDBFactory.dbs[dataset_name] = ExperimentDatabase(dataset_name, embedding_model)
        return EmbeddingDBFactory.dbs[dataset_name]

class RawSplitDBFactory:
    dbs = {}

    @staticmethod
    def get_db(dataset: RawDataset):
        dataset_name = dataset.name
        if dataset_name in RawSplitDBFactory.dbs:
            return RawSplitDBFactory.dbs[dataset_name]

        RawSplitDBFactory.dbs[dataset_name] = RawDataExperimentDatabase(dataset)
        return RawSplitDBFactory.dbs[dataset_name]