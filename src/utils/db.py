import json, pickle

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

from src.dataset.raw import RawDataset
from src.utils.config import config
from src.utils.constansts import (DATA_CACHE_PATH, DB_EMBEDDING_TOKEN, DB_LABEL_TOKEN, DB_RAW_FEATURES_TOKEN,
                                  DB_TRAIN_INDEX_TOKEN, DB_TEST_INDEX_TOKEN, DB_IMM_TRAIN_INDEX_TOKEN)
from src.utils.helpers import one_hot_labels


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
        if os.path.exists(self.db_path):
            self.db = json.load(open(self.db_path, "r"))
        else:
            self.db = {}

        self.empty = True if len(self.db) == 0 else False

    def __del__(self):
        self._save()

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


            self.db[DB_TRAIN_INDEX_TOKEN] = X_train.index.tolist()
            self.db[DB_IMM_TRAIN_INDEX_TOKEN] = X_sample.index.tolist()
            self.db[DB_TEST_INDEX_TOKEN] = X_test.index.tolist()

            self._save()

        else:
            # Get the existing indices and create new dataframes
            X_train = pd.DataFrame(X.loc[self.db[DB_TRAIN_INDEX_TOKEN]])
            y_train = pd.Series(y.loc[self.db[DB_TRAIN_INDEX_TOKEN]])
            X_sample = pd.DataFrame(X.loc[self.db[DB_IMM_TRAIN_INDEX_TOKEN]])
            y_sample = pd.Series(y.loc[self.db[DB_IMM_TRAIN_INDEX_TOKEN]])
            X_test = pd.DataFrame(X.loc[self.db[DB_TEST_INDEX_TOKEN]])
            y_test = pd.Series(y.loc[self.db[DB_TEST_INDEX_TOKEN]])

        return X_train, X_test, X_sample, y_train, y_test, y_sample


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

        with open(self.cache_file_path, 'wb') as f:
            pickle.dump(self.db, f)

    def load(self):

        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, 'rb') as f:
                return pickle.load(f)

        else:
            return {}

    def get_embedding(self, samples):
        if not self.db:
            self.db = self.load()

        embeddings = []
        for i, sample in samples.iterrows():
            embedding = self.db.get(i, {}).get(DB_EMBEDDING_TOKEN, None)
            if embedding is None:
                embedding = self.embedding_model(sample.values.reshape(1, -1))
                self.db.setdefault(i, {})[DB_EMBEDDING_TOKEN] = embedding

            embeddings.append(embedding)

        embeddings = np.vstack(embeddings)

        return embeddings

    def set_label(self, idx, value):
        self.db[idx][DB_LABEL_TOKEN] = value

    def set_feature(self, idx, value):
        self.db[idx][DB_RAW_FEATURES_TOKEN] = value

    def set_embedding(self, idx, value):
        self.db.setdefault(idx, {}).setdefault(DB_EMBEDDING_TOKEN, value)

    def __del__(self):
        self.save()


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