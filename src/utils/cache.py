import pickle, os

import numpy as np
from src.utils.constansts import LLM_CACHE_PATH, DATA_CACHE_PATH, DB_EMBEDDING_TOKEN, DB_LABEL_TOKEN, DB_RAW_FEATURES_TOKEN
from src.utils.config import config
from src.utils.helpers import one_hot_labels

class Cache:
    def __init__(self, cache_file='cache.pkl', flush_every=100):
        self.cache_file = os.path.join(LLM_CACHE_PATH, cache_file)
        self.cache = {}
        self.load()
        self.flush_every = flush_every


    def load(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)

    def save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

        if len(self.cache) > self.flush_every:
            self.save()


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


class DBFactory:
    dbs = {}

    @staticmethod
    def get_db(dataset_name, embedding_model):
        if dataset_name in DBFactory.dbs:
            return DBFactory.dbs[dataset_name]

        DBFactory.dbs[dataset_name] = ExperimentDatabase(dataset_name, embedding_model)
        return DBFactory.dbs[dataset_name]
