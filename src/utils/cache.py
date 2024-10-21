import pickle, os
from src.utils.constansts import LLM_CACHE_PATH, DATA_CACHE_PATH
import src.utils.constansts as consts


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


class ExperimentDataCache:
    """
    To enable faster experiments we need to have a cache with a specific use case of the cloud dataset.
    This class will be tasked with caching the dataset after the pipeline, and with smart fetching.
    """

    def __init__(self, dataset_name, split_ratio, one_hot):
        self.dataset_name = dataset_name
        self.split_ratio =  split_ratio
        self.one_hot = one_hot
        self.cache = self.load()

    def save(self):

        file_name = f"{self.dataset_name}_{self.one_hot}.pkl"
        cache_file_name = os.path.join(DATA_CACHE_PATH, file_name)

        with open(cache_file_name, 'wb') as f:
            pickle.dump(self.cache, f)

    def load(self):

        file_name = f"{self.dataset_name}_{self.one_hot}.pkl"
        cache_file_name = os.path.join(DATA_CACHE_PATH, file_name)

        if not os.path.exists(cache_file_name):
            return {}

        with open(cache_file_name, 'rb') as f:
            return pickle.load(f)

    def get(self, key):

        hit = self.cache.get(key)

        if self.split_ratio < 1:
            return hit[self.split_ratio]

        return hit

    def set(self, key, value):
        self.cache[key] = value
