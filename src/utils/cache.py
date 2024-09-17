import pickle, os
from src.utils.constansts import LLM_CACHE_PATH

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
