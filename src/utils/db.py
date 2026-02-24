import json, pickle
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import os, tensorflow as tf

from src.cloud import CloudModelManager
from src.dataset.base import RawDataset
from src.domain.dataset import IIMDataset
from src.utils.config import config
from src.embeddings.model import RawDataEmbedding
from src.utils.constansts import (DATA_CACHE_PATH,
                                  DB_TRAIN_INDEX_TOKEN, DB_TEST_INDEX_TOKEN, DB_IMM_TRAIN_INDEX_TOKEN,
                                  CLOUD_PRED_CACHE_DIR_NAME
                                  )
from src.utils.helpers import get_experiment_name


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
            # Get the existing indices and create new dataframes
            X_train = pd.DataFrame(X.loc[indexes[DB_TRAIN_INDEX_TOKEN]])
            y_train = pd.Series(y.loc[indexes[DB_TRAIN_INDEX_TOKEN]])
            X_sample = pd.DataFrame(X.loc[indexes[DB_IMM_TRAIN_INDEX_TOKEN]])
            y_sample = pd.Series(y.loc[indexes[DB_IMM_TRAIN_INDEX_TOKEN]])
            X_test = pd.DataFrame(X.loc[indexes[DB_TEST_INDEX_TOKEN]])
            y_test = pd.Series(y.loc[indexes[DB_TEST_INDEX_TOKEN]])
            logger.info(f"{self.dataset.name} LOADED INDEX {config.dataset_config.split_ratio} - INDEX SIZE {len(X_sample)}")


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

    def get_embedding(self, samples, is_test=False):

        key = DB_TRAIN_INDEX_TOKEN if not is_test else DB_TEST_INDEX_TOKEN

        if type(self.embedding_model) is RawDataEmbedding:
            # Special case to not waste resources
            return self.embedding_model(samples)

        if not self.db:
            self.db = self.load()

        embeddings = []
        for i, sample in tqdm(enumerate(samples), total=len(samples), position=0, leave=True, desc=f"{key} Embedding Dataset"):
            embedding = self.db.get(key, {}).get(i, None)
            if embedding is None:
                embedding = self.embedding_model(sample.reshape(1, -1))
                self.db.setdefault(key, {}).setdefault(i, embedding)

            embeddings.append(embedding)

        return np.vstack(embeddings)


class CloudPredictionDataDatabase:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        path = Path(DATA_CACHE_PATH) / dataset_name / CLOUD_PRED_CACHE_DIR_NAME
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.cloud_model_manager = CloudModelManager()

    def get_dataset(self, is_test: bool = False):
        train_test_dir = "train" if not is_test else "test"
        cache_dir = self.path / train_test_dir
        dataset = []
        for file in os.listdir(cache_dir):
            if file.endswith(".npy"):
                dataset.append(np.load(cache_dir / file))

        return np.vstack(dataset)

    def get_predictions(self, cloud_model_name: str, batch: np.ndarray, index: int, is_test: bool):
        train_test_dir = "train" if not is_test else "test"
        rotate_dir = "-rotate" if config.encoder_config.rotating_key else ""
        cache_dir = self.path / cloud_model_name / f"{train_test_dir}{rotate_dir}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = cache_dir / f"{index}.npy"
        if os.path.exists(cache_file): # For rotating key we will have to always use the model
            # Load the cached processed batch from disk if it exists.
            return np.load(cache_file)
        else:
            # Lazy loading, we only load the cloud model if we have not seen it before and
            # use it for the entire batches. We switch models once we see a new model
            # Process the batch and save the result to disk.
            processed_batch = self.cloud_model_manager.predict(cloud_model_name, batch)
            np.save(cache_file, processed_batch)
            return processed_batch


class EncryptionDatasetDB:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.path = Path(DATA_CACHE_PATH) / dataset_name / f"{get_experiment_name()}.pkl"

    def is_db_exists(self):
        return os.path.exists(self.path)

    def get_dataset(self) -> IIMDataset:
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                return pickle.load(f)

        return None

    def get_shape(self) -> tuple:
        return self.get_dataset().train.features.shape if self.get_dataset() else (0, 0)

    def append(self, new_dataset: IIMDataset) -> IIMDataset:

        logger.info(f"Appending {new_dataset.train.features.shape[0]} samples to {self.dataset_name}")
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data: IIMDataset = pickle.load(f)

            new_dataset.train.features = np.vstack([data.train.features, new_dataset.train.features])
            new_dataset.train.labels = np.vstack([data.train.labels, new_dataset.train.labels])
            logger.info(f"New size after appending: {new_dataset.train.features.shape}")

        with open(self.path, "wb") as f:
            pickle.dump(new_dataset, f)

        return new_dataset


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


class KFoldSplitDB:
    """
    Generates and caches K stratified fold index splits for a raw dataset.

    Fold indices are stored as:
        store/dataset/<dataset>_kfold_<K>.json

    Each fold key maps to {"train_index": [...], "test_index": [...]}.
    """

    def __init__(self, dataset: RawDataset, n_splits: int):
        self.dataset = dataset
        self.n_splits = n_splits

        db_path = os.path.join(DATA_CACHE_PATH, dataset.name)
        os.makedirs(db_path, exist_ok=True)
        self.db_path = os.path.join(db_path, f"{dataset.name}_kfold_{n_splits}.json")

        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.db = json.load(f)
            logger.info(
                f"Loaded existing {n_splits}-fold splits for '{dataset.name}' "
                f"from {self.db_path}"
            )
        else:
            self.db = {}
            self._generate_folds()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_folds(self):
        """Create stratified K-fold splits and persist them to disk."""
        X, y = self.dataset.get_dataset()

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            self.db[str(fold_idx)] = {
                "train_index": X.iloc[train_idx].index.tolist(),
                "test_index": X.iloc[test_idx].index.tolist(),
            }

        with open(self.db_path, "w") as f:
            json.dump(self.db, f)

        logger.info(
            f"Generated and cached {self.n_splits}-fold splits for "
            f"'{self.dataset.name}' → {self.db_path}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fold(self, k: int) -> tuple:
        """
        Return (X_train, X_test, y_train, y_test) numpy arrays for fold *k*.

        X_train / y_train : the K-1 folds used for pipeline/model training.
        X_test  / y_test  : the held-out fold used for evaluation only.
        """
        if str(k) not in self.db:
            raise ValueError(
                f"Fold {k} not found in KFoldSplitDB for '{self.dataset.name}'. "
                f"Available folds: 0..{self.n_splits - 1}"
            )

        X, y = self.dataset.get_dataset()
        fold_data = self.db[str(k)]

        X_train = X.loc[fold_data["train_index"]]
        y_train = y.loc[fold_data["train_index"]]
        X_test  = X.loc[fold_data["test_index"]]
        y_test  = y.loc[fold_data["test_index"]]

        logger.info(
            f"Fold {k}/{self.n_splits - 1} '{self.dataset.name}': "
            f"train={len(X_train)}, test={len(X_test)}"
        )

        return X_train.values, X_test.values, y_train.values, y_test.values


class KFoldSplitDBFactory:
    """Singleton factory — one KFoldSplitDB per (dataset_name, n_splits) pair."""

    _dbs: dict = {}

    @staticmethod
    def get_db(dataset: RawDataset, n_splits: int) -> KFoldSplitDB:
        key = (dataset.name, n_splits)
        if key not in KFoldSplitDBFactory._dbs:
            KFoldSplitDBFactory._dbs[key] = KFoldSplitDB(dataset, n_splits)
        return KFoldSplitDBFactory._dbs[key]