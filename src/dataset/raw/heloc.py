import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.utils.constansts import DATASETS_PATH
from src.utils.helpers import preprocess
import pathlib

DATASET_DIR = pathlib.Path(DATASETS_PATH)


class RawDataset:
    def __init__(self, **kwargs):

        self.X, self.y = None, None
        self.sample_split = kwargs.get("ratio")



    def get_dataset(self):
        return self.X, self.y

    def k_fold_iterator(self, n_splits=10, shuffle=True, random_state=None):
        """Yields train and test splits for K-Fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            _, X_sample, _, y_sample = train_test_split(
                X_train, y_train, test_size=self.sample_split, stratify=y_train, random_state=42
            )

            yield X_train.values, X_test.values, X_sample.values, y_sample.values, y_train.values, y_test.values



    def get_baseline(self, X_train, X_test, y_train, y_test):
        clf = XGBClassifier()
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        return accuracy_score(y_test, preds), f1_score(y_test, preds, average='weighted')



class HelocDataset(RawDataset):

    def __init__(self, **kwargs):

        super().__init__()
        self.name = "heloc"
        DATASET_PATH = DATASET_DIR / self.name /  f"{self.name}.csv"
        dataset = pd.read_csv(DATASET_PATH)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")

    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
        y = y.replace({"Bad": 0, "Good": 1}).astype(int)
        return preprocess(X), y


