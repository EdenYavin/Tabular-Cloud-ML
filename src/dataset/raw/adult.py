from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.raw.base import DATASET_DIR, RawDataset

class AdultDataset(RawDataset):
    name = "adult"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / AdultDataset.name /  "dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        if "Unnamed: 0" in dataset.columns:
            dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = AdultDataset.name

    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        y = y.replace({"<=50K": 0, "<=50K.":0, ">50K": 1, ">50K.": 1}).astype(int)
        return preprocess(X, cloud_dataset=True), y


