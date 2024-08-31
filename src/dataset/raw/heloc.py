from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.raw.base import DATASET_DIR, RawDataset

class HelocDataset(RawDataset):
    name = 'heloc'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / HelocDataset.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = HelocDataset.name
        self.metadata["labels"] = ["Bad", "Good"]
        self.metadata['targe_column'] = "RiskPerformance"

    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
        y = y.replace({"Bad": 0, "Good": 1}).astype(int)
        return preprocess(X, cloud_dataset=True), y


