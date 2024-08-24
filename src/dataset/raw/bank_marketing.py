from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.raw.base import DATASET_DIR, RawDataset

class BankMarketing(RawDataset):
    name = 'bank_marketing'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / BankMarketing.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = BankMarketing.name

    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        y = y.replace({"no": 0, "yes": 1}).astype(int)
        return preprocess(X, cloud_dataset=True), y


