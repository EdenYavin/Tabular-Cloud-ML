from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.raw.base import DATASET_DIR, RawDataset

class GesturePhaseDataset(RawDataset):
    name = 'gesture_phase'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / GesturePhaseDataset.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        if "Unnamed: 0" in dataset.columns:
            dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = GesturePhaseDataset.name

    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        y = y.replace({"S":0,"D":1,"P":2, "R":3, "H":4}).astype(int)
        return preprocess(X, cloud_dataset=True), y


