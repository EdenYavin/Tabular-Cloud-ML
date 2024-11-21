from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.base import DATASET_DIR, RawDataset

class AirlineSatisfaction(RawDataset):
    name = 'airline_satisfaction'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / AirlineSatisfaction.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        if "Unnamed: 0" in dataset.columns:
            dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = AirlineSatisfaction.name
        self.metadata["labels"] = [0, 1]
        self.metadata['targe_column'] = "satisfaction"
        self.metadata['description'] = ""


    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, 1:-1], dataset.iloc[:, -1]
        y = y.replace({"satisfied": 1, "neutral or dissatisfied": 0}).astype(int)
        return preprocess(X, cloud_dataset=True), y


