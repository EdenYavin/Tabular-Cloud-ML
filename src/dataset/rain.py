from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.base import DATASET_DIR, RawDataset

class RainDataset(RawDataset):
    name = 'rain'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / RainDataset.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        if "Unnamed: 0" in dataset.columns:
            dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = RainDataset.name
        self.metadata["labels"] = [0, 1]
        self.metadata['targe_column'] = "RainTomorrow"
        self.metadata['description'] = "This dataset contains about 10 years of daily weather observations from numerous Australian weather stations. RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No?"


    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        return preprocess(X, cloud_dataset=True), y


