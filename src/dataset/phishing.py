from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.base import DATASET_DIR, RawDataset

class PhishingDataset(RawDataset):
    name = 'phishing'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / PhishingDataset.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        if "Unnamed: 0" in dataset.columns:
            dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = PhishingDataset.name
        self.metadata["labels"] = [0, 1]
        self.metadata['targe_column'] = "label"
        self.metadata['description'] = "In this dataset, we shed light on the important features that have proved to be sound and effective in predicting phishing websites."


    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
        return preprocess(X, cloud_dataset=True), y


