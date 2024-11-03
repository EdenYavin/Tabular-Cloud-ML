from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.base import DATASET_DIR, RawDataset

class HelocDataset(RawDataset):
    name = 'heloc'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / HelocDataset.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        if "Unnamed: 0" in dataset.columns:
            dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = HelocDataset.name
        self.metadata["labels"] = ["Bad", "Good"]
        self.metadata['targe_column'] = "RiskPerformance"
        self.metadata['description'] = "The HELOC dataset from FICO. Each entry in the dataset is a line of credit, typically offered by a bank as a percentage of home equity (the difference between the current market value of a home and its purchase price). The task is a binary classification task."

    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, 1:], dataset.iloc[:, 0]

        # Remove the bug in the dataset where the entire row has -9 values
        mask = ~(X == -9).all(axis=1)
        X = X[mask]
        y = y[mask]

        y = y.replace({"Bad": 0, "Good": 1}).astype(int)
        return preprocess(X, cloud_dataset=True), y


