from pmlb import fetch_data
import pandas as pd

from src.dataset.base import RawDataset
from src.utils.constansts import DATASET_NAME_TOKEN
from src.utils.helpers import preprocess

class PMLBDataset(RawDataset):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        dataset_name = kwargs.get(DATASET_NAME_TOKEN)

        dataset = fetch_data(dataset_name)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = dataset_name
        self.metadata["labels"] = [0, 1]
        self.metadata['targe_column'] = "target"

    def _preprocess(self, dataset: pd.DataFrame):
        dataset = dataset.dropna().sample(frac=1, random_state=42).reset_index(drop=True)
        X,y = dataset.drop(columns=["target"]), dataset["target"]
        return preprocess(X, cloud_dataset=True), y
