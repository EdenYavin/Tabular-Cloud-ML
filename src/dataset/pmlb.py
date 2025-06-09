import os.path

from pmlb import fetch_data
import pandas as pd

from src.dataset.base import RawDataset, DATASET_DIR
from src.utils.constansts import DATASET_NAME_TOKEN
from src.utils.helpers import preprocess

class PMLBDataset(RawDataset):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        dataset_name = kwargs.get(DATASET_NAME_TOKEN)
        DATASET_PATH = DATASET_DIR / dataset_name
        file_path = DATASET_PATH / "dataset.csv"
        if os.path.exists(file_path):
            dataset = pd.read_csv(file_path)
            if "Unnamed: 0" in dataset.columns:
                dataset.drop(columns=["Unnamed: 0"], inplace=True)
        else:
            dataset = fetch_data(dataset_name)
            os.makedirs(DATASET_PATH, exist_ok=True)
            dataset.to_csv(file_path, index=False)

        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = dataset_name
        self.metadata["labels"] = [0, 1]
        self.metadata['targe_column'] = "target"

    def _preprocess(self, dataset: pd.DataFrame):
        dataset = dataset.dropna().sample(frac=1, random_state=42).reset_index(drop=True)
        X,y = dataset.drop(columns=["target"]), dataset["target"]
        return preprocess(X, cloud_dataset=True), y
