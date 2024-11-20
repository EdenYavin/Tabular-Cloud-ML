from src.utils.helpers import preprocess
import pandas as pd
from src.dataset.base import DATASET_DIR, RawDataset

class LoadApprovalDataset(RawDataset):
    name = 'load_approval'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        DATASET_PATH = DATASET_DIR / LoadApprovalDataset.name /  f"dataset.csv"
        dataset = pd.read_csv(DATASET_PATH)
        if "Unnamed: 0" in dataset.columns:
            dataset.drop(columns=["Unnamed: 0"], inplace=True)
        self.X, self.y = self._preprocess(dataset)
        self.cloud_models = kwargs.get("cloud_models")
        self.name = LoadApprovalDataset.name
        self.metadata["labels"] = [0, 1]
        self.metadata['targe_column'] = "loan_status"
        self.metadata['description'] = "This dataset is a synthetic version inspired by the original Credit Risk dataset on Kaggle and enriched with additional variables based on Financial Risk for Loan Approval data. SMOTENC was used to simulate new data points to enlarge the instances. The dataset is structured for both categorical and continuous features."


    def _preprocess(self, dataset):
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        y = y.astype(int)
        return preprocess(X, cloud_dataset=True), y


