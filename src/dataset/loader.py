from loguru import logger
import tqdm
from src.dataset import DATASETS
from src.utils.config import config

class DataLoader:
    def __init__(self):
        """
        Initializes the DataLoader with a list of dataset names.

        :param dataset_names: List of dataset names to load.
        """
        self.datasets = {k:v for k,v in DATASETS.items() if k in config.dataset_config.names}


    def __iter__(self):
        """
        Returns an iterator that yields datasets one at a time.
        """
        for name, cls in tqdm.tqdm(self.datasets.items(), total=len(self.datasets), desc="Datasets Progress", unit="dataset"):
            logger.info(f"\n####### LOADING DATASET: {name} ########\n")
            yield cls()

# Example usage
if __name__ == "__main__":
    data_loader = DataLoader()

    for dataset in data_loader:
        print(f"Loaded dataset: {dataset.name}")
        # You can access dataset.X and dataset.y here
