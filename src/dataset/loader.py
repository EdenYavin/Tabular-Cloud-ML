from loguru import logger
import tqdm
from src.dataset import DatasetFactory
from src.utils.config import config

class DataLoader:


    def __init__(self):
        self.factory = DatasetFactory()

    def __iter__(self):
        """
        Returns an iterator that yields datasets one at a time.
        """
        for name in tqdm.tqdm(config.dataset_config.names, total=len(config.dataset_config.names), desc="Datasets Progress", unit="dataset"):
            logger.info(f"\n####### LOADING DATASET: {name} ########\n")
            yield self.factory.get_dataset(name)

# Example usage
if __name__ == "__main__":
    data_loader = DataLoader()

    for dataset in data_loader:
        print(f"Loaded dataset: {dataset.name}")
        # You can access dataset.X and dataset.y here
