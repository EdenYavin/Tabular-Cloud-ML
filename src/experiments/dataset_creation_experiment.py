import gc
import os
import pickle
from loguru import logger
from tqdm import tqdm

from src.pipeline.deepset_features_dataset import DeepSetFeatureEngineering
from src.pipeline.triangulations_features_dataset import TriangulationFeatureEngineering
from src.pipeline.raw_features_engineering import RawFeaturesEngineering
from src.cloud import CLOUD_MODELS, DEFAULT_CLOUD_OUTPUT_SHAPE
from src.encryptor import EncryptorFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import DATASET_FILE_NAME, BASELINE_DATASET_FILE_NAME, REPORT_PATH, EXPERIMENTS


class DatasetCreationHandler(ExperimentHandler):

    def __init__(self, report_path = REPORT_PATH):
        super().__init__(get_experiment_name(), report_path=report_path)

    def run_experiment(self):

        datasets = config.dataset_config.names

        # Get the output for the cloud model
        if config.cloud_config.names:
            cloud_model_output = CLOUD_MODELS[config.cloud_config.names[0]].input_shape
        else:
            cloud_model_output = DEFAULT_CLOUD_OUTPUT_SHAPE

        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):
            with logger.contextualize(dataset=dataset_name):

                raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)

                embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name)
                encryptor = EncryptorFactory.get_model(dataset_name=dataset_name, output_shape=cloud_model_output,)

                X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
                logger.debug(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")
                del X_train, y_train

                for n_pred_vectors in tqdm(range(1, self.n_pred_vectors + 1), desc=f"Preparing Dataset {dataset_name}", unit="dataset"):

                    if os.path.exists(get_dataset_path(dataset_name, n_pred_vectors) / DATASET_FILE_NAME) and config.dataset_config.use_cache:
                        logger.info(f"Dataset {get_dataset_path(dataset_name, n_pred_vectors)}"
                                    f" already exists, skipping creation")
                        continue

                    logger.debug(f"Experiment name is {self.experiment_name}, Dataset is {dataset_name}. "
                                 f"#### Number of datasets versions: {n_pred_vectors} ####")

                    if config.experiment_config.use_deepset:
                        dataset_creator = DeepSetFeatureEngineering(
                            dataset_name=dataset_name,
                            encryptor=encryptor,
                            embeddings_model=embedding_model,
                            metadata=raw_dataset.metadata
                        )

                    elif config.experiment_config.n_triangulation_samples > 0 and not config.experiment_config.use_raw:
                        # Create dataset with triangulations
                        dataset_creator = TriangulationFeatureEngineering(
                            dataset_name=dataset_name,
                            encryptor=encryptor,
                            embeddings_model=embedding_model,
                            metadata=raw_dataset.metadata
                        )
                    else:
                        # No need for triangulations
                        dataset_creator = RawFeaturesEngineering(
                            dataset_name=dataset_name,
                            encryptor=encryptor,
                            embeddings_model=embedding_model,
                            metadata=raw_dataset.metadata
                        )

                    dataset, emb_baseline_dataset = (
                        dataset_creator.create(X_sample, y_sample, X_test, y_test)
                    )

                    path = get_dataset_path(dataset_name, n_pred_vectors)
                    os.makedirs(path, exist_ok=True)

                    logger.debug("Finished Creating the dataset.\n"
                                 f"Saving to {path}")

                    with open(path / BASELINE_DATASET_FILE_NAME, "wb") as f:
                        pickle.dump(emb_baseline_dataset, f)

                    with open(path / DATASET_FILE_NAME, "wb") as f:
                        pickle.dump(dataset, f)


                    del dataset, emb_baseline_dataset
                    gc.collect()


