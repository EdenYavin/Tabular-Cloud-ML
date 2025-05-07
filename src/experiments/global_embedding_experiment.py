import numpy as np

from tqdm import tqdm
import pandas as pd

from src.pipeline.golbal_embedding_pipeline import FeatureEngineeringPipeline
from src.cloud import CloudModel, CLOUD_MODELS
from src.encryptor.base import Encryptors
from src.encryptor import EncryptorFactory
from src.internal_model import InternalInferenceModelFactory
from src.embeddings import EmbeddingsFactory
from src.dataset import DATASETS, RawDataset
from src.utils.config import config
from loguru import logger


class GlobalEmbeddingExperimentHandler:

    def __init__(self):
        self.experiment_name = "global embedding"

    def run_experiment(self):

        # Create a final report with average metrics
        final_report = pd.DataFrame()
        cloud_models: list[CloudModel] = [CLOUD_MODELS[name] for name in config.cloud_config.name]

        datasets = config.dataset_config.names
        test_dataset_name = datasets[-1].name


        X_train,y_train, X_test, y_test = [], [], [], []

        for idx, dataset_name in tqdm(enumerate(datasets), total=len(datasets), desc="Datasets Progress", unit="dataset"):
            raw_dataset: RawDataset = DATASETS[dataset_name]()


            embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name.value)
            encryptor = Encryptors(output_shape=cloud_models[0].input_shape,
                                   number_of_encryptors_to_init=1, # We always want only 1 encryptor
                                   enc_base_cls=EncryptorFactory.get_model_cls()
                                   )

            X,y = raw_dataset.get_dataset()
            logger.info(f"Dataset size: {X.shape}")


            logger.debug(f"CREATING THE PREDICTIONS TRAIN DATASET FROM {dataset_name}")

            dataset_creator = FeatureEngineeringPipeline(
                dataset_name=dataset_name,
                cloud_models=cloud_models,
                encryptor=encryptor,
                embeddings_model=embedding_model,
                metadata=raw_dataset.metadata
            )
            dataset = dataset_creator.create(X, y)
            logger.debug("Finished Creating the dataset")

            # The last dataset will be the test set, check if this is the last one
            if idx == len(datasets) - 1:
                X_test.append(dataset.features)
                y_test.append(dataset.y)

            X_train.append(dataset.features)
            y_train.append(dataset.labels)

            del dataset  # Free up space

        X_train, y_train = np.vstack(X_train), np.vstack(y_train)
        X_test, y_test = np.vstack(X_test), np.vstack(y_test)


        logger.info(f"#### EVALUATING INTERNAL MODEL ####\nDataset Shape: Train - {X_train.shape}, Test: {X_test.shape}")
        internal_model = InternalInferenceModelFactory().get_model(
            num_classes=y_train.shape[1],
            input_shape=X_train.shape[1],
            # Only give the number of features
        )
        internal_model.fit(X_train, y_train)
        test_acc, test_f1 = internal_model.evaluate(X_test, y_test)

        logger.info(f"""
              IIM: {test_acc}, {test_f1}\n
              """)

        final_report = pd.concat(
            [
                final_report,
                pd.DataFrame(
                    {
                        "exp_name": [self.experiment_name],
                        "dataset": [test_dataset_name],
                        "iim_model": [internal_model.name],
                        "embedding": [config.embedding_config.name],
                        "encryptor": [config.encoder_config.name],
                        "iim_test_acc": [test_acc],
                        "iim_test_f1": [test_f1]
                    }
                )
            ])




        return final_report
