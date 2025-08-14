import gc
import pickle

import keras.src.backend.common.global_state
import numpy as np
from tqdm import tqdm

from src.internal_model import InternalInferenceModelFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from loguru import logger
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import DATASET_FILE_NAME, REPORT_PATH


class KModelTrainingExperimentHandler(ExperimentHandler):

    def __init__(self, report_path: str = REPORT_PATH):
        super().__init__(get_experiment_name(), report_path=report_path)

    def _collect_datasets(self, dataset_name):
        X_train, y_train = [], []
        for folder in range(1, self.n_pred_vectors + 1):
            path = get_dataset_path(dataset_name, folder) / DATASET_FILE_NAME
            logger.info(f"Loading dataset from {path}")
            with open(path, "rb") as f:
                data = pickle.load(f)
                X_train.append(data.train.features)
                y_train.append(data.train.labels)

        return np.vstack(X_train), np.vstack(y_train), data.test.features, data.test.labels



    def run_experiment(self):

        logger.info(f"Training Model Experiment: {get_experiment_name()}")

        for dataset_name in config.dataset_config.names:

            try:
                raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
                logger.debug(f"Original Dataset Size: {raw_dataset.get_dataset()[0].shape}")
                n_classes = raw_dataset.get_n_classes()
                original_size = raw_dataset.get_dataset()[0].shape
                del raw_dataset

            except:
                logger.warning(f"Error loading Dataset {dataset_name}, using default number of classes -> 2")
                n_classes, original_size = 2, 0


            for model_name in config.iim_config.name:

                logger.info(f"#### Training model experiment: "
                            f"Dataset: {dataset_name}, n_pred_vectors: {self.n_pred_vectors} ####\n")

                path = get_dataset_path(dataset_name=dataset_name, n_pred_vectors=self.n_pred_vectors)


                if path.exists():

                    history_path = path / "history.pkl"
                    plot_path = path / f"{model_name}_{config.experiment_config.to_run}_train_plot.png"

                    X_train, y_train, X_test, y_test = self._collect_datasets(dataset_name=dataset_name)

                    for k in tqdm(range(config.experiment_config.k_folds), total=config.experiment_config.k_folds, desc="K Trainings"):

                        internal_model = InternalInferenceModelFactory().get_model(
                            num_classes=n_classes,
                            input_shape=X_train.shape[1],
                            type=model_name
                        )
                        logger.debug(f"#### EVALUATING INTERNAL MODEL {model_name} ####"
                                     f" Dataset Shape: Train - {X_train.shape}, Test: {X_test.shape}")
                        internal_model.fit(
                            X=X_train[:10], y=y_train[:10],
                            validation_data=(X_test[:100], y_test[:100]),
                        )

                        internal_model.save_history(history_path)
                        internal_model.plot_history(plot_path)

                        test_accuracy =  max(internal_model.history.history.get("val_accuracy", []))
                        self.log_k_results(
                            dataset_name=dataset_name,
                            cloud_models_names=str([cloud_model for cloud_model in config.cloud_config.names]),
                            iim_name=model_name,
                            test_accuracy=test_accuracy,
                            k=k
                        )


            del X_train,X_test,y_test, y_train, internal_model
            gc.collect()
            keras.src.backend.common.global_state.clear_session()


        return self.report
