import gc
import os.path
import pickle


import numpy as np
from keras import backend as K
from tqdm import tqdm

from src.internal_model import InternalInferenceModelFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from loguru import logger
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import DATASET_FILE_NAME, REPORT_PATH, OUTPUT_DIR_PATH, IIM_MODELS


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

            report_path = os.path.join(OUTPUT_DIR_PATH, dataset_name, "k_report.csv")
            self.set_report_path(report_path)

            try:
                raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
                logger.debug(f"Original Dataset Size: {raw_dataset.get_dataset()[0].shape}")
                n_classes = raw_dataset.get_n_classes()
                del raw_dataset

            except:
                logger.warning(f"Error loading Dataset {dataset_name}, using default number of classes -> 2")
                n_classes, original_size = 2, 0


            for model_name in config.iim_config.name:
                if model_name == IIM_MODELS.TRANSFORMER and config.experiment_config.triangulation_mode == "cos":
                    logger.warning("#### TRANSFORMER CAN NOT WORK WITH COSINE DATASET - SKIPPING ###")
                    # Transformer can't work with cosine dataset
                    continue

                logger.info(f"#### Training model experiment: "
                            f"Dataset: {dataset_name}, n_pred_vectors: {self.n_pred_vectors} ####\n")

                path = get_dataset_path(dataset_name=dataset_name, n_pred_vectors=self.n_pred_vectors)


                if path.exists():

                    test_accs, test_aucs = [], []
                    history_path = path / "history.pkl"
                    plot_path = path / f"{model_name}_{config.experiment_config.to_run}_train_plot.png"

                    X_train, y_train, X_test, y_test = self._collect_datasets(dataset_name=dataset_name)

                    for _ in tqdm(range(config.experiment_config.k_folds), total=config.experiment_config.k_folds, desc="K Trainings"):

                        internal_model = InternalInferenceModelFactory().get_model(
                            num_classes=n_classes,
                            input_shape=X_train.shape[1],
                            type=model_name,
                            pretrained_t_network_path=config.experiment_config.pretrained_t_network_path,
                            freeze_t_network=config.experiment_config.freeze_t_network
                        )
                        logger.debug(f"#### EVALUATING INTERNAL MODEL {model_name} ####"
                                     f" Dataset Shape: Train - {X_train.shape}, Test: {X_test.shape}")
                        internal_model.fit(
                            X=X_train, y=y_train,
                            validation_data=(X_test, y_test),
                        )

                        internal_model.save_history(history_path)
                        internal_model.plot_history(plot_path)

                        if "val_auc" in internal_model.history.history:
                            test_auc = internal_model.history.history["val_auc"]
                        elif "val_auc_1" in internal_model.history.history:
                            test_auc = internal_model.history.history["val_auc_1"]
                        else:
                            test_auc = []

                        test_aucs.append(
                            round(max(test_auc),
                                  4)
                        )
                        test_accs.append(
                            round(max(internal_model.history.history.get("val_accuracy", []))
                                               , 4))


                    self.log_k_results(
                        dataset_name=dataset_name,
                        cloud_models_names=str([cloud_model for cloud_model in config.cloud_config.names]),
                        iim_name=model_name,
                        k_test_accuracies=test_accs,
                        k_test_aucs=test_aucs
                    )


            del X_train,X_test,y_test, y_train, internal_model
            gc.collect()
            K.clear_session()


        return self.report
