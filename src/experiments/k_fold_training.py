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
from src.utils.helpers import get_t_network_model_path


class KFoldTrainingExperimentHandler(ExperimentHandler):

    def __init__(self, report_path: str = REPORT_PATH):
        super().__init__(get_experiment_name(), report_path=report_path)

    def _collect_datasets(self, dataset_name, fold_idx):
        """Load and concatenate datasets for a specific fold."""
        X_train, y_train = [], []
        data = None
        for folder in range(1, self.n_pred_vectors + 1):
            path = get_dataset_path(dataset_name, folder, fold_idx=fold_idx) / DATASET_FILE_NAME
            logger.info(f"Loading dataset from {path}")
            with open(path, "rb") as f:
                data = pickle.load(f)
                X_train.append(data.train.features)
                y_train.append(data.train.labels)

        return np.vstack(X_train), np.vstack(y_train), data.test.features, data.test.labels

    def run_experiment(self):
        """
        Classic K-Fold Cross Validation: load each fold ONCE, train once per fold,
        then aggregate results across all K folds.
        """

        k_folds = config.experiment_config.k_folds
        if k_folds <= 1:
            raise ValueError(
                f"K-Fold training requires k_folds > 1, got {k_folds}. "
                f"Use model_training experiment for single-split training."
            )

        logger.info(f"K-Fold Training Experiment: {get_experiment_name()} with {k_folds} folds")

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
                n_classes = 2

            for model_name in config.iim_config.name:
                if model_name == IIM_MODELS.TRANSFORMER and config.experiment_config.triangulation_mode == "cos":
                    logger.warning("#### TRANSFORMER CAN NOT WORK WITH COSINE DATASET - SKIPPING ###")
                    continue

                logger.info(f"#### K-Fold training: "
                            f"Dataset: {dataset_name}, n_pred_vectors: {self.n_pred_vectors}, "
                            f"k_folds: {k_folds} ####\n")

                path = get_dataset_path(dataset_name=dataset_name, n_pred_vectors=self.n_pred_vectors)

                if path.exists():

                    test_accs, test_aucs = [], []

                    # Train once per fold — classic K-Fold CV
                    for fold_idx in tqdm(range(k_folds), total=k_folds, desc="K-Fold CV"):

                        X_train, y_train, X_test, y_test = self._collect_datasets(
                            dataset_name=dataset_name, fold_idx=fold_idx
                        )

                        history_path = path / f"fold_{fold_idx}" / "history.pkl"
                        plot_path = path / f"fold_{fold_idx}" / f"{model_name}_{config.experiment_config.to_run}_train_plot.png"

                        # Auto-determine T-Network path if freeze is enabled but path not specified
                        pretrained_path = config.experiment_config.pretrained_t_network_path
                        if config.experiment_config.freeze_t_network and not pretrained_path:
                            try:
                                pretrained_path = get_t_network_model_path(
                                    dataset_name=dataset_name,
                                    ensure_exists=True
                                )
                                logger.info(f"Auto-determined T-Network path: {pretrained_path}")
                            except FileNotFoundError as e:
                                logger.error(str(e))
                                raise

                        internal_model = InternalInferenceModelFactory().get_model(
                            num_classes=n_classes,
                            input_shape=X_train.shape[1],
                            type=model_name,
                            pretrained_t_network_path=str(pretrained_path) if pretrained_path else None,
                            freeze_t_network=config.experiment_config.freeze_t_network
                        )
                        logger.debug(f"#### EVALUATING INTERNAL MODEL {model_name} (Fold {fold_idx}) ####"
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
                            test_auc = [0.0]

                        test_val_accs = internal_model.history.history.get("val_accuracy", [0.0])

                        test_aucs.append(
                            round(float(np.max(test_auc)), 4)
                        )
                        test_accs.append(
                            round(float(np.max(test_val_accs)), 4)
                        )

                        # Clean up fold memory
                        del X_train, y_train, X_test, y_test, internal_model
                        gc.collect()
                        K.clear_session()

                    self.log_k_results(
                        dataset_name=dataset_name,
                        cloud_models_names=str([cloud_model for cloud_model in config.cloud_config.names]),
                        iim_name=model_name,
                        k_test_accuracies=test_accs,
                        k_test_aucs=test_aucs
                    )

            gc.collect()
            K.clear_session()

        return self.report
