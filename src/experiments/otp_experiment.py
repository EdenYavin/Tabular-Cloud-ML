"""
OTP Experiment Handler.

Runs a two-phase experiment:

Phase 1 – Dataset creation
  For each dataset × n_pred_vectors:
    • Take the raw sparse-autoencoder sample vectors.
    • Apply OTP masking (random blend + zero-pad → image → cloud).
    • Cache the resulting IIMDataset to disk (dataset.pkl).

Phase 2 – Training
  • Stack all n_pred_vectors datasets into one training set.
  • Train the IIM (InternalInferenceModel) on the cloud-response features.
  • Evaluate on the held-out test set.
  • Log accuracy / AUC to the CSV report.

Supports K-Fold Cross Validation when config.experiment_config.k_folds > 1,
identical to the behaviour in FeatureAblationExperimentHandler.
"""

import gc
import os
import pickle
import numpy as np
from tqdm import tqdm
from keras import backend as K
from loguru import logger

from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import (
    DATASET_FILE_NAME,
    BASELINE_DATASET_FILE_NAME,
    REPORT_PATH,
    OUTPUT_DIR_PATH,
)
from src.pipeline.otp_features_dataset import OTPFeatureEngineering
from src.encryptor import EncryptorFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory, KFoldSplitDBFactory
from src.cloud import CLOUD_MODELS, DEFAULT_CLOUD_OUTPUT_SHAPE
from src.internal_model import InternalInferenceModelFactory


class OTPExperimentHandler(ExperimentHandler):
    """
    Experiment handler for the OTP (One-Time Pad) dataset experiment.

    Each raw sample is blended with a fresh random vector before being
    sent to the cloud.  The cloud's response trains the IIM instead of
    the original features, providing a simple but effective privacy screen.

    Configuration is taken entirely from the global ``config`` object so
    the handler can be driven from the CLI via ``main.py`` like all other
    experiment handlers.
    """

    def __init__(self, report_path: str = REPORT_PATH):
        super().__init__(get_experiment_name(), report_path=report_path)
        logger.info("### OTP EXPERIMENT HANDLER ###")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self, dataset_name: str, raw_dataset: RawDataset):
        """Instantiate OTPFeatureEngineering with the right encryptor/embeddings."""
        if config.cloud_config.names:
            cloud_model_output = CLOUD_MODELS[config.cloud_config.names[0]].input_shape
        else:
            cloud_model_output = DEFAULT_CLOUD_OUTPUT_SHAPE

        embedding_model = EmbeddingsFactory().get_model(
            X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name
        )
        encryptor = EncryptorFactory.get_model(
            dataset_name=dataset_name, output_shape=cloud_model_output
        )
        return OTPFeatureEngineering(
            dataset_name=dataset_name,
            encryptor=encryptor,
            embeddings_model=embedding_model,
            metadata=raw_dataset.metadata,
        )

    def _get_dataset_path(self, dataset_name: str, n_pred: int, fold_idx=None):
        """Thin wrapper so we can pass fold_idx cleanly."""
        return get_dataset_path(
            dataset_name,
            n_pred,
            feature_combination="otp",       # keeps OTP cache separate
            fold_idx=fold_idx,
        )

    def _create_dataset_if_missing(
        self,
        dataset_name: str,
        n_pred: int,
        X_sample,
        y_sample,
        X_test,
        y_test,
        fold_idx=None,
    ):
        """
        Create (or load from cache) the OTP IIMDataset for one
        (n_pred, fold) pair.

        Returns:
            Loaded IIMDataset object.
        """
        path = self._get_dataset_path(dataset_name, n_pred, fold_idx)
        dataset_file = path / DATASET_FILE_NAME

        if dataset_file.exists() and config.dataset_config.use_cache:
            logger.info(f"[OTP] Cache hit — loading: {dataset_file}")
            with open(dataset_file, "rb") as f:
                return pickle.load(f)

        logger.info(f"[OTP] Generating dataset → {dataset_file}")

        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        pipeline = self._build_pipeline(dataset_name, raw_dataset)

        dataset, emb_baseline_dataset = pipeline.create(
            X_sample, y_sample, X_test, y_test
        )

        os.makedirs(path, exist_ok=True)
        with open(path / BASELINE_DATASET_FILE_NAME, "wb") as f:
            pickle.dump(emb_baseline_dataset, f)
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)

        del raw_dataset, pipeline, emb_baseline_dataset
        gc.collect()
        return dataset

    # ------------------------------------------------------------------
    # Data collection (single-split and K-fold variants)
    # ------------------------------------------------------------------

    def _collect_datasets(self, dataset_name: str):
        """Single fixed-split data collection (k_folds == 1)."""
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        _, X_test_raw, X_sample_raw, _, y_test_raw, y_sample_raw = (
            RawSplitDBFactory.get_db(raw_dataset).get_split()
        )
        del raw_dataset

        X_train_parts, y_train_parts = [], []
        X_test = y_test = None

        for n_pred in tqdm(
            range(1, self.n_pred_vectors + 1),
            desc=f"[OTP] Loading datasets for {dataset_name}",
            position=0,
            leave=True,
        ):
            data = self._create_dataset_if_missing(
                dataset_name=dataset_name,
                n_pred=n_pred,
                X_sample=X_sample_raw,
                y_sample=y_sample_raw,
                X_test=X_test_raw,
                y_test=y_test_raw,
            )
            X_train_parts.append(data.train.features)
            y_train_parts.append(data.train.labels)
            X_test = data.test.features
            y_test = data.test.labels
            del data
            gc.collect()

        return (
            np.vstack(X_train_parts),
            np.vstack(y_train_parts),
            X_test,
            y_test,
        )

    def _collect_datasets_for_fold(self, dataset_name: str, fold_idx: int):
        """K-Fold data collection for one held-out fold."""
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        k_fold_db = KFoldSplitDBFactory.get_db(
            raw_dataset, n_splits=config.experiment_config.k_folds
        )
        X_sample_raw, X_test_raw, y_sample_raw, y_test_raw = k_fold_db.get_fold(fold_idx)
        del raw_dataset

        X_train_parts, y_train_parts = [], []

        for n_pred in tqdm(
            range(1, self.n_pred_vectors + 1),
            desc=f"[OTP] Fold {fold_idx} | {dataset_name}",
            position=0,
            leave=True,
        ):
            data = self._create_dataset_if_missing(
                dataset_name=dataset_name,
                n_pred=n_pred,
                X_sample=X_sample_raw,
                y_sample=y_sample_raw,
                X_test=X_test_raw,
                y_test=y_test_raw,
                fold_idx=fold_idx,
            )
            X_train_parts.append(data.train.features)
            y_train_parts.append(data.train.labels)
            X_test = data.test.features
            y_test = data.test.labels
            del data
            gc.collect()

        return (
            np.vstack(X_train_parts),
            np.vstack(y_train_parts),
            X_test,
            y_test,
        )

    # ------------------------------------------------------------------
    # Core training runner
    # ------------------------------------------------------------------

    def _run_training(self, dataset_name: str, n_classes: int):
        """
        Train the IIM on OTP features.

        Supports single split (k_folds == 1) and true K-Fold CV.

        Returns:
            Last fold's test_metrics dict.
        """
        k_folds = config.experiment_config.k_folds
        use_kfold = k_folds > 1

        logger.info(
            f"\n{'='*80}\n"
            f"OTP Experiment | Dataset: {dataset_name}\n"
            f"n_pred_vectors: {self.n_pred_vectors}  |  "
            f"{'True K-Fold CV (' + str(k_folds) + ' folds)' if use_kfold else 'Single split'}\n"
            f"{'='*80}"
        )

        for model_name in config.iim_config.name:

            test_accs: list[float] = []
            test_aucs: list[float] = []
            test_metrics = {}

            fold_iter = range(k_folds) if use_kfold else [None]

            for fold_idx in tqdm(
                fold_iter,
                total=k_folds,
                desc=f"[OTP] {'K-fold' if use_kfold else 'Training'}: {dataset_name}",
                position=0,
                leave=True,
            ):
                if use_kfold:
                    logger.info(f"[OTP] Fold {fold_idx + 1}/{k_folds}")
                    X_train, y_train, X_test, y_test = self._collect_datasets_for_fold(
                        dataset_name=dataset_name, fold_idx=fold_idx
                    )
                else:
                    X_train, y_train, X_test, y_test = self._collect_datasets(
                        dataset_name=dataset_name
                    )

                logger.info(
                    f"[OTP] Data ready — Train: {X_train.shape}, Test: {X_test.shape}"
                )

                # Train
                internal_model = InternalInferenceModelFactory().get_model(
                    num_classes=n_classes,
                    input_shape=X_train.shape[1],
                    type=model_name,
                )
                internal_model.fit(
                    X=X_train,
                    y=y_train,
                    validation_data=(X_test, y_test),
                )

                # Evaluate
                test_metrics = internal_model.evaluate(
                    X=X_test, y=y_test, metrics=config.iim_config.metrics
                )

                fold_label = f"Fold {fold_idx}" if use_kfold else "Run"
                logger.info(f"[OTP] {fold_label} | metrics: {test_metrics}")

                test_acc = test_metrics.get(
                    "test_accuracy", test_metrics.get("accuracy", 0.0)
                )
                test_auc = test_metrics.get(
                    "test_auc", test_metrics.get("auc", 0.0)
                )
                test_accs.append(round(float(test_acc), 4))
                test_aucs.append(round(float(test_auc), 4))

                del X_train, y_train, X_test, y_test
                gc.collect()
                K.clear_session()

            # Save history / plot
            path = self._get_dataset_path(dataset_name, self.n_pred_vectors)
            internal_model.save_history(path / "otp_history.pkl")
            internal_model.plot_history(path / "otp_train_plot.png")

            self.log_k_results(
                dataset_name=dataset_name,
                cloud_models_names=str(config.cloud_config.names),
                iim_name=f"{model_name}_otp",
                k_test_accuracies=test_accs,
                k_test_aucs=test_aucs,
            )

            del internal_model
            gc.collect()
            K.clear_session()

        return test_metrics

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_experiment(self):
        """
        Execute the OTP experiment for all datasets in config.

        For each dataset:
          1. Load metadata (n_classes).
          2. Run OTP dataset creation + IIM training.
          3. Save results to the CSV report.
        """
        logger.info(
            f"### STARTING OTP EXPERIMENT ###\n"
            f"Datasets: {config.dataset_config.names}\n"
            f"K-Folds: {config.experiment_config.k_folds}"
        )

        for dataset_name in config.dataset_config.names:
            logger.info(f"\n{'#'*80}\n# Dataset: {dataset_name.upper()}\n{'#'*80}")

            report_path = os.path.join(
                OUTPUT_DIR_PATH, "otp", dataset_name, "k_report.csv"
            )
            self.set_report_path(report_path)

            try:
                raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
                n_classes = raw_dataset.get_n_classes()
                del raw_dataset
            except Exception as e:
                logger.warning(
                    f"[OTP] Could not load {dataset_name}: {e} — defaulting to 2 classes"
                )
                n_classes = 2

            try:
                self._run_training(dataset_name=dataset_name, n_classes=n_classes)
                logger.info(f"[OTP] ✓ Completed {dataset_name}")
            except Exception as e:
                logger.error(f"[OTP] ✗ Error on {dataset_name}: {e}")
                continue

            self.save()

        logger.info(
            f"\n{'#'*80}\n"
            f"# OTP EXPERIMENT COMPLETE\n"
            f"# Results saved to: {self.report_path}\n"
            f"{'#'*80}"
        )
        return self.report
