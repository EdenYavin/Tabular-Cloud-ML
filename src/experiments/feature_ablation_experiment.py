"""
Feature Ablation Experiment Handler.

Orchestrates batch training runs for all 4 feature combinations
using FlexibleSINClassifier to evaluate the relative importance of:
- p_x: Raw sparse autoencoder sample embeddings
- q_x: Encrypted sample embeddings (DINO/CLIP)
- T_context: Frozen T network encoder outputs
- cloud: Cloud model predictions

When config.experiment_config.k_folds > 1, runs TRUE K-Fold Cross Validation:
  - The full dataset is split into K stratified folds (cached to disk)
  - For each fold k: train on K-1 folds, evaluate on fold k
  - Metrics are aggregated across all K held-out test sets

When k_folds == 1, falls back to the original single-split behaviour.
"""

import gc
import pickle
import numpy as np
import os
from tqdm import tqdm
from keras import backend as K
from loguru import logger

from src.internal_model import InternalInferenceModelFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import (
    DATASET_FILE_NAME,
    BASELINE_DATASET_FILE_NAME,
    REPORT_PATH,
    FEATURE_COMBINATIONS,
    IIM_MODELS, OUTPUT_DIR_PATH
)
from src.pipeline.feature_ablation_dataset import FeatureAblationPipeline
from src.encryptor import EncryptorFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory, KFoldSplitDBFactory
from src.cloud import CLOUD_MODELS, DEFAULT_CLOUD_OUTPUT_SHAPE


class FeatureAblationExperimentHandler(ExperimentHandler):
    """
    Batch experiment runner for feature ablation study.

    Executes training for all 4 feature combinations sequentially:
    1. baseline_no_cloud: [p_x, q_x, T_context] - Baseline with all features except cloud
    2. no_raw_embedding: [q_x, T_context] - Remove p_x to test raw embedding importance
    3. full_features: [p_x, q_x, T_context, cloud] - Full feature set
    4. cloud_no_raw: [q_x, T_context, cloud] - Cloud features without raw embeddings

    Each combination uses:
    - Identical train/val/test splits from cached dataset files
    - Same FlexibleSINClassifier architecture
    - Same training hyperparameters
    - Same evaluation metrics (accuracy, AUC)

    When k_folds > 1 in config, performs TRUE K-Fold Cross Validation:
    Each fold uses a *different* held-out test set (1/K of the data) and
    the remaining K-1 folds as training data.

    Results are logged to CSV report for side-by-side comparison.
    """

    def __init__(self, report_path=REPORT_PATH):
        """
        Initialize Feature Ablation Experiment Handler.

        Args:
            report_path: Path to CSV report file for results logging
        """
        super().__init__(get_experiment_name(), report_path=report_path)

        logger.info("### FEATURE ABLATION EXPERIMENT HANDLER ###")
        logger.info(f"Report path: {report_path}")

        self.combinations = [
            # FEATURE_COMBINATIONS.BASELINE_NO_CLOUD,
            FEATURE_COMBINATIONS.NO_RAW_EMBEDDING,
            # FEATURE_COMBINATIONS.FULL_FEATURES,
            # FEATURE_COMBINATIONS.CLOUD_NO_RAW
        ]

    # -------------------------------------------------------------------------
    # Dataset helpers
    # -------------------------------------------------------------------------

    def _build_pipeline_inputs(self, dataset_name) -> tuple:
        """
        Initialise and return the objects needed by FeatureAblationPipeline.

        Returns:
            (raw_dataset, embedding_model, encryptor, cloud_model_output)
        """
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)

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

        return raw_dataset, embedding_model, encryptor, cloud_model_output

    def _create_dataset_if_missing(
        self,
        dataset_name: str,
        feature_combination: str,
        n_pred: int,
        X_sample,
        y_sample,
        X_test,
        y_test,
        fold_idx: int = None,
    ):
        """
        Create (or load from cache) the processed IIM dataset for one
        (n_pred_vector, feature_combination, fold) triplet.

        The dataset is cached under a path that includes
        `fold_{fold_idx}/` as the *last* directory segment when fold_idx
        is provided, so per-fold caches never collide.

        Args:
            dataset_name:        Dataset name (e.g. 'mushroom')
            feature_combination: One of the FEATURE_COMBINATIONS values
            n_pred:              Prediction-vector index (1-based)
            X_sample:            Raw feature array for pipeline training
            y_sample:            Label array for pipeline training
            X_test:              Raw feature array for evaluation
            y_test:              Label array for evaluation
            fold_idx:            Fold index (None → original single-split mode)

        Returns:
            Loaded IIMDataset object
        """
        fold_path = get_dataset_path(
            dataset_name,
            n_pred,
            feature_combination=feature_combination,
            fold_idx=fold_idx,
        )

        if (fold_path / DATASET_FILE_NAME).exists():
            logger.info(
                f"Dataset already cached — loading: {fold_path / DATASET_FILE_NAME}"
            )
            with open(fold_path / DATASET_FILE_NAME, "rb") as f:
                return pickle.load(f)

        logger.info(f"Generating dataset → {fold_path / DATASET_FILE_NAME}")

        # Initialise pipeline
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
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

        dataset_creator = FeatureAblationPipeline(
            dataset_name=dataset_name,
            encryptor=encryptor,
            embeddings_model=embedding_model,
            feature_combination=feature_combination,
            metadata=raw_dataset.metadata,
        )

        dataset, emb_baseline_dataset = dataset_creator.create(
            X_sample, y_sample, X_test, y_test
        )

        # Persist to disk
        os.makedirs(fold_path, exist_ok=True)

        with open(fold_path / BASELINE_DATASET_FILE_NAME, "wb") as f:
            pickle.dump(emb_baseline_dataset, f)

        with open(fold_path / DATASET_FILE_NAME, "wb") as f:
            logger.debug(f"Saving dataset → {fold_path / DATASET_FILE_NAME}")
            pickle.dump(dataset, f)

        del raw_dataset, embedding_model, encryptor, dataset_creator, emb_baseline_dataset
        gc.collect()
        return dataset

    # -------------------------------------------------------------------------
    # Original single-split dataset loading (k_folds == 1)
    # -------------------------------------------------------------------------

    def _collect_datasets(self, dataset_name: str, feature_combination: str):
        """
        Load/create cached datasets for a specific feature combination
        using the *original* single fixed split.

        Used only when ``config.experiment_config.k_folds == 1``.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        X_train_raw, X_test_raw, X_sample_raw, y_train_raw, y_test_raw, y_sample_raw = (
            RawSplitDBFactory.get_db(raw_dataset).get_split()
        )
        del raw_dataset

        X_train, y_train = [], []
        X_test, y_test = None, None

        for folder in tqdm(
            range(1, self.n_pred_vectors + 1),
            desc=f"Loading {feature_combination} datasets",
            position=0,
            leave=True,
        ):
            data = self._create_dataset_if_missing(
                dataset_name=dataset_name,
                feature_combination=feature_combination,
                n_pred=folder,
                X_sample=X_sample_raw,
                y_sample=y_sample_raw,
                X_test=X_test_raw,
                y_test=y_test_raw,
                fold_idx=None,
            )

            X_train.append(data.train.features)
            y_train.append(data.train.labels)
            X_test = data.test.features
            y_test = data.test.labels
            del data
            gc.collect()

        return np.vstack(X_train), np.vstack(y_train), X_test, y_test

    # -------------------------------------------------------------------------
    # True K-Fold dataset loading (k_folds > 1)
    # -------------------------------------------------------------------------

    def _collect_datasets_for_fold(
        self,
        dataset_name: str,
        feature_combination: str,
        fold_idx: int,
    ):
        """
        Load/create cached datasets for *one* K-fold cross-validation fold.

        For fold ``fold_idx``:
        - ``X_sample / y_sample`` = the K-1 training folds fed through the
          FeatureAblationPipeline to produce IIM training features.
        - ``X_test / y_test``    = the held-out fold used only for evaluation.

        Processed outputs are cached under ``…/fold_{fold_idx}/`` so that
        repeated runs skip the expensive pipeline step.

        Args:
            dataset_name:        Dataset name
            feature_combination: Feature combination string
            fold_idx:            Which of the K folds is the held-out test set

        Returns:
            Tuple of (X_train_stacked, y_train_stacked, X_test, y_test)
        """
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        k_fold_db = KFoldSplitDBFactory.get_db(
            raw_dataset, n_splits=config.experiment_config.k_folds
        )
        # X_sample = training portion (K-1 folds), X_test = held-out fold k
        X_sample_raw, X_test_raw, y_sample_raw, y_test_raw = k_fold_db.get_fold(fold_idx)
        del raw_dataset

        X_train_parts, y_train_parts = [], []

        for n_pred in tqdm(
            range(1, self.n_pred_vectors + 1),
            desc=f"Fold {fold_idx} | {feature_combination}",
            position=0,
            leave=True,
        ):
            data = self._create_dataset_if_missing(
                dataset_name=dataset_name,
                feature_combination=feature_combination,
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

    # -------------------------------------------------------------------------
    # Core experiment runner
    # -------------------------------------------------------------------------

    def _run_single_combination(
        self,
        dataset_name: str,
        feature_combination: str,
        n_classes: int,
        original_size: tuple,
    ):
        """
        Train FlexibleSINClassifier on a single feature combination.

        When ``config.experiment_config.k_folds == 1``:
            Uses the original fixed train/test split (legacy behaviour).

        When ``config.experiment_config.k_folds > 1``:
            Performs TRUE K-Fold Cross Validation — each fold uses a
            *different* held-out test set; metrics are aggregated across all K
            held-out evaluations.

        Args:
            dataset_name:        Dataset name
            feature_combination: Feature combination enum value string
            n_classes:           Number of target classes
            original_size:       Original dataset shape (for logging)

        Returns:
            Last fold's test_metrics dict
        """
        k_folds = config.experiment_config.k_folds
        use_true_kfold = k_folds > 1

        logger.info(
            f"\n{'='*80}\n"
            f"Running Feature Ablation: {feature_combination.upper()}\n"
            f"Dataset: {dataset_name}  |  n_pred_vectors: {self.n_pred_vectors}  |  "
            f"{'True K-Fold CV (' + str(k_folds) + ' folds)' if use_true_kfold else 'Single split'}\n"
            f"{'='*80}"
        )

        for model_name in config.iim_config.name:

            test_accs: list[float] = []
            test_aucs: list[float] = []
            test_metrics = {}

            fold_iter = range(k_folds) if use_true_kfold else [None]

            for fold_idx in tqdm(
                fold_iter,
                total=k_folds,
                desc=f"{'K-fold' if use_true_kfold else 'Training'}: {feature_combination}",
                position=0,
                leave=True,
            ):
                # ---- Load / create the (fold-specific) processed dataset ----
                if use_true_kfold:
                    logger.info(
                        f"K-Fold {fold_idx + 1}/{k_folds} — "
                        f"held-out fold: {fold_idx}"
                    )
                    X_train, y_train, X_test, y_test = self._collect_datasets_for_fold(
                        dataset_name=dataset_name,
                        feature_combination=feature_combination,
                        fold_idx=fold_idx,
                    )
                else:
                    X_train, y_train, X_test, y_test = self._collect_datasets(
                        dataset_name=dataset_name,
                        feature_combination=feature_combination,
                    )

                logger.info(
                    f"{'Fold ' + str(fold_idx) + ' ' if use_true_kfold else ''}"
                    f"Data loaded — Train: {X_train.shape}, Test: {X_test.shape}"
                )

                # ---- Train ----
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

                # ---- Evaluate ----
                test_metrics = internal_model.evaluate(
                    X=X_test, y=y_test, metrics=config.iim_config.metrics
                )

                fold_label = f"Fold {fold_idx}" if use_true_kfold else "Run"
                logger.info(
                    f"{feature_combination} | {fold_label} | Test Metrics: {test_metrics}"
                )

                test_acc = test_metrics.get(
                    "test_accuracy", test_metrics.get("accuracy", 0.0)
                )
                test_auc = test_metrics.get(
                    "test_auc", test_metrics.get("auc", 0.0)
                )

                test_accs.append(round(float(test_acc), 4))
                test_aucs.append(round(float(test_auc), 4))

                logger.debug(
                    f"{'Fold ' + str(fold_idx) if use_true_kfold else 'Run'}: "
                    f"acc={test_acc:.4f}, auc={test_auc:.4f}"
                )

                # Per-fold cleanup
                del X_train, y_train, X_test, y_test
                gc.collect()
                K.clear_session()

            # ---- Save history / plot using the last fold's model ----
            path = get_dataset_path(
                dataset_name=dataset_name,
                n_pred_vectors=self.n_pred_vectors,
                feature_combination=feature_combination,
            )
            history_path = path / f"{feature_combination}_history.pkl"
            plot_path = path / f"{feature_combination}_train_plot.png"

            internal_model.save_history(history_path)
            internal_model.plot_history(plot_path)

            # ---- Log aggregated results ----
            self.log_k_results(
                dataset_name=dataset_name,
                cloud_models_names=str(
                    [cloud_model for cloud_model in config.cloud_config.names]
                ),
                iim_name=f"{model_name}_{feature_combination}",
                k_test_accuracies=test_accs,
                k_test_aucs=test_aucs,
            )

            del internal_model
            gc.collect()
            K.clear_session()

        return test_metrics

    def run_experiment(self):
        """
        Execute feature ablation experiment for all combinations.

        For each dataset in config:
        1. Load dataset metadata (n_classes, original size)
        2. Run training for each active feature combination
        3. Log all results to CSV report

        Results are automatically saved after each combination via log_results().

        Returns:
            DataFrame with experiment results for all combinations
        """
        logger.info(
            f"### STARTING FEATURE ABLATION EXPERIMENT ###\n"
            f"Experiment: {get_experiment_name()}\n"
            f"Datasets: {config.dataset_config.names}\n"
            f"Combinations: {[combo.value for combo in self.combinations]}\n"
            f"K-Folds: {config.experiment_config.k_folds} "
            f"({'true K-fold CV' if config.experiment_config.k_folds > 1 else 'single split'})"
        )

        for dataset_name in config.dataset_config.names:
            logger.info(f"\n{'#'*80}\n# Dataset: {dataset_name.upper()}\n{'#'*80}")

            report_path = os.path.join(
                OUTPUT_DIR_PATH, "ablation", dataset_name, "k_report.csv"
            )
            self.set_report_path(report_path)
            logger.info(f"Report path set to {report_path}")

            try:
                raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
                logger.debug(f"Original Dataset Size: {raw_dataset.get_dataset()[0].shape}")
                n_classes = raw_dataset.get_n_classes()
                original_size = raw_dataset.get_dataset()[0].shape
                del raw_dataset
            except Exception as e:
                logger.warning(
                    f"Error loading Dataset {dataset_name}: {e}\n"
                    f"Using default number of classes -> 2"
                )
                n_classes, original_size = 2, (0, 0)

            for feature_combination in self.combinations:
                try:
                    self._run_single_combination(
                        dataset_name=dataset_name,
                        feature_combination=feature_combination.value,
                        n_classes=n_classes,
                        original_size=original_size,
                    )

                    logger.info(
                        f"✓ Completed {feature_combination.value} for {dataset_name}"
                    )

                except FileNotFoundError as e:
                    logger.error(f"✗ Skipping {feature_combination.value} — {e}")
                    continue

                except Exception as e:
                    logger.error(
                        f"✗ Error running {feature_combination.value} "
                        f"for {dataset_name}: {e}"
                    )
                    continue

                logger.info(f"Saving report for {feature_combination.value}")
                self.save()

        logger.info(
            f"\n{'#'*80}\n"
            f"# FEATURE ABLATION EXPERIMENT COMPLETE\n"
            f"# Results saved to: {self.report_path}\n"
            f"{'#'*80}"
        )

        return self.report
