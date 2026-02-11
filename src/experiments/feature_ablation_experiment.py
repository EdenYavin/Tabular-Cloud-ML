"""
Feature Ablation Experiment Handler.

Orchestrates batch training runs for all 4 feature combinations
using FlexibleSINClassifier to evaluate the relative importance of:
- p_x: Raw sparse autoencoder sample embeddings
- q_x: Encrypted sample embeddings (DINO/CLIP)
- T_context: Frozen T network encoder outputs
- cloud: Cloud model predictions

Runs 4 sequential experiments and generates side-by-side comparison report.
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
    IIM_MODELS
)
from src.pipeline.feature_ablation_dataset import FeatureAblationPipeline
from src.encryptor import EncryptorFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory
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

        # Validate all combinations are cacheable
        self.combinations = [
            FEATURE_COMBINATIONS.BASELINE_NO_CLOUD,
            FEATURE_COMBINATIONS.NO_RAW_EMBEDDING,
            FEATURE_COMBINATIONS.FULL_FEATURES,
            FEATURE_COMBINATIONS.CLOUD_NO_RAW
        ]


    def _create_dataset_if_missing(self, dataset_name, feature_combination, n_pred):

        # 1. Load Raw Dataset & Splits
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        
        # Get Cloud Output Shape
        if config.cloud_config.names:
            cloud_model_output = CLOUD_MODELS[config.cloud_config.names[0]].input_shape
        else:
            cloud_model_output = DEFAULT_CLOUD_OUTPUT_SHAPE

        # Initialize Factories
        embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name)
        encryptor = EncryptorFactory.get_model(dataset_name=dataset_name, output_shape=cloud_model_output)

        # Get Split (Train/Test/Sample)
        X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
        
        # Initialize Pipeline
        dataset_creator = FeatureAblationPipeline(
            dataset_name=dataset_name,
            encryptor=encryptor,
            embeddings_model=embedding_model,
            metadata=raw_dataset.metadata
        )

        # Check if this specific fold already exists
        fold_path = get_dataset_path(
            dataset_name,
            n_pred,
            feature_combination=feature_combination
        )

        if (fold_path / DATASET_FILE_NAME).exists():
            logger.info(f"Dataset for {feature_combination} already exists, skipping creation")
            with open(fold_path / DATASET_FILE_NAME, "rb") as f:
                return pickle.load(f)

        logger.info(f"Generating fold {fold_path / DATASET_FILE_NAME} dataset...")

        # Create Dataset
        dataset, emb_baseline_dataset = dataset_creator.create(X_sample, y_sample, X_test, y_test)

        # Save to Disk
        os.makedirs(fold_path, exist_ok=True)

        with open(fold_path / BASELINE_DATASET_FILE_NAME, "wb") as f:
            pickle.dump(emb_baseline_dataset, f)

        with open(fold_path / DATASET_FILE_NAME, "wb") as f:
            logger.debug(f"Saving dataset to {fold_path / DATASET_FILE_NAME}")
            pickle.dump(dataset, f)

        del raw_dataset, embedding_model, encryptor, dataset_creator, emb_baseline_dataset
        gc.collect()
        return dataset


    def _collect_datasets(self, dataset_name, feature_combination):
        """
        Load cached datasets for a specific feature combination.

        Args:
            dataset_name: Dataset name (e.g., 'adult', 'heloc')
            feature_combination: Feature combination enum (baseline_no_cloud, no_raw_embedding, full_features, cloud_no_raw)

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)

        """

        X_train, y_train = [], []
        X_test, y_test = None, None

        for folder in tqdm(range(1, self.n_pred_vectors + 1), desc=f"Loading {feature_combination} datasets", position=0, leave=True):

            data = self._create_dataset_if_missing(dataset_name, feature_combination, folder)

            X_train.append(data.train.features)
            y_train.append(data.train.labels)
            X_test = data.test.features
            y_test = data.test.labels
            del data
            gc.collect()

        return np.vstack(X_train), np.vstack(y_train), X_test, y_test

    def _run_single_combination(self, dataset_name, feature_combination, n_classes, original_size):
        """
        Train FlexibleSINClassifier on a single feature combination.

        Args:
            dataset_name: Dataset name
            feature_combination: Feature combination enum (baseline_no_cloud, no_raw_embedding, full_features, cloud_no_raw)
            n_classes: Number of target classes
            original_size: Original dataset shape (for logging)

        Returns:
            Dictionary of test metrics

        Raises:
            FileNotFoundError: If cached dataset not found
        """
        logger.info(
            f"\n{'='*80}\n"
            f"Running Feature Ablation: {feature_combination.upper()}\n"
            f"Dataset: {dataset_name}, n_pred_vectors: {self.n_pred_vectors}\n"
            f"{'='*80}"
        )

        # Load cached datasets for this combination
        X_train, y_train, X_test, y_test = self._collect_datasets(
            dataset_name=dataset_name,
            feature_combination=feature_combination
        )

        logger.info(
            f"Loaded data for {feature_combination}: "
            f"Train: {X_train.shape}, Test: {X_test.shape}"
        )

        # Initialize metric collectors for K-fold training
        test_accs = []
        test_aucs = []

        logger.info(
            f"Running {config.experiment_config.k_folds} training iterations for {feature_combination}"
        )

        for k_iter in tqdm(
            range(config.experiment_config.k_folds),
            total=config.experiment_config.k_folds,
            desc=f"K-fold training: {feature_combination}",
            position=1,
            leave=False
        ):
            logger.debug(f"K-fold iteration {k_iter + 1}/{config.experiment_config.k_folds}")

            # Create FlexibleSINClassifier - automatically adapts to input dimensions
            internal_model = InternalInferenceModelFactory().get_model(
                num_classes=n_classes,
                input_shape=X_train.shape[1],
                type=IIM_MODELS.FLEXIBLE_SIN
            )

            logger.info(
                f"Training FlexibleSINClassifier for {feature_combination} (iteration {k_iter + 1})\n"
                f"Input shape: {X_train.shape[1]} dims, Output classes: {n_classes}"
            )

            # Train model
            internal_model.fit(
                X=X_train, y=y_train,
                validation_data=(X_test, y_test)
            )

            # Evaluate on test set
            test_metrics = internal_model.evaluate(
                X=X_test, y=y_test, metrics=config.iim_config.metrics
            )

            logger.info(
                f"{feature_combination} K-fold {k_iter + 1} Test Metrics: {test_metrics}"
            )

            # Extract accuracy and AUC from test_metrics dict
            test_acc = test_metrics.get("test_accuracy", test_metrics.get("accuracy", 0.0))
            test_auc = test_metrics.get("test_auc", test_metrics.get("auc", 0.0))

            test_accs.append(round(float(test_acc), 4))
            test_aucs.append(round(float(test_auc), 4))

            logger.debug(f"K-fold {k_iter + 1}: acc={test_acc:.4f}, auc={test_auc:.4f}")

        # Save training history and plots (using last iteration's model)
        path = get_dataset_path(
            dataset_name=dataset_name,
            n_pred_vectors=self.n_pred_vectors,
            feature_combination=feature_combination
        )
        history_path = path / f"{feature_combination}_history.pkl"
        plot_path = path / f"{feature_combination}_train_plot.png"

        internal_model.save_history(history_path)
        internal_model.plot_history(plot_path)

        # Log K-fold results to report
        self.log_k_results(
            dataset_name=dataset_name,
            cloud_models_names=str([cloud_model for cloud_model in config.cloud_config.names]),
            iim_name=f"flexible_sin_{feature_combination}",
            k_test_accuracies=test_accs,
            k_test_aucs=test_aucs
        )

        # Clean up memory
        del X_train, X_test, y_test, y_train, internal_model
        gc.collect()
        K.clear_session()

        return test_metrics

    def run_experiment(self):
        """
        Execute feature ablation experiment for all combinations.

        For each dataset in config:
        1. Load dataset metadata (n_classes, original size)
        2. Run training for baseline_no_cloud (baseline without cloud)
        3. Run training for no_raw_embedding (remove p_x)
        4. Run training for full_features (full feature set)
        5. Run training for cloud_no_raw (cloud without p_x)
        6. Log all results to CSV report

        Results are automatically saved after each combination via log_results().

        Returns:
            DataFrame with experiment results for all combinations
        """
        logger.info(
            f"### STARTING FEATURE ABLATION EXPERIMENT ###\n"
            f"Experiment: {get_experiment_name()}\n"
            f"Datasets: {config.dataset_config.names}\n"
            f"Combinations: {[combo.value for combo in self.combinations]}"
        )

        for dataset_name in config.dataset_config.names:
            logger.info(f"\n{'#'*80}\n# Dataset: {dataset_name.upper()}\n{'#'*80}")

            # Load dataset metadata
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

            # Run experiments for all combinations
            for feature_combination in self.combinations:
                try:
                    self._run_single_combination(
                        dataset_name=dataset_name,
                        feature_combination=feature_combination.value,
                        n_classes=n_classes,
                        original_size=original_size
                    )

                    logger.info(
                        f"✓ Completed {feature_combination.value} for {dataset_name}"
                    )

                except FileNotFoundError as e:
                    logger.error(
                        f"✗ Skipping {feature_combination.value} - {str(e)}"
                    )
                    continue

                except Exception as e:
                    logger.error(
                        f"✗ Error running {feature_combination.value} for {dataset_name}: {e}"
                    )
                    continue

        logger.info(
            f"\n{'#'*80}\n"
            f"# FEATURE ABLATION EXPERIMENT COMPLETE\n"
            f"# Results saved to: {self.report_path}\n"
            f"{'#'*80}"
        )

        return self.report
