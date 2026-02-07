"""
T Network Training Handler

Trains the T network (Encoder + Decoder) for reconstruction-only experiments.
This verifies that the T network can converge and properly reconstruct
encrypted embeddings without the classification head.
"""

from datetime import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

from src.experiments.base import ExperimentHandler
from src.internal_model.model import TNetworkOnlyIIM
from src.pipeline.deepset_features_dataset import DeepSetFeatureEngineering
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from src.utils.constansts import REPORT_PATH, OUTPUT_DIR_PATH
from src.utils.db import RawSplitDBFactory
from src.embeddings import EmbeddingsFactory
from src.encryptor import EncryptorFactory
from src.cloud import CLOUD_MODELS, DEFAULT_CLOUD_OUTPUT_SHAPE
from src.utils.db import EmbeddingDBFactory


class TNetworkTrainingHandler(ExperimentHandler):
    """
    Experiment handler for T network-only training.

    This handler:
    1. Creates the dataset using DeepSetFeatureEngineering
    2. Builds a TNetworkOnlyIIM model
    3. Trains using MSE reconstruction loss only
    4. Logs metrics and saves results
    """

    def __init__(
        self,
        experiment_name: str = "t_network_training",
        report_path: str = REPORT_PATH,
    ):
        super().__init__(experiment_name, report_path)
        self.history = None
        self.model = None

    def _get_cache_path(self, dataset_name: str) -> Path:
        """
        Generate a unique cache path for the T network dataset.

        The cache key includes:
        - Dataset name
        - Number of triangulation samples (anchors)
        - Embedding model type
        - Rotating key flag
        - Cloud model names (if any)
        """
        n_anchors = config.experiment_config.n_triangulation_samples
        embedding = config.encoder_config.embedding
        rotate_key = "rotate" if config.encoder_config.rotating_key else "no_rotate"
        cloud_models = "_".join(config.cloud_config.names) if config.cloud_config.names else "no_cloud"

        cache_dir = Path(OUTPUT_DIR_PATH) / "t_network_cache" / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_filename = f"t_net_{n_anchors}anchors_{embedding}_{rotate_key}_{cloud_models}.pkl"
        return cache_dir / cache_filename

    def run_experiment(self):
        """Run the T network training experiment."""
        logger.info("=" * 60)
        logger.info("Starting T Network-Only Training Experiment")
        logger.info("=" * 60)

        for dataset_name in config.dataset_config.names:
            logger.info(f"Processing dataset: {dataset_name}")

            # Step 1: Create dataset
            logger.info("Step 1: Creating dataset...")
            train_data, val_data, test_data, metadata = self._create_dataset(dataset_name)

            # Step 2: Build model
            logger.info("Step 2: Building T Network model...")
            self.model = self._build_model(metadata)

            # Step 3: Train model
            logger.info("Step 3: Training T Network...")
            self.history = self._train_model(train_data, val_data)

            # Step 4: Evaluate on test set
            logger.info("Step 4: Evaluating on test set...")
            test_metrics = self._evaluate_model(test_data)

            # Step 5: Save trained model
            logger.info("Step 5: Saving trained T Network model...")
            model_save_path = self._save_model(dataset_name, metadata)

            # Step 6: Log results
            logger.info("Step 6: Logging results...")
            self._log_experiment_results(metadata, test_metrics)

            logger.info("=" * 60)
            logger.info("T Network Training Experiment Complete")
            logger.info(f"Model saved to: {model_save_path}")
            logger.info(f"Final Test MSE: {test_metrics['test_mse']:.6f}")
            logger.info(f"Final Cosine Similarity: {test_metrics['test_cosine_similarity']:.6f}")
            logger.info("=" * 60)

        return self.history, test_metrics

    def _create_dataset(self, dataset_name: str) -> tuple:
        """
        Create the dataset for T network training.

        Returns:
            Tuple of (train_data, val_data, test_data, metadata)
            Each data tuple is (X, y) where y is classification labels (ignored during training)
        """
        # Check for cached dataset
        cache_path = self._get_cache_path(dataset_name)
        if cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info("Successfully loaded cached dataset")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recreating dataset...")

        logger.info("Creating new dataset (cache not found)")

        # Load raw dataset
        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        logger.debug(f"Original Dataset Size: {raw_dataset.get_dataset()[0].shape}")

        # Get train/test split
        X_train, X_test, _, y_train, y_test, _ = RawSplitDBFactory.get_db(raw_dataset).get_split()

        # Get sparse autoencoder embedding model (configured default)
        # This produces 64-dim embeddings from raw tabular data
        embedding_model = EmbeddingsFactory().get_model(
            X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name
        )

        # Embed raw data using SparseAE to get plaintext features (p_x, p_i)
        # These 64-dim embeddings are used as "plaintext" before encryption
        db = EmbeddingDBFactory.get_db(dataset_name, embedding_model)
        X_train_emb = db.get_embedding(X_train, is_test=False)
        X_test_emb = db.get_embedding(X_test, is_test=True)

        # Get cloud model output shape
        if config.cloud_config.names:
            cloud_model_output = CLOUD_MODELS[config.cloud_config.names[0]].input_shape
        else:
            cloud_model_output = DEFAULT_CLOUD_OUTPUT_SHAPE

        # Create encryptor
        encryptor = EncryptorFactory.get_model(
            dataset_name=dataset_name,
            output_shape=cloud_model_output
        )

        # Create feature engineering pipeline
        feature_engineering = DeepSetFeatureEngineering(
            dataset_name=dataset_name,
            encryptor=encryptor,
            embeddings_model=embedding_model,
            metadata=None
        )

        # Generate features
        train_X, train_y, _ = feature_engineering._get_features(
            X_train, X_train_emb, y_train, is_test=False
        )
        test_X, test_y, _ = feature_engineering._get_features(
            X_test, X_test_emb, y_test, is_test=True
        )

        # Calculate metadata
        n_anchors = config.experiment_config.n_triangulation_samples
        embedding_dim = 768 if "dino" in config.encoder_config.embedding else 512
        raw_dim = X_train_emb.shape[1]

        num_cloud_models = len(config.cloud_config.names) if config.cloud_config.names else 0
        cloud_vector_size = 1000 * num_cloud_models

        metadata = {
            "dataset_name": dataset_name,
            "n_anchors": n_anchors,
            "raw_dim": raw_dim,
            "emb_dim": embedding_dim,
            "cloud_vector_size": cloud_vector_size,
            "train_samples": len(train_X),
            "test_samples": len(test_X),
            "input_dim": train_X.shape[1],
        }

        logger.info(f"Dataset metadata: {metadata}")

        # Create validation split from training data
        split_idx = int(len(train_X) * 0.9)
        val_X = train_X[split_idx:]
        val_y = train_y[split_idx:]
        train_X = train_X[:split_idx]
        train_y = train_y[:split_idx]
        metadata["train_samples"] = len(train_X)
        metadata["val_samples"] = len(val_X)

        # Save to cache for future runs
        dataset_tuple = ((train_X, train_y), (val_X, val_y), (test_X, test_y), metadata)
        try:
            logger.info(f"Saving dataset to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset_tuple, f)
            logger.info("Dataset successfully cached")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        return dataset_tuple

    def _build_model(self, metadata: dict) -> TNetworkOnlyIIM:
        """Build the T Network model."""
        model = TNetworkOnlyIIM(
            n_anchors=metadata["n_anchors"],
            raw_dim=metadata["raw_dim"],
            emb_dim=metadata["emb_dim"],
            cloud_vector_size=metadata["cloud_vector_size"],
            context_dim=128,
        )

        # Build the model with input shape
        model.build((None, metadata["input_dim"]))

        # Compile with Adam optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )

        model.summary(print_fn=logger.info)

        return model

    def _train_model(
        self,
        train_data: tuple,
        val_data: tuple,
    ) -> tf.keras.callbacks.History:
        """
        Train the T network model.

        Args:
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple

        Returns:
            Training history
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Get training config
        epochs = config.iim_config.neural_net_config.epochs
        batch_size = config.iim_config.neural_net_config.batch_size

        logger.info(f"Training for {epochs} epochs with batch size {batch_size}")

        # Create output directory for logs
        log_dir = os.path.join(OUTPUT_DIR_PATH, "logs", f"t_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            ),
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def _evaluate_model(self, test_data: tuple) -> dict:
        """
        Evaluate the model on test set.

        Args:
            test_data: (X_test, y_test) tuple

        Returns:
            Dictionary of test metrics
        """
        X_test, _ = test_data
        y_test = self.model.extract_target(X_test)

        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0)

        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))

        # Cosine similarity (how well reconstruction preserves direction)
        y_test_norm = y_test / (np.linalg.norm(y_test, axis=1, keepdims=True) + 1e-8)
        y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-8)
        cosine_sim = np.mean(np.sum(y_test_norm * y_pred_norm, axis=1))

        metrics = {
            "test_mse": float(mse),
            "test_mae": float(mae),
            "test_cosine_similarity": float(cosine_sim),
        }

        logger.info(f"Test metrics: {metrics}")

        return metrics

    def _save_model(self, dataset_name: str, metadata: dict) -> Path:
        """Save the trained T Network model with metadata."""
        from src.utils.helpers import get_t_network_model_path

        # Use shared utility to generate path
        model_path = get_t_network_model_path(
            dataset_name=dataset_name,
            n_anchors=metadata["n_anchors"]
        )

        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(model_path)
        logger.info(f"T Network saved to {model_path}")

        # Save metadata as JSON
        metadata_path = model_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        return model_path

    def _log_experiment_results(self, metadata: dict, test_metrics: dict):
        """Log experiment results to the report."""
        new_row = {
            "date": [datetime.now().strftime("%d/%m/%Y %H:%M")],
            "experiment": [self.experiment_name],
            "dataset": [metadata["dataset_name"]],
            "n_anchors": [metadata["n_anchors"]],
            "raw_dim": [metadata["raw_dim"]],
            "emb_dim": [metadata["emb_dim"]],
            "cloud_vector_size": [metadata["cloud_vector_size"]],
            "train_samples": [metadata["train_samples"]],
            "val_samples": [metadata["val_samples"]],
            "test_samples": [metadata["test_samples"]],
            "test_mse": [test_metrics["test_mse"]],
            "test_mae": [test_metrics["test_mae"]],
            "test_cosine_similarity": [test_metrics["test_cosine_similarity"]],
            "final_train_loss": [self.history.history['loss'][-1]],
            "final_val_loss": [self.history.history['val_loss'][-1]],
            "epochs_trained": [len(self.history.history['loss'])],
        }

        self.report = pd.concat([self.report, pd.DataFrame(new_row)], ignore_index=True)
        self.save()
