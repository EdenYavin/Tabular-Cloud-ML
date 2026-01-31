# T Network-Only Training Experiment

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train only the T network (Decoder) without the ISM classification head to verify it can converge and properly reconstruct encrypted embeddings (q_x).

**Architecture:** Replace the meta_learning module with a simplified T network training pipeline. Create a new experiment handler that trains Encoder + Decoder (T) using only MSE reconstruction loss, without any classification. Dataset creation remains similar but classification labels are ignored.

**Tech Stack:** TensorFlow/Keras, existing DeepSetFeatureEngineering pipeline, loguru logging

---

## Background

The current `DeepSetsReconstructionIIM` model has three components:
1. **Encoder** - Processes anchor pairs (p_i, q_i) and raw sample (p_x) → produces context vector (128-dim)
2. **Decoder (T Network)** - Takes context → reconstructs q_x (encrypted embedding)
3. **Classifier** - Takes context → predicts class labels

This experiment isolates the Encoder + T Network to verify the reconstruction task works independently.

---

## Task 1: Remove meta_learning Directory

**Files:**
- Delete: `src/meta_learning/__init__.py`
- Delete: `src/meta_learning/set_transformer.py`
- Delete: `src/meta_learning/key_encoder.py`
- Delete: `src/meta_learning/keyspace_generator.py`
- Delete: `src/meta_learning/train_key_encoder.py`

**Step 1: Remove the meta_learning directory**

```bash
rm -rf src/meta_learning
```

**Step 2: Remove imports referencing meta_learning from experiments**

Check if `src/experiments/key_encoder_training_handler.py` imports from meta_learning and either delete or update it.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove meta_learning module (preparing for T network-only experiment)"
```

---

## Task 2: Create T Network-Only Model

**Files:**
- Modify: `src/internal_model/model.py` (add new model class at end)
- Modify: `src/utils/constansts.py` (add to IIM_MODELS enum)

**Step 1: Write the T Network-Only model class**

Add to `src/internal_model/model.py` after the existing `DeepSetsReconstructionIIM` class:

```python
class TNetworkOnlyIIM(tf.keras.Model):
    """
    T Network-Only model for reconstruction experiments.

    This model contains only the Encoder and Decoder (T Network) without
    the classification head. Used to verify the T network can converge
    and properly reconstruct encrypted embeddings.

    Input vector structure: [p_x | p_i | q_i | q_x_target]
    - p_x: Raw tabular sample (raw_dim,)
    - p_i: Raw tabular anchors (n_anchors * raw_dim,)
    - q_i: Encrypted anchor embeddings (n_anchors * emb_dim,)
    - q_x_target: Target encrypted embedding to reconstruct (emb_dim,)

    Output: Reconstructed q_x prediction (emb_dim,)
    """

    def __init__(
        self,
        n_anchors: int,
        raw_dim: int,
        emb_dim: int,
        context_dim: int = 128,
        name: str = "t_network_only",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.n_anchors = n_anchors
        self.raw_dim = raw_dim
        self.emb_dim = emb_dim
        self.context_dim = context_dim

        # Calculate input slice indices
        self.p_x_end = raw_dim
        self.p_i_end = raw_dim + (n_anchors * raw_dim)
        self.q_i_end = self.p_i_end + (n_anchors * emb_dim)
        self.q_x_target_end = self.q_i_end + emb_dim

        # Encoder: processes (p_i, q_i, p_x) to produce context
        self._build_encoder()

        # Decoder (T Network): reconstructs q_x from context
        self._build_decoder()

    def _build_encoder(self):
        """Build the Deep Sets encoder for anchor pairs + sample."""
        # Process p_i (raw anchors)
        self.p_i_dense1 = tf.keras.layers.Dense(128, activation='relu', name='p_i_dense1')
        self.p_i_dense2 = tf.keras.layers.Dense(64, activation='relu', name='p_i_dense2')

        # Process q_i (encrypted anchor embeddings)
        self.q_i_dense1 = tf.keras.layers.Dense(128, activation='relu', name='q_i_dense1')
        self.q_i_dense2 = tf.keras.layers.Dense(64, activation='relu', name='q_i_dense2')

        # Process p_x (raw sample)
        self.p_x_dense1 = tf.keras.layers.Dense(128, activation='relu', name='p_x_dense1')
        self.p_x_dense2 = tf.keras.layers.Dense(64, activation='relu', name='p_x_dense2')

        # Combine all features
        self.combine_dense1 = tf.keras.layers.Dense(256, activation='relu', name='combine_dense1')
        self.combine_bn1 = tf.keras.layers.BatchNormalization(name='combine_bn1')
        self.combine_dense2 = tf.keras.layers.Dense(self.context_dim, activation='relu', name='combine_dense2')

    def _build_decoder(self):
        """Build the T network decoder for q_x reconstruction."""
        self.decoder_dense1 = tf.keras.layers.Dense(256, name='decoder_dense1')
        self.decoder_bn1 = tf.keras.layers.BatchNormalization(name='decoder_bn1')
        self.decoder_act1 = tf.keras.layers.LeakyReLU(name='decoder_act1')

        self.decoder_dense2 = tf.keras.layers.Dense(512, name='decoder_dense2')
        self.decoder_bn2 = tf.keras.layers.BatchNormalization(name='decoder_bn2')
        self.decoder_act2 = tf.keras.layers.LeakyReLU(name='decoder_act2')

        self.decoder_output = tf.keras.layers.Dense(self.emb_dim, activation='linear', name='decoder_output')

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, p_x + p_i + q_i + q_x_target)
            training: Whether in training mode

        Returns:
            q_x_pred: Reconstructed embedding (batch, emb_dim)
        """
        # Slice input components
        p_x = inputs[:, :self.p_x_end]
        p_i = inputs[:, self.p_x_end:self.p_i_end]
        q_i = inputs[:, self.p_i_end:self.q_i_end]
        # q_x_target is at inputs[:, self.q_i_end:self.q_x_target_end] but not used in forward pass

        # Reshape p_i and q_i for anchor processing
        p_i_reshaped = tf.reshape(p_i, (-1, self.n_anchors, self.raw_dim))
        q_i_reshaped = tf.reshape(q_i, (-1, self.n_anchors, self.emb_dim))

        # Process p_i anchors
        p_i_features = self.p_i_dense1(p_i_reshaped)
        p_i_features = self.p_i_dense2(p_i_features)
        p_i_pooled = tf.reduce_mean(p_i_features, axis=1)  # (batch, 64)

        # Process q_i anchors
        q_i_features = self.q_i_dense1(q_i_reshaped)
        q_i_features = self.q_i_dense2(q_i_features)
        q_i_pooled = tf.reduce_mean(q_i_features, axis=1)  # (batch, 64)

        # Process p_x sample
        p_x_features = self.p_x_dense1(p_x)
        p_x_features = self.p_x_dense2(p_x_features)  # (batch, 64)

        # Combine all features to create context
        combined = tf.concat([p_i_pooled, q_i_pooled, p_x_features], axis=-1)  # (batch, 192)
        context = self.combine_dense1(combined)
        context = self.combine_bn1(context, training=training)
        context = self.combine_dense2(context)  # (batch, context_dim)

        # Decode context to reconstruct q_x
        x = self.decoder_dense1(context)
        x = self.decoder_bn1(x, training=training)
        x = self.decoder_act1(x)

        x = self.decoder_dense2(x)
        x = self.decoder_bn2(x, training=training)
        x = self.decoder_act2(x)

        q_x_pred = self.decoder_output(x)  # (batch, emb_dim)

        return q_x_pred

    def get_config(self):
        return {
            "n_anchors": self.n_anchors,
            "raw_dim": self.raw_dim,
            "emb_dim": self.emb_dim,
            "context_dim": self.context_dim,
        }

    def extract_target(self, inputs):
        """Extract q_x_target from input tensor for loss computation."""
        return inputs[:, self.q_i_end:self.q_x_target_end]
```

**Step 2: Add to IIM_MODELS enum**

In `src/utils/constansts.py`, find the `IIM_MODELS` class and add:

```python
T_NETWORK_ONLY = "t_network_only"
```

**Step 3: Run basic import test**

```bash
python -c "from src.internal_model.model import TNetworkOnlyIIM; print('Import successful')"
```

Expected: `Import successful`

**Step 4: Commit**

```bash
git add src/internal_model/model.py src/utils/constansts.py
git commit -m "feat: add TNetworkOnlyIIM model for reconstruction-only experiments"
```

---

## Task 3: Create T Network Training Handler

**Files:**
- Create: `src/experiments/t_network_training_handler.py`

**Step 1: Create the experiment handler file**

```python
"""
T Network Training Handler

Trains the T network (Encoder + Decoder) for reconstruction-only experiments.
This verifies that the T network can converge and properly reconstruct
encrypted embeddings without the classification head.
"""

from datetime import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

from src.experiments.base import ExperimentHandler
from src.internal_model.model import TNetworkOnlyIIM
from src.pipeline.deepset_features_dataset import DeepSetFeatureEngineering
from src.utils.config import config
from src.utils.constansts import REPORT_PATH


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

    def run_experiment(self):
        """Run the T network training experiment."""
        logger.info("=" * 60)
        logger.info("Starting T Network-Only Training Experiment")
        logger.info("=" * 60)

        # Step 1: Create dataset
        logger.info("Step 1: Creating dataset...")
        train_data, val_data, test_data, metadata = self._create_dataset()

        # Step 2: Build model
        logger.info("Step 2: Building T Network model...")
        self.model = self._build_model(metadata)

        # Step 3: Train model
        logger.info("Step 3: Training T Network...")
        self.history = self._train_model(train_data, val_data)

        # Step 4: Evaluate on test set
        logger.info("Step 4: Evaluating on test set...")
        test_metrics = self._evaluate_model(test_data)

        # Step 5: Log results
        logger.info("Step 5: Logging results...")
        self._log_experiment_results(metadata, test_metrics)

        logger.info("=" * 60)
        logger.info("T Network Training Experiment Complete")
        logger.info(f"Final Test MSE: {test_metrics['test_mse']:.6f}")
        logger.info("=" * 60)

        return self.history, test_metrics

    def _create_dataset(self) -> tuple:
        """
        Create the dataset for T network training.

        Returns:
            Tuple of (train_data, val_data, test_data, metadata)
            Each data tuple is (X, y_target) where y_target is q_x_target
        """
        # Use DeepSetFeatureEngineering to create the dataset
        feature_engineering = DeepSetFeatureEngineering()

        # Get dataset names from config
        dataset_names = config.dataset_config.names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        dataset_name = dataset_names[0]
        logger.info(f"Creating dataset for: {dataset_name}")

        # Create the engineered features
        result = feature_engineering.create(dataset_name=dataset_name)

        # Extract train/val/test splits
        # The result structure depends on DeepSetFeatureEngineering implementation
        # Typically returns arrays with structure: [p_x | p_i | q_i | cloud | q_x_target | label]

        train_X = result.train.X
        train_y = result.train.y  # Classification labels (unused, but kept for compatibility)

        val_X = result.validation.X if hasattr(result, 'validation') else None
        val_y = result.validation.y if hasattr(result, 'validation') else None

        test_X = result.test.X
        test_y = result.test.y

        # Calculate metadata from first sample
        n_anchors = config.experiment_config.n_triangulation_samples
        raw_dim = config.dataset_config.input_dim if hasattr(config.dataset_config, 'input_dim') else self._infer_raw_dim(train_X, n_anchors)
        emb_dim = config.embedding_config.dim if hasattr(config.embedding_config, 'dim') else self._infer_emb_dim(train_X, n_anchors, raw_dim)

        metadata = {
            "dataset_name": dataset_name,
            "n_anchors": n_anchors,
            "raw_dim": raw_dim,
            "emb_dim": emb_dim,
            "train_samples": len(train_X),
            "val_samples": len(val_X) if val_X is not None else 0,
            "test_samples": len(test_X),
            "input_dim": train_X.shape[1],
        }

        logger.info(f"Dataset metadata: {metadata}")

        # Create validation split if not provided
        if val_X is None:
            split_idx = int(len(train_X) * 0.9)
            val_X = train_X[split_idx:]
            val_y = train_y[split_idx:]
            train_X = train_X[:split_idx]
            train_y = train_y[:split_idx]
            metadata["train_samples"] = len(train_X)
            metadata["val_samples"] = len(val_X)

        return (train_X, train_y), (val_X, val_y), (test_X, test_y), metadata

    def _infer_raw_dim(self, X, n_anchors):
        """Infer raw dimension from data shape."""
        # This is a heuristic - adjust based on actual data structure
        total_dim = X.shape[1]
        # Assuming structure: p_x + p_i + q_i + cloud + q_x_target + label
        # This needs refinement based on actual data
        logger.warning("Inferring raw_dim from data - may need adjustment")
        return config.dataset_config.get("input_dim", 10)

    def _infer_emb_dim(self, X, n_anchors, raw_dim):
        """Infer embedding dimension from data shape."""
        logger.warning("Inferring emb_dim from data - may need adjustment")
        return config.embedding_config.get("dim", 384)

    def _build_model(self, metadata: dict) -> TNetworkOnlyIIM:
        """Build the T Network model."""
        model = TNetworkOnlyIIM(
            n_anchors=metadata["n_anchors"],
            raw_dim=metadata["raw_dim"],
            emb_dim=metadata["emb_dim"],
            context_dim=128,
        )

        # Build the model with input shape
        model.build((None, metadata["input_dim"]))

        # Compile with MSE loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae'],
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
        X_train, _ = train_data
        X_val, _ = val_data

        # Extract targets (q_x_target is embedded in X)
        y_train = self.model.extract_target(X_train)
        y_val = self.model.extract_target(X_val)

        # Get training config
        epochs = config.iim_config.neural_net_config.epochs if hasattr(config.iim_config, 'neural_net_config') else 100
        batch_size = config.iim_config.neural_net_config.batch_size if hasattr(config.iim_config, 'neural_net_config') else 32

        logger.info(f"Training for {epochs} epochs with batch size {batch_size}")

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
                log_dir=f"output/logs/t_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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

    def _log_experiment_results(self, metadata: dict, test_metrics: dict):
        """Log experiment results to the report."""
        new_row = {
            "date": [datetime.now().strftime("%d/%m/%Y %H:%M")],
            "experiment": [self.experiment_name],
            "dataset": [metadata["dataset_name"]],
            "n_anchors": [metadata["n_anchors"]],
            "raw_dim": [metadata["raw_dim"]],
            "emb_dim": [metadata["emb_dim"]],
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
```

**Step 2: Run import test**

```bash
python -c "from src.experiments.t_network_training_handler import TNetworkTrainingHandler; print('Import successful')"
```

Expected: `Import successful`

**Step 3: Commit**

```bash
git add src/experiments/t_network_training_handler.py
git commit -m "feat: add TNetworkTrainingHandler for T network-only experiments"
```

---

## Task 4: Add Experiment Entry Point to main.py

**Files:**
- Modify: `main.py` (add new experiment type)
- Modify: `src/utils/constansts.py` (add to EXPERIMENTS enum)

**Step 1: Add T_NETWORK_TRAINING to EXPERIMENTS enum**

In `src/utils/constansts.py`, find the `EXPERIMENTS` class and add:

```python
T_NETWORK_TRAINING = "t_network_training"
```

**Step 2: Read main.py to understand current structure**

Read `main.py` to understand the experiment dispatch logic.

**Step 3: Add T network training to main.py dispatch**

Add the import at the top:

```python
from src.experiments.t_network_training_handler import TNetworkTrainingHandler
```

Add the case in the experiment dispatch section (find where other experiments are handled):

```python
elif config.experiment_config.to_run == EXPERIMENTS.T_NETWORK_TRAINING:
    handler = TNetworkTrainingHandler()
    handler.run_experiment()
```

**Step 4: Add command-line argument**

If using argparse, ensure `--experiment-to-run t_network_training` is a valid option.

**Step 5: Test the entry point**

```bash
python main.py --experiment-to-run t_network_training --help
```

Expected: Help text shows available options

**Step 6: Commit**

```bash
git add main.py src/utils/constansts.py
git commit -m "feat: add t_network_training experiment entry point"
```

---

## Task 5: Create Dataset Adapter for T Network

**Files:**
- Modify: `src/experiments/t_network_training_handler.py` (update `_create_dataset`)

**Step 1: Read DeepSetFeatureEngineering to understand output format**

Read `src/pipeline/deepset_features_dataset.py` to understand the exact output structure.

**Step 2: Update _create_dataset method**

Based on the actual output format of DeepSetFeatureEngineering, update the `_create_dataset` method to correctly extract:
- `p_x`: Raw sample features
- `p_i`: Raw anchor features
- `q_i`: Encrypted anchor embeddings
- `q_x_target`: Target encrypted embedding

The key is understanding the slice indices for each component in the concatenated feature vector.

**Step 3: Run dataset creation test**

```bash
python -c "
from src.experiments.t_network_training_handler import TNetworkTrainingHandler
handler = TNetworkTrainingHandler()
train, val, test, meta = handler._create_dataset()
print(f'Train shape: {train[0].shape}')
print(f'Metadata: {meta}')
"
```

Expected: Shapes printed without errors

**Step 4: Commit**

```bash
git add src/experiments/t_network_training_handler.py
git commit -m "fix: update dataset adapter for correct feature slicing"
```

---

## Task 6: Run Full Experiment and Verify Convergence

**Files:**
- No new files

**Step 1: Run the experiment**

```bash
python main.py --experiment-to-run t_network_training
```

**Step 2: Monitor training logs**

Watch for:
- Loss decreasing over epochs
- Early stopping triggered (good sign of convergence)
- Final MSE and cosine similarity values

**Step 3: Check TensorBoard logs**

```bash
tensorboard --logdir output/logs/
```

Navigate to localhost:6006 to view training curves.

**Step 4: Review results in report**

```bash
cat output/report.csv | tail -5
```

**Step 5: Commit results (optional)**

```bash
git add output/report.csv
git commit -m "results: T network convergence verification"
```

---

## Task 7: Clean Up and Documentation

**Files:**
- Remove: `src/experiments/key_encoder_training_handler.py` (if no longer needed)
- Modify: `CLAUDE.md` (update documentation)

**Step 1: Remove key_encoder_training_handler if obsolete**

If the meta_learning-based key encoder is no longer used:

```bash
rm src/experiments/key_encoder_training_handler.py
```

**Step 2: Update CLAUDE.md with new experiment type**

Add to the "Running Experiments" section:

```markdown
# Train T network only (reconstruction verification)
python main.py --experiment-to-run t_network_training
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "docs: update documentation for T network experiment"
```

---

## Summary

After completing all tasks:

1. **meta_learning/** directory removed
2. **TNetworkOnlyIIM** model added to `src/internal_model/model.py`
3. **TNetworkTrainingHandler** created in `src/experiments/`
4. **main.py** updated with new experiment entry point
5. **T network convergence verified** through actual training run

The experiment proves the T network (Encoder + Decoder) can learn to reconstruct encrypted embeddings independently of the classification task.
