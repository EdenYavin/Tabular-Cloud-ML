import gc, numpy as np
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')


from src.internal_model import InternalInferenceModelFactory, LSTMIIM
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from loguru import logger
from src.experiments.base import ExperimentHandler
from src.utils.db import EmbeddingDBFactory, RawSplitDBFactory
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.encryptor import EncryptorFactory
from src.embeddings import EmbeddingsFactory, ClipEmbedding
from src.cloud import CLOUD_MODELS, DEFAULT_CLOUD_OUTPUT_SHAPE, CloudModelManager

def plot(val_losses, train_losses, val_accuracies, train_accuracies, dataset_name, model_name, n_pred_vectors=1):

    path = get_dataset_path(dataset_name=dataset_name, n_pred_vectors=n_pred_vectors)
    plot_path = path / f"{model_name}_train_plot.png"
    os.makedirs(path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{dataset_name} Training Curve")

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot accuracy if available
    if train_accuracies is not None:
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        if val_accuracies is not None:
            plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
    logger.info(f"Saving plot to: {plot_path}")
    plt.savefig(plot_path)


def encrypt_and_embed(dataset_name, triangulation_embedding,cloud, X, y):
    # Get the output for the cloud model
    if config.cloud_config.names:
        cloud_model_output = CLOUD_MODELS[config.cloud_config.names[0]].input_shape
    else:
        cloud_model_output = DEFAULT_CLOUD_OUTPUT_SHAPE

    encryptor = EncryptorFactory.get_model(dataset_name=dataset_name, output_shape=cloud_model_output)

    triangulation_samples = X[:config.experiment_config.n_triangulation_samples]
    with tqdm(total=len(X), leave=True, position=0, desc="Encrypting, Embedding, Predicting") as pbar:

        observations, new_y = [], []
        for x, label in zip(X, y):
            pbar.update(1)
            # Triangulation features vector = X', Y_1', Y_2',...
            x_tag = encryptor.encode(x.reshape(1, -1))
            # 1. Encrypt them using the new key
            y_tag = encryptor.encode(triangulation_samples)
            # 2. Embed the encryption
            y_tag_emb = triangulation_embedding(y_tag)

            x_tag_emb = triangulation_embedding.forward(np.vstack(x_tag))

            observation = [x_tag_emb.flatten(), y_tag_emb.flatten()]
            del x_tag, x_tag_emb, y_tag_emb, y_tag_emb

            # Add the cloud predictions as features if needed:
            if config.cloud_config.names:
                for cloud_model in config.cloud_config.names:
                    predictions = cloud.predict(model_name=cloud_model, batch=x_tag)
                    observations.append(np.hstack([np.hstack(observation), predictions.flatten()]))
                    # Duplicate the labels
                    new_y.append(label)
                    del predictions
            else:
                # No cloud models need to be used, just use the features up until now
                observations.append(np.hstack(observation))
                # Duplicate the labels
                new_y.append(label)
            if config.encoder_config.rotating_key:
                # Switch key for the next example
                encryptor.switch_key()

    del triangulation_samples, encryptor

    return np.vstack(observations), np.vstack(new_y)

class ModelTrainingLoopExperimentHandler(ExperimentHandler):

    def __init__(self):
        super().__init__(get_experiment_name())
        self.checkpoint_metadata = {}

    def _load_checkpoint_medata(self, dataset_name):
        path = get_dataset_path(dataset_name, 1)
        checkpoint_path = path / "checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                self.checkpoint_metadata = json.load(f)
        else:
            self.checkpoint_metadata['start_epoch'] = 0
            self.checkpoint_metadata['model_file'] = None

    def run_experiment(self):

        logger.info(f"Training Loop Experiment: {get_experiment_name()}")
        cloud = CloudModelManager().__enter__()
        logger.info(f"Triangulation model is on, using {ClipEmbedding.name}")
        triangulation_embedding = ClipEmbedding()

        for dataset_name in config.dataset_config.names:

            self._load_checkpoint_medata(dataset_name)

            raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
            logger.debug(f"Original Dataset Size: {raw_dataset.get_dataset()[0].shape}")
            X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
            embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y,dataset_name=dataset_name)

            n_classes = raw_dataset.get_n_classes()
            del raw_dataset, X_sample, y_sample

            db = EmbeddingDBFactory.get_db(dataset_name, embedding_model)
            X_train = db.get_embedding(X_train, is_test=False)
            X_test = db.get_embedding(X_test, is_test=True)

            del db, embedding_model
            gc.collect()

            for model_name in config.iim_config.name:

                logger.info(f"#### Training model experiment: "
                            f"Dataset: {dataset_name}, Model: {model_name} ####")

                if self.checkpoint_metadata['model_file']:
                    model = keras.models.load_model(self.checkpoint_metadata['model_file'])
                else:
                    model = None

                with tf.device('/GPU:0'):
                    logger.info(
                        f'Using GPU: {list(filter(lambda d: "GPU:0" in d.name, tf.config.list_physical_devices()))}')

                    epoch_train_loss, epoch_train_acc = [], []
                    train_losses, train_accuracies = [], []
                    val_losses, val_accuracies  = [], []
                    optimizer = keras.optimizers.Adam()
                    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                    logger.warning(f"Starting from Epoch: {self.checkpoint_metadata['start_epoch']}")
                    for epoch in range(self.checkpoint_metadata['start_epoch'], config.iim_config.neural_net_config.epochs):
                        try:
                            new_X_train, new_y_train = encrypt_and_embed(dataset_name, triangulation_embedding, cloud, X_train, y_train)
                            new_X_test, new_y_test = encrypt_and_embed(dataset_name, triangulation_embedding, cloud, X_test, y_test)
                            train_dataset = tf.data.Dataset.from_tensor_slices((new_X_train, new_y_train)).batch(config.iim_config.neural_net_config.batch_size)
                            test_dataset = tf.data.Dataset.from_tensor_slices((new_X_test, new_y_test)).batch(config.iim_config.neural_net_config.batch_size)

                            if not model:
                                model = LSTMIIM(num_classes=n_classes,
                                    input_shape=new_X_train.shape[1],
                                    type=model_name).model

                            # Iterate over batches
                            for step, (x_batch, y_batch) in enumerate(train_dataset):
                                # Forward pass + gradients
                                with tf.GradientTape() as tape:
                                    x_batch, y_batch = LSTMIIM.prepare_data_for_training(x_batch, y_batch)
                                    logits = model(x_batch, training=True)
                                    loss = loss_fn(y_batch, logits)
                                    acc = tf.reduce_mean(keras.metrics.sparse_categorical_accuracy(y_batch, logits))

                                # Backpropagation
                                gradients = tape.gradient(loss, model.trainable_weights)
                                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                                # Record batch metrics
                                epoch_train_loss.append(loss.numpy())
                                epoch_train_acc.append(acc.numpy())

                            # Calculate epoch averages
                            current_train_loss = np.mean(epoch_train_loss)
                            train_losses.append(current_train_loss)
                            current_train_acc = np.mean(epoch_train_acc)
                            train_accuracies.append(current_train_acc)

                            epoch_val_loss = []
                            epoch_val_acc = []

                            for step, (x_val_batch, y_val_batch) in enumerate(test_dataset):
                                x_val_batch, y_val_batch = LSTMIIM.prepare_data_for_training(x_val_batch, y_val_batch)
                                val_logits = model(x_val_batch, training=False)
                                val_loss = loss_fn(y_val_batch, val_logits)
                                val_acc = tf.reduce_mean(
                                    keras.metrics.sparse_categorical_accuracy(y_val_batch, val_logits))
                                epoch_val_loss.append(val_loss.numpy())
                                epoch_val_acc.append(val_acc.numpy())

                            current_val_loss = np.mean(epoch_val_loss)
                            val_losses.append(current_val_loss)
                            current_val_acc = np.mean(epoch_val_acc)
                            val_accuracies.append(current_val_acc)

                            print(f"\nEpoch {epoch + 1}/100: Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Train Acc: {current_train_acc:.4f}, Val Acc: {current_val_acc:.4f}")
                            plot(train_losses, val_losses, train_accuracies, val_accuracies, dataset_name, model_name)
                            del new_X_train, new_X_test, new_y_train, new_y_test, train_dataset, test_dataset, x_val_batch, x_batch, y_batch, y_val_batch
                            gc.collect()

                        except Exception as e:
                            logger.error(f"Error in training: {e}")
                            path = get_dataset_path(dataset_name, 1)
                            model_path = path / f"{model_name}_{epoch}.keras"
                            self.checkpoint_metadata['model_file'] = model_path
                            self.checkpoint_metadata['start_epoch'] = epoch
                            logger.warning(f"Saving checkpoint to: {model_path}")
                            model.save(model_path)

                cloud.__exit__(None, None, None)

        return self.report