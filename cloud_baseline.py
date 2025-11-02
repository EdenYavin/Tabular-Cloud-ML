import gc
import os
import pathlib
import pickle

import numpy as np


# os.environ["PROJECT_DIR"] = "/Users/eden.yavin/Projects/Tabular-Cloud-ML"

from src.dataset import RawDataset, DatasetFactory
from src.embeddings import EmbeddingsFactory
from src.encryptor import EncryptorFactory
from src.pipeline.cloud_features_dataset import CloudFeatureEngineering
from src.utils.config import config, update_config_from_args
import argparse
from src.utils.constansts import EXPERIMENTS, IIM_MODELS, PMLB_DATASETS, OUTPUT_DIR_PATH
from src.utils.db import RawSplitDBFactory
from src.utils.helpers import get_experiment_name, get_dataset_path
from loguru import logger
from src.cloud import CLOUD_MODELS, DEFAULT_CLOUD_OUTPUT_SHAPE
from src.utils.constansts import DATASET_FILE_NAME
from src.internal_model import InternalInferenceModelFactory


def main():
    dataset_name = config.dataset_config.names[0]

    # Get the output for the cloud model
    if config.cloud_config.names:
        cloud_model_output = CLOUD_MODELS[config.cloud_config.names[0]].input_shape
    else:
        cloud_model_output = DEFAULT_CLOUD_OUTPUT_SHAPE

    raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
    embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name)
    encryptor = EncryptorFactory.get_model(dataset_name=dataset_name, output_shape=cloud_model_output, )
    X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
    del X_train, y_train
    dataset_creator = CloudFeatureEngineering(
        dataset_name=dataset_name,
        encryptor=encryptor,
        embeddings_model=embedding_model,
        metadata=raw_dataset.metadata
    )
    dataset, emb_baseline_dataset = (
        dataset_creator.create(X_sample, y_sample, X_test, y_test)
    )

    return dataset, emb_baseline_dataset

def train(dataset):

    X_train, y_train, X_test, y_test = np.vstack(dataset.train.features), np.vstack(dataset.train.labels), dataset.test.features, dataset.test.labels

    internal_model = InternalInferenceModelFactory().get_model(
        num_classes=2,
        input_shape=X_train.shape[1],
        type="lstm"
    )
    logger.debug(f"#### EVALUATING INTERNAL MODEL lstm ####"
                 f" Dataset Shape: Train - {X_train.shape}, Test: {X_test.shape}")
    internal_model.fit(
        X=X_train, y=y_train,
        validation_data=(X_test, y_test),
    )
    metrics_results = internal_model.evaluate(
        X=X_test, y=y_test, metrics=["accuracy", "auc"]
    )
    print(f"Metrics Result: {metrics_results}")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run experiments with specified configurations.")
    parser.add_argument("--iim-train-baseline", action="store_true", help="Enable baseline mode.")
    parser.add_argument("--experiment-to-run", type=EXPERIMENTS,
                        choices=list(EXPERIMENTS), help="Experiment type: training or dataset.")

    parser.add_argument("--use-cloud-models",
                        type=str,
                        nargs="+",
                        default=[],
                        dest="cloud_names",
                        help="The cloud models to use")

    parser.add_argument("--encoder-rotating-key",
                        action="store_true",  # Sets to True if flag is present
                        help="Use triangulation samples or not")

    parser.add_argument(
        "--dataset-batch-size",
        type=int,
        default=100,
        dest="dastaset_batch_size",
        help="Batch size for dataset creation. In the feature engineering pipeline we will iterate the raw dataset"
             " with batches where each batch has the dataset_batch_size"
    )

    # Example of using 'dest'
    parser.add_argument(
        "--number-of-prediction-vector",  # User-facing name
        type=int,
        default=1,
        dest="experiment_n_pred_vectors",  # Internal name for your config
        help="Specify the number of prediction vectors for the experiment."
    )

    parser.add_argument(
        "--iim-name",
        type=str,
        nargs="+",
        default=[IIM_MODELS.LSTM],
        help="Specify one or more IIM model names (e.g. --iim-model-name lstm, dense)"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        dest="dataset_names",  # Internal name for your config
        default=PMLB_DATASETS,
        help="The raw dataset(s) used to create the new cloud features dataset."
    )

    parser.add_argument(
        "--iim-epochs",
        type=int,
        dest="iim_neural_net_epochs",  # Internal name for your config
        default=30,
        help="The number of epochs to use for training the model."
    )

    parser.add_argument(
        "--dataset-use-cache",
        action="store_true",  # Sets to True if flag is present
        help="Use dataset folders instead of creating the data on the fly each epoch."
    )

    parser.add_argument(
        "--use-raw-features",
        dest="experiment_use_embedding",
        action="store_true",  # Sets to True if flag is present
        help="Use the raw dataset features as part of the feature engineering pipeline."
    )

    parser.add_argument(
        "--use-horizontal-cloud-features",
        type=bool,
        default=True,
        help="Add the cloud features horizontally to each sample or duplicate each sample as the number of cloud models, each sample with a different cloud feature.",
        dest="cloud_horizontal_append"
    )

    parser.add_argument(
        "--triangulation-choosing",  # User-facing name
        type=str,
        default="classes",
        dest="experiment_triangulation_choosing",  # Internal name for your config
        help="Specify how to choose the triangulation - first (First N samples), last, random (random N samples)"
    )

    parser.add_argument(
        "--triangulation-samples",  # User-facing name
        type=int,
        default=3,
        dest="experiment_n_triangulation_samples",  # Internal name for your config
        help="Specify how many triangulation samples to use in case of first / last / random triangulation type."
    )

    parser.add_argument(
        "--k-training",  # User-facing name
        type=int,
        default=1,
        dest="experiment_k_folds",  # Internal name for your config
        help="Number of times we will train the iim to get number of results. Useful for testing statistic about model performance."
    )

    args = parser.parse_args()

    update_config_from_args(config, args)

    dataset, _ = main()
    train(dataset)
    logger.info("Finished.")