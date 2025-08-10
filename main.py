import argparse

import src.utils.constansts as consts
from src.experiments import DatasetCreationHandler, IncrementEvalExperimentHandler, ModelTrainingExperimentHandler
from src.experiments.model_training_loop import ModelTrainingLoopExperimentHandler
from src.utils.config import config, update_config_from_args
import tensorflow as tf
import numpy as np

from src.utils.constansts import EXPERIMENTS, IIM_MODELS, PMLB_DATASETS

np.random.seed(42)

def main():

    parser = argparse.ArgumentParser(description="Run experiments with specified configurations.")
    parser.add_argument("--iim-train-baseline", action="store_true", help="Enable baseline mode.")
    parser.add_argument("--experiment-to-run", type=EXPERIMENTS,
                        choices=list(EXPERIMENTS), help="Experiment type: training or dataset.")

    parser.add_argument(    "--use-cloud-models",
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
        default='first',
        dest="experiment_triangulation_choosing",  # Internal name for your config
        help="Specify how to choose the embedding - first (First N samples), last, random (random N samples)"
    )

    args = parser.parse_args()

    update_config_from_args(config, args)

    # Enable TensorFlow’s “allow growth” option so it only uses as much GPU memory as needed, rather than trying to allocate all memory up front
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


    # Use GPU only when using Decon
    if config.encoder_config.name not in consts.GPU_MODELS:
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')

    if config.experiment_config.to_run == consts.EXPERIMENTS.INCREMENT_EVALUATION:
        experiment_handler = IncrementEvalExperimentHandler

    elif config.experiment_config.to_run == consts.EXPERIMENTS.DATASET_CREATION:
        experiment_handler = DatasetCreationHandler

    elif config.experiment_config.to_run == consts.EXPERIMENTS.TRAINING_LOOP:
        experiment_handler = ModelTrainingLoopExperimentHandler
    else:
        experiment_handler = ModelTrainingExperimentHandler


    with experiment_handler() as experiment:
        experiment.run_experiment()



if __name__ == '__main__':
    main()