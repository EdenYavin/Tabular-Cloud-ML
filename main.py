from loguru import logger
import argparse
from pathlib import Path
import src.utils.constansts as consts
from src.experiments import DatasetCreationHandler, IncrementEvalExperimentHandler, ModelTrainingExperimentHandler
from src.experiments.model_training_loop import ModelTrainingLoopExperimentHandler
from src.experiments.k_fold_handler import KModelTrainingExperimentHandler
from src.utils.config import config, update_config_from_args
from src.utils.constansts import EXPERIMENTS, IIM_MODELS, PMLB_DATASETS, REPORT_PATH, OUTPUT_DIR_PATH

import tensorflow as tf
import numpy as np
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
        dest="dataset_batch_size",
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
        dest="experiment_use_raw",
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
        nargs="+",
        default=["classes"],
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

    parser.add_argument(
        "--triangulation-mode",
        type=str,
        default="diff",
        dest="experiment_triangulation_mode",  # Maps to config.experiment_config.triangulation_mode
        choices=["concat", "diff"],
        help="Choose how to represent triangulation features: 'concat' (Original: Sample + Anchors) or 'diff' (Relative: Sample - Anchors)"
    )

    parser.add_argument(
        "--triangulation-embedding-model",
        type=str,
        default="dino",
        dest="encoder_embedding",
        choices=["clip", "dino"],
        help="Choose the embedding model for triangulation: 'clip' (original thesis) or 'dino' (new experiment)"
    )

    args = parser.parse_args()

    # Check if the list contains a single string with spaces and split it if necessary
    triang_args = args.experiment_triangulation_choosing
    if triang_args and len(triang_args) == 1 and len(triang_args[0].split()) > 1 :
        # Split the single string "kmeans classes" into ["kmeans", "classes"]
        args.experiment_triangulation_choosing = triang_args[0].split()
        logger.debug(f"DEBUG: Fixed triangulation args to: {args.experiment_triangulation_choosing}")
    # --- END FIX ---

    update_config_from_args(config, args)

    # 1. First, decide if TF should see the GPU at all.
    # If the current encoder is NOT a GPU model, hide the GPU from TensorFlow immediately.
    # This prevents TF from touching the GPU, leaving it entirely free for DINO (PyTorch).
    if config.encoder_config.name not in consts.GPU_MODELS:
        try:
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU hidden from TensorFlow (reserved for PyTorch/DINO or CPU execution).")
        except RuntimeError as e:
            # This happens if TF was somehow initialized before this point
            logger.error(f"Could not hide GPU: {e}")

    # 2. If TF *can* see the GPU, enable memory growth.
    # This ensures that if TF uses the GPU, it doesn't grab 100% of the VRAM,
    # allowing PyTorch/DINO to coexist if they are sharing the GPU.
    gpus = tf.config.list_physical_devices(
        'GPU')  # list_physical_devices sees all GPUs regardless of visibility settings above?
    # Actually, set_visible_devices affects what list_logical_devices sees,
    # but set_memory_growth must be called on physical devices.
    if gpus:
        try:
            for gpu in gpus:
                # Only set growth if the device is actually visible/available to TF context?
                # It's safer to just set it. If it was hidden above, TF won't use it anyway.
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")

    report_path = REPORT_PATH

    if config.experiment_config.to_run == consts.EXPERIMENTS.INCREMENT_EVALUATION:
        experiment_handler = IncrementEvalExperimentHandler

    elif config.experiment_config.to_run == consts.EXPERIMENTS.DATASET_CREATION:
        experiment_handler = DatasetCreationHandler

    elif config.experiment_config.to_run == consts.EXPERIMENTS.TRAINING_LOOP:
        experiment_handler = ModelTrainingLoopExperimentHandler
    else:
        experiment_handler = ModelTrainingExperimentHandler
        if config.experiment_config.k_folds > 1:
            report_path = Path(OUTPUT_DIR_PATH) / "k_report.csv"
            experiment_handler = KModelTrainingExperimentHandler


    with experiment_handler(report_path=report_path) as experiment:
        experiment.run_experiment()



if __name__ == '__main__':
    main()