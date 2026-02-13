from loguru import logger
import argparse
from pathlib import Path
import src.utils.constansts as consts
from src.experiments import DatasetCreationHandler, IncrementEvalExperimentHandler, ModelTrainingExperimentHandler, FeatureAblationExperimentHandler
from src.experiments.model_training_loop import ModelTrainingLoopExperimentHandler
from src.experiments.k_fold_handler import KModelTrainingExperimentHandler
from src.experiments.t_network_training_handler import TNetworkTrainingHandler
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

    parser.add_argument(
        "--iim-dropout",
        type=float,
        default=0,
        dest="iim_neural_net_dropout",  # Auto-maps to config.iim_config.neural_net_config.dropout
        help="Dropout rate for the IIM (0.0 to 1.0). Overrides config default."
    )

    parser.add_argument(
        "--use-cloud-models",
        type=str,
        nargs="+",  # Allows passing multiple models: --use-cloud-models xception convnext_large
        default=[],  # CHANGE: Default is None so we don't overwrite config if flag is missing
        dest="cloud_names",  # Maps to config.cloud_config.names
        help="The cloud models to use (e.g., xception, convnext_large, efficientnet)"
    )

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
        dest="iim_name",
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
        help="Specify how many triangulation to sample."
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
        choices=["concat", "diff", "cos"],
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

    parser.add_argument(
        "--use-calibration-vector",
        action="store_true",
        dest="experiment_use_calibration_vector",
        default=False,
        help="Flag to indicate if the calibration vector should be used in the IIM training",
    )

    parser.add_argument(
        "--scaling-factor",
        type=float,
        dest="experiment_scaling_factor",
        default=5,
        help="Scaling factor to scale the encrypted images embedding (for example 1.08 pixel will turn to 0.42)",
    )

    parser.add_argument(
        "--calibration-distributions",
        type=str,
        nargs="+",
        default=["gaussian"],
        dest="experiment_calibration_distributions",
        help="Calibration vector distribution types. Options: uniform, gaussian, sparse, bimodal, edges. "
             "Use multiple to create a richer key fingerprint (e.g., --calibration-distributions gaussian sparse bimodal)",
    )

    # Key Encoder Training Arguments
    parser.add_argument(
        "--num-keys",
        type=int,
        default=500,
        dest="num_keys",
        help="Number of unique encryption keys to generate for key encoder training",
    )

    parser.add_argument(
        "--num-calibration-pairs",
        type=int,
        default=50,
        dest="num_calibration_pairs",
        help="Number of calibration pairs per key for key encoder training",
    )

    parser.add_argument(
        "--output-embedding-dim",
        type=int,
        default=256,
        dest="output_embedding_dim",
        help="Dimension of functional embeddings for key encoder training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        dest="num_epochs",
        help="Number of training epochs for key encoder",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Batch size for key encoder training",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="Learning rate for key encoder training",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Output directory for key encoder training results",
    )

    # Use trained key encoder for feature extraction
    parser.add_argument(
        "--use-key-encoder",
        action="store_true",
        dest="experiment_use_key_encoder",
        help="Use trained key encoder model for key fingerprinting during feature extraction",
    )

    parser.add_argument(
        "--deepset",
        action="store_true",
        dest="experiment_use_deepset",
        help="Use trained key encoder model for key fingerprinting during feature extraction",
    )

    parser.add_argument(
        "--pretrained-t-network",
        type=str,
        default=None,
        dest="experiment_pretrained_t_network_path",
        help="Path to pretrained T network model (.keras file)"
    )

    parser.add_argument(
        "--freeze-t-network",
        action="store_true",
        dest="experiment_freeze_t_network",
        help="Freeze T network and train only classifier"
    )

    parser.add_argument(
        "--feature-combination",
        type=str,
        default=None,
        dest="experiment_feature_combination",
        choices=["baseline_no_cloud", "no_raw_embedding", "full_features", "cloud_no_raw"],
        help="Feature combination for ablation study: baseline_no_cloud (all local), no_raw_embedding (no p_x), full_features (all), cloud_no_raw (cloud without p_x)"
    )

    args = parser.parse_args()

    # Check if the list contains a single string with spaces and split it if necessary
    triang_args = args.experiment_triangulation_choosing
    if triang_args and len(triang_args) == 1 and len(triang_args[0].split()) > 1 :
        # Split the single string "kmeans classes" into ["kmeans", "classes"]
        args.experiment_triangulation_choosing = triang_args[0].split()
        logger.debug(f"DEBUG: Fixed triangulation args to: {args.experiment_triangulation_choosing}")

    if args.dataset_names and len(args.dataset_names[0].split()) > 1:
        args.dataset_names = args.dataset_names[0].split()
        logger.debug(f"DEBUG: Fixed dataset names args to: {args.dataset_names}")

    # Check if the list contains a single string with spaces and split it if necessary
    models = args.iim_name
    # FIX: Use models[0].split(), not models.split()
    if models and len(models) == 1 and len(models[0].split()) > 1:
        # FIX: Assign back to iim_name, NOT experiment_triangulation_choosing
        args.iim_name = models[0].split()
        logger.debug(f"DEBUG: Fixed IIM models args to: {args.iim_name}")

    # Handle calibration distributions if passed as single space-separated string
    calib_dists = args.experiment_calibration_distributions
    if calib_dists and len(calib_dists) == 1 and len(calib_dists[0].split()) > 1:
        args.experiment_calibration_distributions = calib_dists[0].split()
        logger.debug(f"DEBUG: Fixed calibration distributions to: {args.experiment_calibration_distributions}")

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

    elif config.experiment_config.to_run == consts.EXPERIMENTS.T_NETWORK_TRAINING:
        experiment_handler = TNetworkTrainingHandler

    elif config.experiment_config.to_run == consts.EXPERIMENTS.ABLATION_EXPERIMENT:
        experiment_handler = FeatureAblationExperimentHandler

    else:
        experiment_handler = ModelTrainingExperimentHandler
        if config.experiment_config.k_folds > 1:
            report_path = Path(OUTPUT_DIR_PATH) / "k_report.csv"
            experiment_handler = KModelTrainingExperimentHandler

    # Instantiate handler with appropriate arguments
    handler_kwargs = {"report_path": report_path}

    with experiment_handler(**handler_kwargs) as experiment:
        experiment.run_experiment()



if __name__ == '__main__':
    main()