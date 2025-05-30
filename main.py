import argparse

import src.utils.constansts as consts
from src.experiments import DatasetCreationHandler, IncrementEvalExperimentHandler, ModelTrainingExperimentHandler
from src.utils.config import config, update_config_from_args
import tensorflow as tf
import numpy as np

from src.utils.constansts import EXPERIMENTS

np.random.seed(42)

def main():

    parser = argparse.ArgumentParser(description="Run experiments with specified configurations.")
    parser.add_argument("--iim-train-baseline", action="store_true", help="Enable baseline mode.")
    parser.add_argument("--experiment-to-run", type=EXPERIMENTS, choices=list(EXPERIMENTS), help="Experiment type: training or dataset.")
    parser.add_argument(    "--experiment-use-preds",
    action="store_true",  # Sets to True if flag is present
    help="Use the cloud as features or not. (default: False)")
    parser.add_argument("--encoder-rotating-key",
    action="store_true",  # Sets to True if flag is present
    help="Use triangulation samples or not")

    args = parser.parse_args()
    update_config_from_args(config, args)

    # Use GPU only when using Decon
    if config.encoder_config.name not in consts.GPU_MODELS:
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')

    if config.experiment_config.to_run == consts.EXPERIMENTS.INCREMENT_EVALUATION:
        experiment_handler = IncrementEvalExperimentHandler

    elif config.experiment_config.to_run == consts.EXPERIMENTS.DATASET_CREATION:
        experiment_handler = DatasetCreationHandler

    else:
        experiment_handler = ModelTrainingExperimentHandler


    with experiment_handler() as experiment:
        experiment.run_experiment()



if __name__ == '__main__':
    main()