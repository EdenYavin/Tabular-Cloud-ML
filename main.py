import src.utils.constansts as consts
from src.experiments import StackingExperimentHandler, NoStackingExperimentHandler, IncrementEvalExperimentHandler, GlobalEmbeddingExperimentHandler
from src.utils.config import config
import tensorflow as tf
import numpy as np

np.random.seed(42)

def main():

    # Use GPU only when using Decon
    if config.encoder_config.name not in consts.GPU_MODELS:
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')

    if config.experiment_config.exp_type == consts.EXPERIMENTS.INCREMENT_EVALUATION:
        experiment_handler = IncrementEvalExperimentHandler

    elif config.experiment_config.exp_type == consts.EXPERIMENTS.PREDICTIONS_LEARNING:

        if len(config.cloud_config.names) > 1 and config.iim_config.stacking:
            experiment_handler = StackingExperimentHandler

        else:
            experiment_handler = NoStackingExperimentHandler

    else:
        experiment_handler = GlobalEmbeddingExperimentHandler

    with experiment_handler() as experiment:
        experiment.run_experiment()



if __name__ == '__main__':
    main()