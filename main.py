import src.utils.constansts as consts
from src.experiments.k_fold_handler import KFoldExperimentHandler
from src.experiments.global_embedding_experiment import GlobalEmbeddingExperimentHandler
from src.experiments.stacking_experiment_handler import StackingExperimentHandler
from src.experiments.multiple_cm_experiment_handler import NoStackingExperimentHandler as MultipleCmExperimentHandler
from src.utils.config import config
import pandas as pd
import os
import tensorflow as tf
import numpy as np

np.random.seed(42)

def main():
    report = pd.DataFrame()

    # Use GPU only when using Decon
    if config.encoder_config.name not in consts.GPU_MODELS:
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')

    if config.experiment_config.exp_type == consts.EXPERIMENTS.PREDICTIONS_LEARNING:

        if len(config.cloud_config.names) > 1 and config.experiment_config.stacking:
            experiment_handler = StackingExperimentHandler()

        else:
            experiment_handler = MultipleCmExperimentHandler()

        new_report_lines = experiment_handler.run_experiment()
        report_path = consts.REPORT_PATH


    else: # Global embedding experiment
        experiment_handler = GlobalEmbeddingExperimentHandler()
        new_report_lines = experiment_handler.run_experiment()
        report_path = consts.GLOBAL_EMB_REPORT_PATH

    if os.path.exists(report_path):
        report = pd.read_csv(consts.REPORT_PATH, index_col="Unnamed: 0")

    report = pd.concat([report, new_report_lines], ignore_index=True)

    print(f"Saving report to {report_path}")
    report.to_csv(report_path)

if __name__ == '__main__':
    main()