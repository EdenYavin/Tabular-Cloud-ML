import src.utils.constansts as consts
from src.experiments.handler import ExperimentHandler
from src.experiments.k_fold_handler import KFoldExperimentHandler
from src.utils.config import config
import pandas as pd
import os
import tensorflow as tf
import numpy as np

np.random.seed(42)

def main():


    # Use GPU only when using Decon
    if config.encoder_config.name not in consts.GPU_MODELS:
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')

    if config.experiment_config.k_folds == 1:
        experiment_handler = ExperimentHandler(config)
    else:
        experiment_handler = KFoldExperimentHandler(config)
    new_report_lines = experiment_handler.run_experiment()
    report = pd.DataFrame()

    if os.path.exists(consts.REPORT_PATH):
        report = pd.read_csv(consts.REPORT_PATH, index_col="Unnamed: 0")

    report = pd.concat([report, new_report_lines], ignore_index=True)

    print(f"Saving report to {consts.REPORT_PATH}")
    report.to_csv(consts.REPORT_PATH)

if __name__ == '__main__':
    main()