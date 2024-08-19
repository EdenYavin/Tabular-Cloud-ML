import yaml
import src.utils.constansts as consts
from src.experiments.handler import ExperimentHandler
import pandas as pd
import os
import tensorflow as tf


def main():

    with open(consts.CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Use GPU only when using Decon
    if config['ENCRYPTOR']['name'] != 'dc':
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')

    experiment_handler = ExperimentHandler(config)

    new_report_lines = experiment_handler.run_experiment()
    report = pd.DataFrame()

    if os.path.exists(consts.REPORT_PATH):
        report = pd.read_csv(consts.REPORT_PATH, index_col="Unnamed: 0")

    report = pd.concat([report, new_report_lines], ignore_index=True)

    print(f"Saving report to {consts.REPORT_PATH}")
    report.to_csv(consts.REPORT_PATH)

if __name__ == '__main__':
    main()