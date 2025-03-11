from abc import ABC, abstractmethod
from loguru import logger
import pandas as pd

from src.utils.config import config


class ExperimentHandler(ABC):

    def __init__(self, experiment_name: str):
        self.experiment_name: str = experiment_name
        self.n_pred_vectors = config.experiment_config.n_pred_vectors
        self.report = pd.DataFrame()

    def log_results(self,
                    dataset_name: str, train_shape: tuple, test_shape: tuple, cloud_models_names,
                    raw_baseline_acc: float, raw_baseline_f1: float,
                    embeddings_baseline_acc: float, embeddings_baseline_f1: float,
                    prediction_baseline_acc: float, prediction_baseline_f1: float,
                    iim_baseline_acc: float, iim_baseline_f1: float,
                    iim_model_name: str = None
                    ):

        logger.info(f"""
                 Raw Baseline: {raw_baseline_acc}, {raw_baseline_f1}\n
                 Emb Baseline: {embeddings_baseline_acc}, {embeddings_baseline_f1}\n
                 Prediction Baseline: {prediction_baseline_acc}, {prediction_baseline_f1}\n
                 IIM {config.iim_config.name}: {iim_baseline_acc}, {iim_baseline_f1}\n
                 """)

        iim_name = " ".join([iim for iim in config.iim_config.name]) if not iim_model_name else iim_model_name
        new_row = pd.DataFrame({
            "exp_name": [self.experiment_name],
            "dataset": [dataset_name],
            "train_size": [str(train_shape)],
            "test_size": [str(test_shape)],
            "n_pred_vectors": [self.n_pred_vectors],
            "n_noise_sample": [1],
            "iim_model": [iim_name],
            "embedding": [config.embedding_config.name],
            "encryptor": [config.encoder_config.name],
            "cloud_model": [cloud_models_names],
            "raw_baseline_acc": [raw_baseline_acc],
            "raw_baseline_f1": [raw_baseline_f1],
            "emb_baseline_acc": [embeddings_baseline_acc],
            "emb_baseline_f1": [embeddings_baseline_f1],
            "pred_baseline_acc": [prediction_baseline_acc],
            "pred_baseline_f1": [prediction_baseline_f1],
            "iim_test_acc": [iim_baseline_acc],
            "iim_test_f1": [iim_baseline_f1]
        })
        self.report = pd.concat([self.report, new_row])

    @abstractmethod
    def run_experiment(self):
        pass