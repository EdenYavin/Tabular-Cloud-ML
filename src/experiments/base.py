from abc import ABC, abstractmethod
from loguru import logger
import pandas as pd

from src.dataset import RawDataset
from src.domain.dataset import PredictionBaselineDataset, EmbeddingBaselineDataset
from src.internal_model.baseline import EmbeddingBaselineModelFactory
from src.utils.config import config


class ExperimentHandler(ABC):

    def __init__(self, experiment_name: str):
        self.experiment_name: str = experiment_name
        self.n_pred_vectors = config.experiment_config.n_pred_vectors
        self.report = pd.DataFrame()

    def get_embedding_baseline(self, raw_dataset: RawDataset, dataset: EmbeddingBaselineDataset) -> tuple[float, float]:
        logger.debug(
            f"#### EVALUATING EMBEDDING BASELINE MODEL ####\n"
            f"Dataset Shape: Train - {dataset.train.embeddings.shape}, Test: {dataset.test.embeddings.shape}")
        baseline_model = EmbeddingBaselineModelFactory.get_model(
            num_classes=raw_dataset.get_n_classes(),
            input_shape=dataset.train.embeddings.shape[1],
            type=config.iim_config.name[0]  # IIM_MODELS.NEURAL_NET # The baseline will be only neural network
        )
        baseline_model.fit(
            dataset.train.embeddings, dataset.train.labels,
        )
        baseline_emb_acc, baseline_emb_f1 = baseline_model.evaluate(
            dataset.test.embeddings, dataset.test.labels
        )

        return baseline_emb_acc, baseline_emb_f1

    def get_prediction_baseline(self, raw_dataset: RawDataset, dataset: PredictionBaselineDataset) -> tuple[float, float]:

        logger.debug(
            f"#### EVALUATING PREDICTIONS BASELINE MODEL ####\n"
            f"Dataset Shape: Train - {dataset.train.predictions.shape}, Test: {dataset.test.predictions.shape}")

        try:
            baseline_model = EmbeddingBaselineModelFactory.get_model(
                num_classes=raw_dataset.get_n_classes(),
                input_shape=dataset.train.predictions.shape[1],
                type=config.iim_config.name[0]
            )
            baseline_model.fit(
                dataset.train.predictions, dataset.train.labels,
            )
            baseline_pred_acc, baseline_pred_f1 = baseline_model.evaluate(
                dataset.test.predictions, dataset.test.labels
            )
        except Exception as e:
            logger.error("Error while evaluating the Prediction baseline model. Skipping the baseline")
            logger.error(e)
            baseline_pred_acc, baseline_pred_f1 = -1, -1

        return baseline_pred_acc, baseline_pred_f1

    def log_results(self,
                    dataset_name: str, train_shape: tuple, test_shape: tuple, cloud_models_names,
                    raw_baseline_acc: float, raw_baseline_f1: float,
                    embeddings_baseline_acc: float, embeddings_baseline_f1: float,
                    prediction_baseline_acc: float, prediction_baseline_f1: float,
                    iim_baseline_acc: float, iim_baseline_f1: float,
                    iim_model_name: str = None
                    ):

        iim_name = " ".join([iim for iim in config.iim_config.name]) if not iim_model_name else iim_model_name

        logger.info(f"""
                 Raw Baseline: {raw_baseline_acc}, {raw_baseline_f1}\n
                 Emb Baseline: {embeddings_baseline_acc}, {embeddings_baseline_f1}\n
                 Prediction Baseline: {prediction_baseline_acc}, {prediction_baseline_f1}\n
                 IIM {iim_name}: {iim_baseline_acc}, {iim_baseline_f1}\n
                 """)

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