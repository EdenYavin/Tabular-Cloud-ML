from abc import ABC, abstractmethod

from loguru import logger
import pandas as pd
import os

from src.domain.dataset import PredictionBaselineDataset, EmbeddingBaselineDataset
from src.internal_model.baseline import EmbeddingBaselineModelFactory
from src.utils.config import config
from src.utils.constansts import REPORT_PATH, EXPERIMENTS
from src.utils.helpers import get_num_classes


class ExperimentHandler(ABC):

    def __init__(self, experiment_name: str, report_path: str = REPORT_PATH):
        self.experiment_name: str = experiment_name
        if type(config.experiment_config.n_pred_vectors) is list:
            self.n_pred_vectors = config.experiment_config.n_pred_vectors
        else:
            self.n_pred_vectors = [config.experiment_config.n_pred_vectors]

        self.report_path = report_path
        if os.path.exists(self.report_path):
            try:
                self.report = pd.read_csv(report_path)
            except Exception as e:
                logger.error(f"Error loading report: {e} \n Starting a new one")
                self.report = pd.DataFrame()
        else:
            self.report = pd.DataFrame()

    def get_embedding_baseline(self, dataset: EmbeddingBaselineDataset) -> tuple[float, float]:
        logger.debug(
            f"#### EVALUATING EMBEDDING BASELINE MODEL ####\n"
            f"Dataset Shape: Train - {dataset.train.embeddings.shape}, Test: {dataset.test.embeddings.shape}")

        num_classes = get_num_classes(dataset.train.labels)
        baseline_model = EmbeddingBaselineModelFactory.get_model(
            num_classes=num_classes,
            input_shape=dataset.train.embeddings.shape[1],
            type=config.iim_config.name[0]
        )
        baseline_model.fit(
            dataset.train.embeddings, dataset.train.labels,
        )
        baseline_emb_acc, baseline_emb_f1 = baseline_model.evaluate(
            dataset.test.embeddings, dataset.test.labels
        )

        return baseline_emb_acc, baseline_emb_f1

    def get_prediction_baseline(self, dataset: PredictionBaselineDataset) -> tuple[float, float]:

        logger.debug(
            f"#### EVALUATING PREDICTIONS BASELINE MODEL ####\n"
            f"Dataset Shape: Train - {dataset.train.predictions.shape}, Test: {dataset.test.predictions.shape}")

        try:
            num_classes = get_num_classes(dataset.train.labels)
            baseline_model = EmbeddingBaselineModelFactory.get_model(
                num_classes=num_classes,
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
                    dataset_name: str, train_shape: tuple, new_train_shape: tuple, test_shape: tuple, cloud_models_names,
                    embeddings_baseline_acc: float, embeddings_baseline_f1: float,
                    # prediction_baseline_acc: float, prediction_baseline_f1: float,
                    iim_baseline_acc: float, iim_baseline_f1: float,
                    iim_model_name: str, total_params: int,
                    raw_baseline_acc: float = None, raw_baseline_f1: float = None,
                    ):

        iim_name = " ".join([iim for iim in config.iim_config.name]) if not iim_model_name else iim_model_name

        log_msg = ""
        if raw_baseline_acc:
            log_msg += f"Raw Baseline: {raw_baseline_acc}, {raw_baseline_f1}\n"

        log_msg += f"""
         Emb Baseline: {embeddings_baseline_acc}, {embeddings_baseline_f1}\n
         IIM {iim_name}: {iim_baseline_acc}, {iim_baseline_f1}\n
        """

        logger.info(log_msg)

        train_samples = config.experiment_config.n_triangulation_samples if config.encoder_config.rotating_key else 0
        new_row = {
            "exp_name": [self.experiment_name],
            "triangulation_samples": [train_samples],
            "dataset": [dataset_name],
            "train_size": [str(train_shape)],
            "new_train_size": [str(new_train_shape)],
            "test_size": [str(test_shape)],
            "iim_model": [iim_name],
            "total_params": [total_params],
            "embedding": [config.embedding_config.name],
            "encryptor": [config.encoder_config.name],
            "cloud_model": [cloud_models_names],
            # "pred_baseline_acc": [prediction_baseline_acc],
            # "pred_baseline_f1": [prediction_baseline_f1],
            "emb_baseline_acc": [embeddings_baseline_acc],
            "emb_baseline_f1": [embeddings_baseline_f1],
            "iim_test_acc": [iim_baseline_acc],
            "iim_test_f1": [iim_baseline_f1],
        }
        if raw_baseline_acc:
            new_row["raw_baseline_acc"] = [raw_baseline_acc]
            new_row["raw_baseline_f1"] = [raw_baseline_f1]

        new_row = pd.DataFrame(new_row)
        self.report = pd.concat([self.report, new_row])

        # Save results every 5 rows
        if len(self.report)  // 5 and config.experiment_config.to_run == EXPERIMENTS.INCREMENT_EVALUATION:
            self.save()

    def save(self):
        logger.info(f"Saving report to {self.report_path}")
        self.report.to_csv(self.report_path, index=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (
                config.experiment_config.to_run == EXPERIMENTS.INCREMENT_EVALUATION
                or
                config.experiment_config.to_run == EXPERIMENTS.MODEL_TRAINING
        ):
            self.save()

    @abstractmethod
    def run_experiment(self):
        pass