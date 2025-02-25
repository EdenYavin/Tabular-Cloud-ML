import numpy as np
from tqdm import tqdm
import pandas as pd

from src.pipeline.stacking_encoding_pipeline import FeatureEngineeringPipeline
from src.cloud import CloudModel, CLOUD_MODELS
from src.encryptor.base import Encryptors
from src.encryptor import EncryptorFactory
from src.internal_model.model import InternalInferenceModelFactory
from src.internal_model.baseline import EmbeddingBaselineModelFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory
from src.dataset import DATASETS, RawDataset
from src.utils.config import config
from loguru import logger


class ExperimentHandler:

    def __init__(self):
        self.experiment_name = "stacking_multiple_clouds"
        self.n_pred_vectors = config.experiment_config.n_pred_vectors

    def run_experiment(self):

        assert len(config.cloud_config.names) > 1 # This experiment has an assumption that multiple cloud models are used


        if type(self.n_pred_vectors) is int:
            self.n_pred_vectors = [self.n_pred_vectors]


        # Create a final report with average metrics
        final_report = pd.DataFrame()

        datasets = config.dataset_config.names

        cloud_models: list[CloudModel] = [CLOUD_MODELS[name]() for name in config.cloud_config.names]

        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):
            raw_dataset: RawDataset = DATASETS[dataset_name]()

            embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name.value)
            encryptor = Encryptors(output_shape=cloud_models[0].input_shape,
                                   number_of_encryptors_to_init=config.experiment_config.n_pred_vectors,
                                   enc_base_cls=EncryptorFactory.get_model_cls()
                                   )

            X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
            logger.debug(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")


            logger.debug("#### GETTING CLOUD DATASET FULL BASELINE ####")
            cloud_acc, cloud_f1 = raw_dataset.get_cloud_model_baseline(X_train, X_test, y_train, y_test)

            logger.debug("#### GETTING RAW BASELINE PREDICTION ####")
            raw_baseline_acc, raw_baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)

            for n_pred_vectors in self.n_pred_vectors:

                logger.debug(f"CREATING THE CLOUD-TRAINSET FROM {dataset_name},"
                      f" WITH {n_pred_vectors} PREDICTION VECTORS")

                datasets_creator = FeatureEngineeringPipeline(
                    dataset_name=dataset_name,
                    cloud_models=cloud_models,
                    encryptor=encryptor,
                    embeddings_model=embedding_model,
                    n_pred_vectors=n_pred_vectors,
                    metadata=raw_dataset.metadata
                )
                datasets, emb_baseline, pred_baseline, = datasets_creator.create(X_sample, y_sample, X_test, y_test)
                logger.debug("Finished Creating the dataset")

                iim_models = config.iim_config.name
                if isinstance(iim_models, str):
                    iim_models = [iim_models]

                for iim_model in iim_models:
                    logger.info(f"############# USING {iim_model} FOR ALL BASELINES #############")
                    logger.debug(f"#### EVALUATING EMBEDDING BASELINE MODEL ####\nDataset Shape: Train - {emb_baseline.train.embeddings.shape}, Test: {emb_baseline.test.embeddings.shape}")
                    baseline_model = EmbeddingBaselineModelFactory.get_model(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=emb_baseline.train.embeddings.shape[1],
                        type=iim_model
                    )
                    baseline_model.fit(
                        emb_baseline.train.embeddings, emb_baseline.train.labels,
                    )
                    baseline_emb_acc, baseline_emb_f1 = baseline_model.evaluate(
                        emb_baseline.test.embeddings, emb_baseline.test.labels
                    )

                    logger.debug(f"#### EVALUATING PREDICTIONS BASELINE MODEL ####\nDataset Shape: Train - {pred_baseline.train.predictions.shape}, Test: {pred_baseline.test.predictions.shape}")
                    baseline_model = EmbeddingBaselineModelFactory.get_model(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=pred_baseline.train.predictions.shape[1],
                        type=iim_model
                    )
                    baseline_model.fit(
                        pred_baseline.train.predictions, pred_baseline.train.labels,
                    )
                    baseline_pred_acc, baseline_pred_f1 = baseline_model.evaluate(
                        pred_baseline.test.predictions, pred_baseline.test.labels
                    )

                    logger.debug(f"#### EVALUATING INTERNAL MODEL ####\nDataset Shape: Train - {datasets[0].train.features.shape}, Test: {datasets[0].test.features.shape}")
                    internal_model = InternalInferenceModelFactory().get_model(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=datasets[0].train.features.shape[1],
                        type=iim_model
                    )
                    internal_model.fit(
                        X=[dataset.train.features for dataset in datasets],y=datasets[0].train.labels,
                    )
                    test_acc, test_f1 = internal_model.evaluate(
                        X=[dataset.test.features for dataset in datasets],y=datasets[0].test.labels
                    )

                    logger.info(f"""
                          Cloud: {cloud_acc}, {cloud_f1}\n
                          Raw Baseline: {raw_baseline_acc}, {raw_baseline_f1}\n
                          Emb Baseline: {baseline_emb_acc}, {baseline_emb_f1}\n
                          Prediction Baseline: {baseline_pred_acc}, {baseline_pred_f1}\n
                          IIM: {test_acc}, {test_f1}\n
                          """)

                    final_report = pd.concat(
                        [
                            final_report,
                            pd.DataFrame(
                                {
                                    "exp_name": [self.experiment_name],
                                    "dataset": [dataset_name],
                                    "n_pred_vectors": [n_pred_vectors],
                                    "iim_model": [internal_model.name],
                                    "embedding": [embedding_model.name],
                                    "encryptor": [encryptor.name],
                                    "cloud_model": [str([cloud_model.name for cloud_model in cloud_models])],
                                    "raw_baseline_acc": [raw_baseline_acc],
                                    "raw_baseline_f1": [raw_baseline_f1],
                                    "emb_baseline_acc": [baseline_emb_acc],
                                    "emb_baseline_f1": [baseline_emb_f1],
                                    "pred_baseline_acc": [baseline_pred_acc],
                                    "pred_baseline_f1": [baseline_pred_f1],
                                    "iim_test_acc": [test_acc],
                                    "iim_test_f1": [test_f1]
                                }
                            )
                        ])

                del datasets  # Free up space

        return final_report
