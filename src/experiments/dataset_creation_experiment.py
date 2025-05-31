import gc
import os
import pathlib
import pickle

from tqdm import tqdm

from src.pipeline.feature_engineering_pipeline import DatasetCreation as FeatureEngineeringPipeline
from src.cloud import CLOUD_MODELS
from src.encryptor import EncryptorFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from loguru import logger
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import  DATASET_FILE_NAME, BASELINE_DATASET_FILE_NAME


class DatasetCreationHandler(ExperimentHandler):

    def __init__(self):
        super().__init__(get_experiment_name())

    def run_experiment(self):

        assert len(config.cloud_config.names) >= 1

        if type(self.n_pred_vectors) is int:
            self.n_pred_vectors = [self.n_pred_vectors]

        datasets = config.dataset_config.names

        # Get the output for the cloud model
        cloud_model_output = CLOUD_MODELS[config.cloud_config.names[0]].input_shape

        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):
            with logger.contextualize(dataset=dataset_name):

                raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)

                embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y, dataset_name=dataset_name)
                encryptor = EncryptorFactory.get_model(dataset_name=dataset_name, output_shape=cloud_model_output)

                X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
                logger.debug(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")

                for n_pred_vectors in self.n_pred_vectors:

                    logger.debug(f"Experiment name is {self.experiment_name}, Dataset is {dataset_name} and"
                          f" will have {n_pred_vectors} prediction vector for each cloud model")

                    dataset_creator = FeatureEngineeringPipeline(
                        dataset_name=dataset_name,
                        encryptor=encryptor,
                        embeddings_model=embedding_model,
                        n_pred_vectors=n_pred_vectors,
                        metadata=raw_dataset.metadata
                    )

                    dataset, emb_baseline_dataset = (
                        dataset_creator.create(X_sample, y_sample, X_test, y_test)
                    )

                    # # Log size for the final report
                    # train_shape = dataset_creator.original_train_size or X_sample.shape
                    # test_shape = X_test.shape
                    # del X_test, X_sample, y_test, y_sample, dataset_creator, raw_dataset
                    #
                    # logger.info(f"############# USING {config.iim_config.name} FOR ALL BASELINES #############")
                    # baseline_emb_acc, baseline_emb_f1 = self.get_embedding_baseline(emb_baseline_dataset)
                    # del emb_baseline_dataset # Free up memory
                    #
                    # # if len(pred_baseline_dataset.train.predictions) > 0:
                    # #     # If we are not using the use_pred flag in the config, the prediction dataset will be empty
                    # #     baseline_pred_acc, baseline_pred_f1 = self.get_prediction_baseline(pred_baseline_dataset)
                    # #     del pred_baseline_dataset # Free up memory
                    # # else:
                    # #     baseline_pred_acc, baseline_pred_f1 = 0, 0
                    #
                    # logger.debug(f"#### EVALUATING INTERNAL MODEL ####\nDataset {dataset_name} Shape: Train - {dataset.train.features.shape}, Test: {dataset.test.features.shape}")
                    # internal_model = InternalInferenceModelFactory().get_model(
                    #     num_classes=n_classes,
                    #     input_shape=dataset.train.features.shape[1],
                    #     type=config.iim_config.name[0]
                    # )
                    # internal_model.fit(
                    #     X=dataset.train.features, y=dataset.train.labels,
                    #     validation_data=(dataset.test.features, dataset.test.labels),
                    # )
                    # test_acc, test_f1 = internal_model.evaluate(
                    #     X=dataset.test.features, y=dataset.test.labels
                    # )
                    #
                    # self.log_results(
                    #     dataset_name=dataset_name,
                    #     train_shape=train_shape,
                    #     new_train_shape=dataset.train.features.shape,
                    #     test_shape=test_shape,
                    #     cloud_models_names=str([cloud_model for cloud_model in config.cloud_config.names]),
                    #     embeddings_baseline_acc=baseline_emb_acc, embeddings_baseline_f1=baseline_emb_f1,
                    #     prediction_baseline_acc=-1, prediction_baseline_f1=-1,
                    #     iim_baseline_acc=test_acc, iim_baseline_f1=test_f1,
                    #     iim_model_name=internal_model.name,
                    # )
                    path = get_dataset_path(dataset_name, n_pred_vectors)
                    os.makedirs(path, exist_ok=True)

                    logger.debug("Finished Creating the dataset.\n"
                                 f"Saving to {path}")

                    with open(path / BASELINE_DATASET_FILE_NAME, "wb") as f:
                        pickle.dump(emb_baseline_dataset, f)

                    with open(path / DATASET_FILE_NAME, "wb") as f:
                        pickle.dump(dataset, f)


                    del dataset, emb_baseline_dataset
                    gc.collect()


        return self.report
