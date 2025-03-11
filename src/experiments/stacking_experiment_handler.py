from loguru import logger

from src.pipeline.stacking_encoding_pipeline import StackingFeatureEngineeringPipeline
from src.cloud import CloudModel, CLOUD_MODELS
from src.encryptor.base import Encryptors
from src.encryptor import EncryptorFactory
from src.internal_model.model import StackingDenseInternalModel, StackingMixedInternalModel, StackingXGInternalModel
from src.internal_model.baseline import EmbeddingBaselineModelFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory
from src.utils.config import config
from src.dataset.loader import DataLoader
from src.experiments.base import ExperimentHandler

class StackingExperimentHandler(ExperimentHandler):

    def __init__(self):
        super().__init__("stacking_multiple_clouds")

    def run_experiment(self):

        assert len(config.cloud_config.names) > 1 # This experiment has an assumption that multiple cloud models are used


        if type(self.n_pred_vectors) is int:
            self.n_pred_vectors = [self.n_pred_vectors]


        # Data loader to lazy load each dataset in the experiment
        data_loader = DataLoader()

        cloud_input_shape = CLOUD_MODELS[config.cloud_config.names[0]].input_shape

        for raw_dataset in data_loader:

            embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y, dataset_name=raw_dataset.name)
            encryptor = Encryptors(output_shape=cloud_input_shape,
                                   number_of_encryptors_to_init=config.experiment_config.n_pred_vectors,
                                   enc_base_cls=EncryptorFactory.get_model_cls()
                                   )

            X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
            logger.debug(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")

            logger.debug("#### GETTING RAW BASELINE PREDICTION ####")
            raw_baseline_acc, raw_baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)

            for n_pred_vectors in self.n_pred_vectors:

                logger.debug(f"CREATING THE CLOUD-TRAINSET FROM {raw_dataset.name},"
                      f" WITH {n_pred_vectors} PREDICTION VECTORS")

                datasets_creator = StackingFeatureEngineeringPipeline(
                    dataset_name=raw_dataset.name,
                    encryptor=encryptor,
                    embeddings_model=embedding_model,
                    n_pred_vectors=n_pred_vectors,
                    metadata=raw_dataset.metadata
                )
                datasets, emb_baseline, pred_baseline, = datasets_creator.create(X_sample, y_sample, X_test, y_test)
                logger.debug("Finished Creating the dataset")

                # Log size for the final report
                train_shape = X_sample.shape
                test_shape = X_test.shape
                del X_test, X_sample, y_test, y_sample

                logger.info(f"############# USING {config.iim_config.name} FOR ALL BASELINES #############")
                baseline_emb_acc, baseline_emb_f1 = self.get_embedding_baseline(raw_dataset, emb_baseline)
                baseline_pred_acc, baseline_pred_f1 = self.get_prediction_baseline(raw_dataset, pred_baseline)


                internal_models = [
                    StackingDenseInternalModel(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=datasets[0].train.features.shape[1],
                    ),
                    StackingMixedInternalModel(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=datasets[0].train.features.shape[1],
                    ),
                    StackingXGInternalModel(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=datasets[0].train.features.shape[1],
                    )

                ]

                for iim_model in internal_models:


                    logger.debug(f"#### EVALUATING INTERNAL MODEL: {iim_model.name} ####\nDataset Shape: Train - {datasets[0].train.features.shape}, Test: {datasets[0].test.features.shape}")

                    iim_model.fit(
                        X=[dataset.train.features for dataset in datasets],y=datasets[0].train.labels,
                    )
                    test_acc, test_f1 = iim_model.evaluate(
                        X=[dataset.test.features for dataset in datasets],y=datasets[0].test.labels
                    )

                    self.log_results(
                        raw_dataset.name, train_shape, test_shape,
                        str([cloud_model for cloud_model in config.cloud_config.names]),
                        raw_baseline_acc, raw_baseline_f1,
                        baseline_emb_acc, baseline_emb_f1,
                        baseline_pred_acc, baseline_pred_f1,
                        test_acc, test_f1,
                        iim_model.name
                    )

                del datasets  # Free up space

        return self.report
