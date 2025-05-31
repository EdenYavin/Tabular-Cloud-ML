import gc
import pathlib
import pickle


from src.internal_model import InternalInferenceModelFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from loguru import logger
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import OUTPUT_DIR_PATH, DATASET_FILE_NAME, BASELINE_DATASET_FILE_NAME


class ModelTrainingExperimentHandler(ExperimentHandler):

    def __init__(self):
        super().__init__(get_experiment_name())

    def run_experiment(self):


        for dataset_name in config.dataset_config.names:

            raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
            logger.debug(f"Original Dataset Size: {raw_dataset.get_dataset()[0].shape}")

            for n_pred_vectors in self.n_pred_vectors:

                for model_name in config.iim_config.name:

                    logger.info(f"\n#### Training model experiment: "
                                f"Dataset: {dataset_name}, n_pred_vectors: {n_pred_vectors} ####\n")

                    path = get_dataset_path(dataset_name=dataset_name, n_pred_vectors=n_pred_vectors)

                    if path.exists():

                        data_path = path / DATASET_FILE_NAME

                        with open(data_path, "rb") as f:
                            dataset = pickle.load(f)

                        logger.debug(f"#### EVALUATING INTERNAL MODEL {model_name}####\n"
                                     f" Shape: Train - {dataset.train.features.shape}, Test: {dataset.test.features.shape}")
                        internal_model = InternalInferenceModelFactory().get_model(
                            num_classes=raw_dataset.get_n_classes(),
                            input_shape=dataset.train.features.shape[1],
                            type=config.iim_config.name[0]
                        )
                        internal_model.fit(
                            X=dataset.train.features, y=dataset.train.labels,
                            validation_data=(dataset.test.features, dataset.test.labels),
                        )
                        test_acc, test_f1 = internal_model.evaluate(
                            X=dataset.test.features, y=dataset.test.labels
                        )


                        if config.iim_config.train_baseline:
                            baseline_path = path / BASELINE_DATASET_FILE_NAME
                            with open(baseline_path, "rb") as f:
                                baseline_dataset = pickle.load(f)

                            logger.info(f"Baseline flag is set, training the baseline embedding model.\n "
                                        f"Dataset size: {baseline_dataset.train.features.shape}")

                            baseline_emb_acc, embeddings_baseline_f1 = self.get_embedding_baseline(baseline_dataset)
                        else:
                            baseline_emb_acc, embeddings_baseline_f1 = 0, 0

                        self.log_results(
                            dataset_name=dataset_name,
                            train_shape=dataset.train.features.shape,
                            new_train_shape=dataset.train.features.shape,
                            test_shape=dataset.test.features.shape,
                            cloud_models_names=str([cloud_model for cloud_model in config.cloud_config.names]),
                            embeddings_baseline_acc=baseline_emb_acc, embeddings_baseline_f1=embeddings_baseline_f1,
                            iim_baseline_acc=test_acc, iim_baseline_f1=test_f1,
                            iim_model_name=internal_model.name,
                        )


                    del dataset
                    gc.collect()

            del raw_dataset
            gc.collect()


        return self.report
