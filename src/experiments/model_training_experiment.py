import gc
import pickle


from src.internal_model import InternalInferenceModelFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from loguru import logger
from src.experiments.base import ExperimentHandler
from src.utils.helpers import get_experiment_name, get_dataset_path
from src.utils.constansts import DATASET_FILE_NAME, BASELINE_DATASET_FILE_NAME


class ModelTrainingExperimentHandler(ExperimentHandler):

    def __init__(self):
        super().__init__(get_experiment_name())

    def run_experiment(self):

        logger.info(f"Training Model Experiment: {get_experiment_name()}")

        for dataset_name in config.dataset_config.names:

            try:
                raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
                logger.debug(f"Original Dataset Size: {raw_dataset.get_dataset()[0].shape}")
                n_classes = raw_dataset.get_n_classes()
                original_size = raw_dataset.get_dataset()[0].shape
                del raw_dataset

            except:
                logger.warning(f"Error loading Dataset {dataset_name}, using default number of classes -> 2")
                n_classes, original_size = 2, 0

            for n_pred_vectors in self.n_pred_vectors:

                for model_name in config.iim_config.name:

                    logger.info(f"#### Training model experiment: "
                                f"Dataset: {dataset_name}, n_pred_vectors: {n_pred_vectors} ####\n")

                    path = get_dataset_path(dataset_name=dataset_name, n_pred_vectors=n_pred_vectors)

                    if path.exists():

                        data_path = path / DATASET_FILE_NAME
                        history_path = path / "history.pkl"
                        plot_path = path / f"{model_name}_train_plot.png"

                        with open(data_path, "rb") as f:
                            dataset = pickle.load(f)


                        internal_model = InternalInferenceModelFactory().get_model(
                            num_classes=n_classes,
                            input_shape=dataset.train.features.shape[1],
                            type=model_name
                        )
                        logger.debug(f"#### EVALUATING INTERNAL MODEL {model_name} ####\n"
                                     f" Shape: Train - {dataset.train.features.shape}, Test: {dataset.test.features.shape}")
                        internal_model.fit(
                            X=dataset.train.features, y=dataset.train.labels,
                            validation_data=(dataset.test.features, dataset.test.labels),
                        )
                        test_acc, test_f1 = internal_model.evaluate(
                            X=dataset.test.features, y=dataset.test.labels
                        )

                        internal_model.save_history(history_path)
                        internal_model.plot_history(plot_path, title=f"Loss Curve {dataset_name} Dataset,"
                                                                     f" Internal Model {model_name}"
                                                                     f", Samples: {dataset.test.features.shape[0]}")

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
                            train_shape=original_size,
                            new_train_shape=dataset.train.features.shape,
                            test_shape=dataset.test.features.shape,
                            cloud_models_names=str([cloud_model for cloud_model in config.cloud_config.names]),
                            embeddings_baseline_acc=baseline_emb_acc, embeddings_baseline_f1=embeddings_baseline_f1,
                            iim_baseline_acc=test_acc, iim_baseline_f1=test_f1,
                            iim_model_name=model_name,
                            total_params=internal_model.model.count_params(),
                            n_pred_vectors=n_pred_vectors
                        )


                        del dataset, internal_model
                        gc.collect()


        return self.report
