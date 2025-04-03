from tqdm import tqdm
from tqdm.auto import trange

from src.pipeline.no_stacking_encoding_pipeline import NoStackingFeatureEngineeringPipeline as FeatureEngineeringPipeline
from src.cloud import CloudModel, CLOUD_MODELS
from src.encryptor.base import Encryptors
from src.encryptor import EncryptorFactory
from src.internal_model.base import InternalInferenceModelFactory
from src.embeddings import EmbeddingsFactory
from src.utils.constansts import EVAL_DATASET_REPORT_PATH
from src.utils.db import RawSplitDBFactory
from src.dataset import DatasetFactory, RawDataset
from src.utils.config import config
from loguru import logger
from src.experiments.base import ExperimentHandler


class IncrementDatasetBatch:

    def __init__(self, dataset_name: str, number_of_folds = 10):

        raw_dataset: RawDataset = DatasetFactory().get_dataset(dataset_name)
        X_trian, X_test, _, y_train, y_test, _ = RawSplitDBFactory.get_db(raw_dataset).get_split()

        self.X = X_trian
        self.X_test = X_test
        self.y = y_train
        self.y_test = y_test
        self.metadata = raw_dataset.metadata
        self.num_classes = raw_dataset.get_n_classes()

        self.dataset_size = len(self.X)
        self.number_of_folds = number_of_folds
        self.batch_size = self.dataset_size // self.number_of_folds
        self.current_end_index = 0
        self.current_batch_index = 0
        self.progress_bar = trange(
            self.number_of_folds, desc=f"Increment Batches for {dataset_name}", unit="batch"
        )


    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next incremental batch of X and y.

        Returns:
            tuple: A tuple containing the next batch of X and y.

        Raises:
            StopIteration: If all batches have been iterated through.
        """
        if self.current_batch_index >= self.number_of_folds:
            raise StopIteration

        self.current_end_index += self.batch_size
        if self.current_end_index > self.dataset_size:
            self.current_end_index = self.dataset_size

        start_index = 0  # Always start from index 0
        end_index = self.current_end_index

        X_batch = self.X[start_index:end_index]
        y_batch = self.y[start_index:end_index]

        self.current_batch_index += 1
        self.progress_bar.update(1)

        return X_batch, self.X_test, y_batch, self.y_test

    def reset(self):
        self.current_end_index = 0
        self.current_batch_index = 0
        self.progress_bar.reset()

class IncrementEvalExperimentHandler(ExperimentHandler):

    def __init__(self):
        super().__init__("no_stacking_multiple_clouds", EVAL_DATASET_REPORT_PATH)


    def run_experiment(self):

        assert len(config.cloud_config.names) >= 1


        datasets = config.dataset_config.names

        cloud_models: list[CloudModel] = [CLOUD_MODELS[name]() for name in config.cloud_config.names]


        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):
            increment_batch_iterator = IncrementDatasetBatch(dataset_name, number_of_folds=10)

            for X_train, X_test, y_train, y_test in increment_batch_iterator:
                # Log size for the final report
                train_shape = X_train.shape
                test_shape = X_test.shape

                logger.info(f"#### CURRENT DATASET BATCH SIZE: TRAIN: {train_shape} TEST: {test_shape}")


                embedding_model = EmbeddingsFactory().get_model(X=X_train, y=y_train, dataset_name=dataset_name)
                encryptor = Encryptors(dataset_name=dataset_name,
                                       output_shape=cloud_models[0].input_shape,
                                       number_of_encryptors_to_init=config.experiment_config.n_pred_vectors,
                                       enc_base_cls=EncryptorFactory.get_model_cls()
                                       )

                dataset_creator = FeatureEngineeringPipeline(
                    dataset_name=dataset_name,
                    encryptor=encryptor,
                    embeddings_model=embedding_model,
                    n_pred_vectors=1,
                    metadata=increment_batch_iterator.metadata
                )
                dataset, emb_baseline_dataset, pred_baseline_dataset, = dataset_creator.create(X_train, y_train, X_test, y_test)




                logger.info(f"############# USING {config.iim_config.name} FOR ALL BASELINES #############")
                baseline_emb_acc, baseline_emb_f1 = self.get_embedding_baseline(emb_baseline_dataset)
                del emb_baseline_dataset # Free up memory

                baseline_pred_acc, baseline_pred_f1 = self.get_prediction_baseline(pred_baseline_dataset)
                del pred_baseline_dataset # Free up memory

                logger.debug(f"#### EVALUATING INTERNAL MODEL ####\nDataset Shape: Train - {dataset.train.features.shape}, Test: {dataset.test.features.shape}")
                internal_model = InternalInferenceModelFactory().get_model(
                    num_classes=increment_batch_iterator.num_classes,
                    input_shape=dataset.train.features.shape[1],
                    type=config.iim_config.name[0]
                )
                internal_model.fit(
                    X=dataset.train.features, y=dataset.train.labels,
                )
                test_acc, test_f1 = internal_model.evaluate(
                    X=dataset.test.features, y=dataset.test.labels
                )

                self.log_results(
                    dataset_name, train_shape, test_shape,
                    str([cloud_model for cloud_model in config.cloud_config.names]),
                    baseline_pred_acc, baseline_pred_f1,
                    baseline_emb_acc, baseline_emb_f1,
                    test_acc, test_f1,
                    internal_model.name
                )

            del dataset # Free up space


        return self.report
