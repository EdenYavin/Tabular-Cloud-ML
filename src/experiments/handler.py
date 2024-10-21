from tqdm import tqdm
import pandas as pd

from src.dataset.cloud_dataset.creator import Dataset
from src.cloud import CloudModels, CLOUD_MODELS
from src.encryptor import BaseEncryptor, EncryptorFactory
from src.internal_model.model import InternalInferenceModelFactory
from src.embeddings import EmbeddingsFactory
from src.dataset.raw import DATASETS, RawDataset
from src.utils.config import config


class ExperimentHandler:

    def __init__(self):
        self.experiment_name = config.experiment_config.name
        self.n_pred_vectors = config.experiment_config.n_pred_vectors
        self.n_noise_samples = config.experiment_config.n_noise_samples
        self.k_folds = config.experiment_config.k_folds

    def run_experiment(self):

        # For dynamic run time - the list type is preferred
        if type(self.n_noise_samples) is int:
            self.n_noise_samples = [self.n_noise_samples]

        if type(self.n_pred_vectors) is int:
            self.n_pred_vectors = [self.n_pred_vectors]


        # Create a final report with average metrics
        final_report = pd.DataFrame()

        datasets = config.dataset_config.names


        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):
            raw_dataset: RawDataset = DATASETS[dataset_name]()

            embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X)
            encryptor: BaseEncryptor = EncryptorFactory().get_model(
                output_shape=(1, *embedding_model.output_shape),
            )

            X_train, X_test, X_sample, y_train, y_test, y_sample = raw_dataset.get_split(force_new_split=False)
            print(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")

            cloud_models: CloudModels = CLOUD_MODELS[config.cloud_config.name](
                num_classes=raw_dataset.get_n_classes()
            )
            cloud_models.fit(X_train, y_train, **raw_dataset.metadata)

            print("#### GETTING DATASET FULL BASELINE####")
            cloud_acc, cloud_f1 = raw_dataset.get_cloud_model_baseline(X_train, X_test, y_train, y_test)

            print("#### GETTING BASELINE PREDICTION ####")
            baseline_acc, baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)

            for n_noise_samples in self.n_noise_samples:
                for n_pred_vectors in self.n_pred_vectors:

                    print(f"CREATING THE CLOUD-TRAINSET FROM {dataset_name},"
                          f" WITH {n_noise_samples} NOISE SAMPLES AND {n_pred_vectors} PREDICTION VECTORS")

                    dataset_creator = Dataset(
                        dataset_name=dataset_name,
                        cloud_models=cloud_models,
                        encryptor=encryptor,
                        embeddings_model=embedding_model,
                        n_noise_samples=n_noise_samples,
                        n_pred_vectors=n_pred_vectors,
                        metadata=raw_dataset.metadata
                    )
                    dataset = dataset_creator.create(X_sample, y_sample, X_test, y_test)
                    print("Finished Creating the dataset")
                    print(f"##### CLOUD DATASET SIZE - {dataset['train'][0].shape} ###########")

                    internal_model = InternalInferenceModelFactory().get_model(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=dataset['train'][0].shape[1],  # Only give the number of features
                    )

                    print(f"Training the IIM {internal_model.name} Model")
                    internal_model.fit(
                        *dataset['train']
                    )
                    train_acc, train_f1 = internal_model.evaluate(*dataset['train'])
                    test_acc, test_f1 = internal_model.evaluate(*dataset['test'])


                    print(f"""
                          Cloud: {cloud_acc}, {cloud_f1}\n
                          Baseline: {baseline_acc}, {baseline_f1}\n
                          IIM: {test_acc}, {test_f1}\n
                          """)

                    final_report = pd.concat(
                        [
                            final_report,
                            pd.DataFrame(
                                {
                                    "exp_name": [self.experiment_name],
                                    "dataset": [dataset_name],
                                    "train_size_ratio": [dataset_creator.split_ratio],
                                    "n_pred_vectors": [n_pred_vectors],
                                    "n_noise_sample": [n_noise_samples],
                                    "iim_model": [internal_model.name],
                                    "encryptor": [encryptor.name],
                                    "cloud_model": [cloud_models.name],
                                    "iim_train_acc": [train_acc],
                                    "iim_train_f1": [train_f1],
                                    "iim_test_acc": [test_acc],
                                    "iim_test_f1": [test_f1],
                                    "baseline_acc": [baseline_acc],
                                    "baseline_f1": [baseline_f1],
                                    "cloud_acc": [cloud_acc],
                                    "cloud_f1": [cloud_f1],

                                }
                            )
                        ])


        return final_report




