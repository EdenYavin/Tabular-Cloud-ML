from tqdm import tqdm
import pandas as pd

from src.pipeline.encoding_pipeline import Pipeline
from src.cloud import CloudModel, CLOUD_MODELS
from src.encryptor.base import BaseEncryptor, Encryptors
from src.encryptor import EncryptorFactory
from src.internal_model.model import InternalInferenceModelFactory
from src.internal_model.baseline import EmbeddingBaselineModelFactory
from src.embeddings import EmbeddingsFactory
from src.utils.db import RawSplitDBFactory
from src.dataset import DATASETS, RawDataset
from src.utils.config import config
import src.utils.constansts as consts

class ExperimentHandler:

    def __init__(self):
        w_emb = "w_emb" if config.experiment_config.use_embedding else "wo_emb"
        w_noise_labels = "w_noise_labels" if config.experiment_config.use_labels else "wo_noise_labels"
        w_pred = "w_pred" if config.experiment_config.use_preds else "wo_pred"
        self.experiment_name = f"{w_emb}_{w_noise_labels}_{w_pred}"
        self.n_pred_vectors = config.experiment_config.n_pred_vectors
        self.n_noise_samples = config.experiment_config.n_noise_samples

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

            cloud_model: CloudModel = CLOUD_MODELS[config.cloud_config.name](
                num_classes=raw_dataset.get_n_classes()
            )
            embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y)
            # encryptor: BaseEncryptor = EncryptorFactory().get_model(
            #     output_shape=(1, *cloud_model.input_shape),
            # )
            encryptor = Encryptors(output_shape=(1, *cloud_model.input_shape),
                                   number_of_encryptors_to_init=config.experiment_config.n_pred_vectors,
                                   enc_base_cls=EncryptorFactory.get_model_cls()
                                   )

            X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
            print(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")


            cloud_model.fit(X_train, y_train, **raw_dataset.metadata)

            print("#### GETTING CLOUD DATASET FULL BASELINE####")
            cloud_acc, cloud_f1 = raw_dataset.get_cloud_model_baseline(X_train, X_test, y_train, y_test)

            print("#### GETTING RAW BASELINE PREDICTION ####")
            raw_baseline_acc, raw_baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)

            for n_noise_samples in self.n_noise_samples:
                for n_pred_vectors in self.n_pred_vectors:

                    print(f"CREATING THE CLOUD-TRAINSET FROM {dataset_name},"
                          f" WITH {n_noise_samples} NOISE SAMPLES AND {n_pred_vectors} PREDICTION VECTORS")

                    dataset_creator = Pipeline(
                        dataset_name=dataset_name,
                        cloud_models=cloud_model,
                        encryptor=encryptor,
                        embeddings_model=embedding_model,
                        n_noise_samples=n_noise_samples,
                        n_pred_vectors=n_pred_vectors,
                        metadata=raw_dataset.metadata
                    )
                    dataset = dataset_creator.create(X_sample, y_sample, X_test, y_test)
                    print("Finished Creating the dataset")

                    print("#### GETTING EMBEDDING BASELINE PREDICTION ####")

                    baseline_model = EmbeddingBaselineModelFactory.get_model(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=dataset[consts.IIM_BASELINE_TRAIN_SET_TOKEN][0].shape[1:],
                    )
                    baseline_model.fit(
                        *dataset[consts.IIM_BASELINE_TRAIN_SET_TOKEN]
                    )
                    baseline_emb_acc, baseline_emb_f1 = baseline_model.evaluate(*dataset[consts.IIM_BASELINE_TEST_SET_TOKEN])

                    internal_model = InternalInferenceModelFactory().get_model(
                        num_classes=raw_dataset.get_n_classes(),
                        input_shape=dataset[consts.IIM_TRAIN_SET_TOKEN][0].shape[1],  # Only give the number of features
                    )

                    print(f"Training the IIM {internal_model.name} Model")
                    internal_model.fit(
                        *dataset[consts.IIM_TRAIN_SET_TOKEN]
                    )
                    train_acc, train_f1 = internal_model.evaluate(*dataset[consts.IIM_TRAIN_SET_TOKEN])
                    test_acc, test_f1 = internal_model.evaluate(*dataset[consts.IIM_TEST_SET_TOKEN])


                    print(f"""
                          Cloud: {cloud_acc}, {cloud_f1}\n
                          Raw Baseline: {raw_baseline_acc}, {raw_baseline_f1}\n
                          Emb Baseline: {baseline_emb_acc}, {baseline_emb_f1}\n
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
                                    "embedding": [embedding_model.name],
                                    "encryptor": [encryptor.name],
                                    "cloud_model": [cloud_model.name],
                                    "iim_train_acc": [train_acc],
                                    "iim_train_f1": [train_f1],
                                    "iim_test_acc": [test_acc],
                                    "iim_test_f1": [test_f1],
                                    "raw_baseline_acc": [raw_baseline_acc],
                                    "raw_baseline_f1": [raw_baseline_f1],
                                    "emb_baseline_acc": [baseline_emb_acc],
                                    "emb_baseline_f1": [baseline_emb_f1],
                                    "cloud_acc": [cloud_acc],
                                    "cloud_f1": [cloud_f1],

                                }
                            )
                        ])


        return final_report
