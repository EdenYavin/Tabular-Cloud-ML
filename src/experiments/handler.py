from src.dataset.meta_dataset.creator import Dataset
from src.cloud.models import CloudModels
from src.encryptor.model import Encryptor
from src.internal_model.model import TabularInternalModel as InternalModel

import src.utils.constansts as consts

import pandas as pd

class ExperimentHandler:

    def __init__(self, experiment_config: dict):

        self.experiment_name = experiment_config[consts.CONFIG_EXPERIMENT_SECTION].get("name")
        self.n_pred_vectors = experiment_config[consts.CONFIG_EXPERIMENT_SECTION].get("n_pred_vectors", 1)
        self.n_noise_samples = experiment_config[consts.CONFIG_EXPERIMENT_SECTION].get("n_noise_samples", 3)
        self.config = experiment_config

    def run_experiment(self):
        cloud_models = CloudModels(self.config[consts.CONFIG_CLOUD_MODEL_SECTION])
        encryptor = Encryptor(self.config[consts.CONFIG_ENCRYPTOR_SECTION])
        internal_model = InternalModel(self.config[consts.CONFIG_INN_SECTION])

        dataset = Dataset(
            config=self.config[consts.CONFIG_DATASET_SECTION],
            cloud_models=cloud_models,
            encryptor=encryptor,
            n_pred_vectors=self.n_pred_vectors,
            n_noise_samples=self.n_noise_samples,
        ).create()

        internal_model.fit(*dataset.train)
        train_acc, train_f1 = internal_model.evaluate(*dataset.train)
        test_acc, test_f1 = internal_model.evaluate(*dataset.test)

        report = pd.DataFrame(columns=[
            "dataset","train_size_ratio", "iim_model", "cloud_models", "train_accuracy", "train_f1", "test_accuracy", "test_f1",
            "n_pred_vectors", "n_noise_sample", "train_size_ratio", "exp_name"
        ])

        report["dataset"] = dataset.name
        report["train_size_ratio"] = dataset.split_ratio
        report["iim_model"] = internal_model.name
        report["cloud_models"] = cloud_models.name
        report["n_pred_vectors"] = self.n_pred_vectors
        report["n_noise_sample"] = self.n_noise_samples
        report["exp_name"] = self.experiment_name
        report["train_accuracy"] = train_acc
        report["train_f1"] = train_f1
        report["test_accuracy"] = test_acc
        report["test_f1"] = test_f1

        return report




