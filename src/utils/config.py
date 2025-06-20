import argparse

from pydantic import BaseModel, Field
from src.utils.constansts import (EMBEDDING_TYPES, ENCODERS_TYPES, IIM_MODELS, CLOUD_MODELS, EXPERIMENTS,
                                  HARD_DATASETS, LARGE_DATASETS, ALL_DATASETS,
                                    PMLB_DATASETS_IMBALANCE, PMLB_DATASETS, DATASETS
                                  )


class Config(BaseModel):

    class EmbeddingConfig(BaseModel):
        name: str = Field(description="Name of the embedding model to use")
        base_model: str = Field(description="Used in case the embedding model apply vary of models like different resnets", default="resnet101")

    class EncoderConfig(BaseModel):
        name: str = Field(description="Name of encryptor / encoder model to use", default=ENCODERS_TYPES.DCONV)
        rotating_key: bool = Field(description="A flag indicating if the encryptor should switch to a new encryption key on each new samples", default=False)


    class DatasetConfig(BaseModel):
        names: list = Field(description="The datasets to run the experiments on")
        split_ratio : float = Field(description="How much of the original train set (90%) will be used to train the IIM")
        one_hot: bool = Field(description="A flag to indicate if the ground truth labels should be one-hot encoded", default=False)
        batch_size: int = Field(description="Batch size to accumulate", default=64)

    class IIMConfig(BaseModel):

        class NEURAL_NET_CONFIG(BaseModel):
            epochs: int = 20
            batch_size: int = 8
            dropout: float = 0.3

        neural_net_config: NEURAL_NET_CONFIG = Field(description="Neural network config", default=NEURAL_NET_CONFIG())
        name: list[str] | str = Field(description="IIM model to use. Can be multiple models", default=IIM_MODELS.NEURAL_NET)
        stacking: bool = Field(description="A flag to indicate if the stacking should be used in training the IIM")
        train_baseline: bool = Field(description="A flag to indicate if the baseline should be used in training the IIM")


    class CloudModelsConfig(BaseModel):
        names: list[str] = Field(description="Cloud model to use")

    class ExperimentConfig(BaseModel):
        use_embedding: bool = Field(description="A flag to indicate if the embedding should be used in training the IIM")
        n_pred_vectors: int = Field(description="Number of prediction vectors to query from the cloud models")
        n_triangulation_samples: int = Field(description="Number samples to sample from the dataset to use for the triangulation")
        k_folds : int = Field(description="Number of folds to use for cross-validation. If 1 - No k-fold", default=1)
        to_run: str = Field(description="type of the experiment - embedding learning, or prediction learning")


    experiment_config: ExperimentConfig = ExperimentConfig(n_triangulation_samples=5, n_pred_vectors=1, k_folds=1,
                                                           use_embedding=False,
                                                           to_run=EXPERIMENTS.DATASET_CREATION,
                                                           )
    cloud_config: CloudModelsConfig = CloudModelsConfig(names=[
        # CLOUD_MODELS.EFFICIENTNET, CLOUD_MODELS.MOBILE_NET, CLOUD_MODELS.Xception,
        # CLOUD_MODELS.DENSENET, CLOUD_MODELS.VGG16
        CLOUD_MODELS.Xception
    ])
    iim_config: IIMConfig = IIMConfig(name=[IIM_MODELS.LSTM], stacking=False, train_baseline=False,
                                      neural_net_config=IIMConfig.NEURAL_NET_CONFIG(
                                          batch_size=10,
                                          dropout=0,
                                          epochs=30
                                      ))
    dataset_config: DatasetConfig = DatasetConfig(split_ratio=1,
                                                  names=PMLB_DATASETS,
                                                  batch_size=100
                                                  )
    embedding_config: EmbeddingConfig = EmbeddingConfig(name=EMBEDDING_TYPES.SPARSE_AE)
    encoder_config: EncoderConfig = EncoderConfig(name=ENCODERS_TYPES.DCONV, rotating_key=True)


config = Config()


def update_config_from_args(config: Config, args: argparse.Namespace):
    """Updates the config object with values from the parsed arguments."""
    args_dict = vars(args)

    for field_name, field in Config.model_fields.items():
        arg_name = field_name.replace('-', '_')
        if arg_name in args_dict and args_dict[arg_name] is not None:
            setattr(config, field_name, args_dict[arg_name])

    for field_name, field in Config.EmbeddingConfig.model_fields.items():
        arg_name = f"embedding_{field_name.replace('-', '_')}"
        if arg_name in args_dict and args_dict[arg_name] is not None:
            setattr(config.embedding_config, field_name, args_dict[arg_name])

    for field_name, field in Config.EncoderConfig.model_fields.items():
        arg_name = f"encoder_{field_name.replace('-', '_')}"
        if arg_name in args_dict and args_dict[arg_name] is not None:
            setattr(config.encoder_config, field_name, args_dict[arg_name])

    for field_name, field in Config.DatasetConfig.model_fields.items():
        arg_name = f"dataset_{field_name.replace('-', '_')}"
        if arg_name in args_dict and args_dict[arg_name] is not None:
            setattr(config.dataset_config, field_name, args_dict[arg_name])

    for field_name, field in Config.IIMConfig.model_fields.items():
        arg_name = f"iim_{field_name.replace('-', '_')}"
        if arg_name in args_dict and args_dict[arg_name] is not None:
            setattr(config.iim_config, field_name, args_dict[arg_name])
        if field_name == "neural_net_config":
            for nn_field_name, nn_field in Config.IIMConfig.NEURAL_NET_CONFIG.model_fields.items():
                arg_name = f"iim_neural_net_{nn_field_name.replace('-', '_')}"
                if arg_name in args_dict and args_dict[arg_name] is not None:
                    setattr(config.iim_config.neural_net_config, nn_field_name, args_dict[arg_name])

    for field_name, field in Config.CloudModelsConfig.model_fields.items():
        arg_name = f"cloud_{field_name.replace('-', '_')}"
        if arg_name in args_dict and args_dict[arg_name] is not None:
            setattr(config.cloud_config, field_name, args_dict[arg_name])

    for field_name, field in Config.ExperimentConfig.model_fields.items():
        arg_name = f"experiment_{field_name.replace('-', '_')}"
        if arg_name in args_dict and args_dict[arg_name] is not None:
            setattr(config.experiment_config, field_name, args_dict[arg_name])