from pydantic import BaseModel, Field
from src.utils.constansts import EMBEDDING_TYPES, ENCODERS_TYPES, IIM_MODELS, CLOUD_MODELS, EXPERIMENTS, HARD_DATASETS, LARGE_DATASETS, DATASETS


class Config(BaseModel):

    class EmbeddingConfig(BaseModel):
        name: str = Field(description="Name of the embedding model to use")
        base_model: str = Field(description="Used in case the embedding model apply vary of models like different resnets", default="resnet101")

    class EncoderConfig(BaseModel):
        name: str = Field(description="Name of encryptor / encoder model to use", default=ENCODERS_TYPES.DCONV)

    class DatasetConfig(BaseModel):
        names: list = Field(description="The datasets to run the experiments on")
        split_ratio : float = Field(description="How much of the original train set (90%) will be used to train the IIM")
        one_hot: bool = Field(description="A flag to indicate if the ground truth labels should be one-hot encoded", default=False)
        batch_size: int = Field(description="Batch size to accumulate", default=200)

    class NEURAL_NET_CONFIG(BaseModel):
        epochs: int = 100
        batch_size: int = 64
        dropout: float = 0.3

    class IIMConfig(BaseModel):
        name: list[str] | str = Field(description="IIM model to use. Can be multiple models", default=IIM_MODELS.NEURAL_NET)

    class CloudModelsConfig(BaseModel):
        names: list[str] = Field(description="Cloud model to use", default=[CLOUD_MODELS.VGG16])

    class ExperimentConfig(BaseModel):
        use_labels: bool = Field(description="A flag to indicate if the noise labels should be used in training the IIM")
        use_preds: bool = Field(description="A flag to indicate if the predictions should be used in training the IIM")
        use_embedding: bool = Field(description="A flag to indicate if the embedding should be used in training the IIM")
        n_pred_vectors: int = Field(description="Number of prediction vectors to query from the cloud models")
        n_noise_samples: int = Field(description="Number samples to sample from the dataset and use as noise")
        k_folds : int = Field(description="Number of folds to use for cross-validation. If 1 - No k-fold", default=1)
        exp_type: str = Field(description="type of the experiment - embedding learning, or prediction learning")
        stacking: bool = Field(description="A flag to indicate if the stacking should be used in training the IIM")



    experiment_config: ExperimentConfig = ExperimentConfig(n_noise_samples=0,n_pred_vectors=1,k_folds=1,
                                                           use_preds=True, use_embedding=True, use_labels=False,
                                                           exp_type=EXPERIMENTS.PREDICTIONS_LEARNING,
                                                           stacking=False)
    cloud_config: CloudModelsConfig = CloudModelsConfig(names=[
        CLOUD_MODELS.SEQUENCE_CLASSIFICATION_LLM
    ])
    iim_config: IIMConfig = IIMConfig(name=[IIM_MODELS.NEURAL_NET])
    neural_net_config: NEURAL_NET_CONFIG = NEURAL_NET_CONFIG()
    dataset_config: DatasetConfig = DatasetConfig(
                                                  split_ratio=1,
                                                  names=[
                                                     DATASETS.ADULT
                                                  ]
                                                  )
    embedding_config: EmbeddingConfig = EmbeddingConfig(name=EMBEDDING_TYPES.SPARSE_AE)
    encoder_config: EncoderConfig = EncoderConfig(name=ENCODERS_TYPES.DCONV)


config = Config()