from pydantic import BaseModel, Field
from src.utils.constansts import EMBEDDING_TYPES, ENCODERS_TYPES, IIM_MODELS, CLOUD_MODELS, DATASETS


class Config(BaseModel):

    class EmbeddingConfig(BaseModel):
        name: str = Field(description="Name of the embedding model to use")
        base_model: str = Field(description="Used in case the embedding model apply vary of models like different resnets", default="resnet101")

    class EncoderConfig(BaseModel):
        name: str = Field(description="Name of encryptor / encoder model to use", default=ENCODERS_TYPES.DCONV)

    class PipelineConfig(BaseModel):
        force_to_create_again: bool = Field(description="Flag to indicate if dataset should be created again", default=True)


    class DatasetConfig(BaseModel):
        names: list = Field(description="The datasets to run the experiments on")
        split_ratio : float = Field(description="How much of the original train set (90%) will be used to train the IIM")
        one_hot: bool = Field(description="A flag to indicate if the ground truth labels should be one-hot encoded", default=False)
        baseline_model: str = IIM_MODELS.NEURAL_NET

    class NEURAL_NET_CONFIG(BaseModel):
        epochs: int = 20
        batch_size: int = 64
        dropout: float = 0.3

    class IIMConfig(BaseModel):
        name: str = Field(description="IIM model to use", default=IIM_MODELS.NEURAL_NET)

    class CloudModelConfig(BaseModel):
        name: str = Field(description="Cloud model to use", default=CLOUD_MODELS.VGG16)

    class ExperimentConfig(BaseModel):
        use_labels: bool = Field(description="A flag to indicate if the noise labels should be used in training the IIM")
        use_preds: bool = Field(description="A flag to indicate if the predictions should be used in training the IIM")
        use_embedding: bool = Field(description="A flag to indicate if the embedding should be used in training the IIM")
        n_pred_vectors: int = Field(description="Number of prediction vectors to query from the cloud models")
        n_noise_samples: int = Field(description="Number samples to sample from the dataset and use as noise")
        k_folds : int = Field(description="Number of folds to use for cross-validation. If 1 - No k-fold", default=1)

    experiment_config: ExperimentConfig = ExperimentConfig(n_noise_samples=0,n_pred_vectors=1,
                                                           use_preds=True, use_embedding=False, use_labels=False)
    cloud_config: CloudModelConfig = CloudModelConfig(name=CLOUD_MODELS.VGG16)
    iim_config: IIMConfig = IIMConfig(name=IIM_MODELS.NEURAL_NET)
    neural_net_config: NEURAL_NET_CONFIG = NEURAL_NET_CONFIG()
    dataset_config: DatasetConfig = DatasetConfig(one_hot=True,
                                                  split_ratio=1,
                                                  names=[DATASETS.HELOC, DATASETS.ADULT, DATASETS.STUDENTS_DROPOUT,
                                                         DATASETS.BANK_MARKETING, DATASETS.GESTURE_PHASE
                                                         ]
                                                  )
    pipeline_config: PipelineConfig = PipelineConfig(force_to_create_again=True)
    embedding_config: EmbeddingConfig = EmbeddingConfig(name=EMBEDDING_TYPES.IMAGE)
    encoder_config: EncoderConfig = EncoderConfig(name=ENCODERS_TYPES.DCONV)


config = Config()