from pydantic import BaseModel, Field
from src.utils.constansts import EMBEDDING_TYPES, ENCODERS_TYPES, IIM_MODELS, CLOUD_MODELS, DATASETS

class Config(BaseModel):

    class EmbeddingConfig(BaseModel):
        name: str = Field(description="Name of the embedding model to use", default=EMBEDDING_TYPES.IDENTITY)
        base_model: str = Field(description="Used in case the embedding model apply vary of models like different resnets", default="resnet101")

    class EncoderConfig(BaseModel):
        name: str = Field(description="Name of encryptor / encoder model to use", default=ENCODERS_TYPES.DCONV)

    class DatasetConfig(BaseModel):
        names: list = [DATASETS.HELOC]
        split_ratio : float = Field(description="How much of the original train set (90%) will be used to train the IIM")
        one_hot: bool = Field(description="A flag to indicate if the groudn truth labels should be one-hot encoded", default=False)
        force_to_create_again: bool = Field(description="Flag to indicate if dataset should be created again", default=True)
        baseline_model: str = IIM_MODELS.NEURAL_NET
        use_pd_df: bool = Field(description="Flag to indicate if the dataset should be converted from pandas dataframe"
                                               "to numpy array or not", default=True)

    class NEURAL_NET_CONFIG(BaseModel):
        epochs: int = 10
        batch_size: int = 64
        dropout: float = 0.4

    class IIMConfig(BaseModel):
        name: str = Field(description="IIM model to use", default=IIM_MODELS.NEURAL_NET)

    class CloudModelConfig(BaseModel):
        name: str = Field(description="Cloud model to use", default=CLOUD_MODELS.VGG16)

    class ExperimentConfig(BaseModel):
        name: str = Field(description="Name of the experiment with indicators of what to use")
        use_labels: bool = Field(description="A flag to indicate if the noise labels should be used in training the IIM", default=True)
        use_preds: bool = Field(description="A flag to indicate if the predictions should be used in training the IIM", default=True)
        n_pred_vectors: int = Field(description="Number of prediction vectors to query from the cloud models", default=1)
        n_noise_samples: int = Field(description="Number samples to sample from the dataset and use as noise", default=0)
        k_folds : int = Field(description="Number of folds to use for cross-validation. If 1 - No k-fold", default=1)

    experiment_config: ExperimentConfig = ExperimentConfig(name="w_emb_w_label_w_pred", n_noise_samples=0)
    cloud_config: CloudModelConfig = CloudModelConfig()
    iim_config: IIMConfig = IIMConfig(name=IIM_MODELS.NEURAL_NET)
    neural_net_config: NEURAL_NET_CONFIG = NEURAL_NET_CONFIG()
    dataset_config: DatasetConfig = DatasetConfig(one_hot=True, name=DATASETS.HELOC, split_ratio=0.2)
    embedding_config: EmbeddingConfig = EmbeddingConfig(name=EMBEDDING_TYPES.DIGIT_TO_IMAGE_TO_EMBEDDING)
    encoder_config: EncoderConfig = EncoderConfig(name=ENCODERS_TYPES.DCONV)


config = Config()