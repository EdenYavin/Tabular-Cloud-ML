from pydantic import BaseModel, Field
from src.utils.constansts import EMBEDDING_TYPES, ENCODERS_TYPES, IIM_MODELS, CLOUD_MODELS, DATASETS

class Config(BaseModel):

    class EmbeddingConfig(BaseModel):
        name: str = Field(description="Name of the embedding model to use", default=EMBEDDING_TYPES.DIGIT_TO_IMAGE_TO_EMBEDDING)

    class EncoderConfig(BaseModel):
        name: str = Field(description="Name of encryptor / encoder model to use", default=ENCODERS_TYPES.RESNET)

    class DatasetConfig(BaseModel):
        names: list = [DATASETS.HELOC]
        split_ratio : float = 0.2
        one_hot: bool = True
        force_to_create_again: bool = Field(description="Flag to indicate if dataset should be created again", default=True)
        baseline_model: str = "neural_network" # xgboost / neural_network
        use_pd_df: bool = Field(description="Flag to indicate if the dataset should be converted from pandas dataframe"
                                               "to numpy array or not", default=True)

    class NEURAL_NET_CONFIG(BaseModel):
        epochs: int = 10
        batch_size: int = 64
        dropout: float = 0.4

    class IIMConfig(BaseModel):
        name: str = Field(description="IIM model to use", default=IIM_MODELS.NEURAL_NET)

    class CloudModelConfig(BaseModel):
        name: str = Field(description="Cloud model to use", default=CLOUD_MODELS.RESNET)

    class ExperimentConfig(BaseModel):
        name: str = Field(description="Name of the experiment with indicators of what to use", default="w_emb_w_label_w_pred")
        use_labels: bool = True
        use_preds: bool = True
        n_pred_vectors: int = Field(description="Number of prediction vectors to query from the cloud models", default=1)
        n_noise_samples: int = Field(description="Number samples to sample from the dataset and use as noise", default=2)
        k_folds : int = Field(description="Number of folds to use for cross-validation. If 1 - No k-fold", default=1)

    experiment_config: ExperimentConfig = ExperimentConfig()
    cloud_config: CloudModelConfig = CloudModelConfig()
    iim_config: IIMConfig = IIMConfig()
    neural_net_config: NEURAL_NET_CONFIG = NEURAL_NET_CONFIG()
    dataset_config: DatasetConfig = DatasetConfig()
    embedding_config: EmbeddingConfig = EmbeddingConfig()
    encoder_config: EncoderConfig = EncoderConfig()


config = Config()