from src.cloud.base import CloudModels
from src.cloud.vision import EfficientNetB2CloudModels, ResNetEmbeddingCloudModel
from src.cloud.tabular import EnsembleCloudModels, TabularCloudModels, NeuralNetCloudModels
from src.cloud.llm import CasualLLMCloudModel, MaskedLLMCloudModel

CLOUD_MODELS = {
    NeuralNetCloudModels.name: NeuralNetCloudModels,
    TabularCloudModels.name: TabularCloudModels,
    EnsembleCloudModels.name: EnsembleCloudModels,
    EfficientNetB2CloudModels.name: EfficientNetB2CloudModels,
    ResNetEmbeddingCloudModel.name: ResNetEmbeddingCloudModel,
    CasualLLMCloudModel.name: CasualLLMCloudModel,
    MaskedLLMCloudModel.name: MaskedLLMCloudModel,

}