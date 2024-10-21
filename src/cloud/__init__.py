from src.cloud.base import CloudModels
from src.cloud.vision import EfficientNetB2CloudModels, ResNetEmbeddingCloudModel, VGG16CloudModel, ImagePatchEfficientCloudModel
from src.cloud.tabular import EnsembleCloudModels, TabularCloudModels, NeuralNetCloudModels
from src.cloud.llm import CasualLLMCloudModel, MaskedLLMCloudModel, SequenceClassificationLLMCloudModel

CLOUD_MODELS = {
    NeuralNetCloudModels.name: NeuralNetCloudModels,
    TabularCloudModels.name: TabularCloudModels,
    EnsembleCloudModels.name: EnsembleCloudModels,
    EfficientNetB2CloudModels.name: EfficientNetB2CloudModels,
    ResNetEmbeddingCloudModel.name: ResNetEmbeddingCloudModel,
    ImagePatchEfficientCloudModel.name: ImagePatchEfficientCloudModel,
    VGG16CloudModel.name: VGG16CloudModel,
    CasualLLMCloudModel.name: CasualLLMCloudModel,
    MaskedLLMCloudModel.name: MaskedLLMCloudModel,
    SequenceClassificationLLMCloudModel.name: SequenceClassificationLLMCloudModel,
}