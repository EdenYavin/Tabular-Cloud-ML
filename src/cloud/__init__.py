from src.cloud.base import CloudModel
from src.cloud.vision import XceptionCloudModel, ResNetEmbeddingCloudModel, VGG16CloudModel
from src.cloud.tabular import EnsembleCloudModel, TabularCloudModel, NeuralNetCloudModel
# from src.cloud.llm import CasualLLMCloudModel, MaskedLLMCloudModel, SequenceClassificationLLMCloudModel

CLOUD_MODELS = {
    NeuralNetCloudModel.name: NeuralNetCloudModel,
    TabularCloudModel.name: TabularCloudModel,
    EnsembleCloudModel.name: EnsembleCloudModel,
    XceptionCloudModel.name: XceptionCloudModel,
    ResNetEmbeddingCloudModel.name: ResNetEmbeddingCloudModel,
    VGG16CloudModel.name: VGG16CloudModel,
    # CasualLLMCloudModel.name: CasualLLMCloudModel,
    # MaskedLLMCloudModel.name: MaskedLLMCloudModel,
    # SequenceClassificationLLMCloudModel.name: SequenceClassificationLLMCloudModel,
}