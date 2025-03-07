from src.cloud.base import CloudModel
from src.cloud.vision import (XceptionCloudModel,
      ResNetEmbeddingCloudModel,
      VGG16CloudModel, VGG16Cifer10CloudModel, VGG16Cifar100CloudModel,
      InceptionCloudModel, EfficientNetCloudModel, DenseNetCloudModel
      )
from src.cloud.tabular import EnsembleCloudModel, TabularCloudModel, NeuralNetCloudModel
from src.cloud.llm import SequenceClassificationLLMCloudModel

CLOUD_MODELS = {
    NeuralNetCloudModel.name: NeuralNetCloudModel,
    TabularCloudModel.name: TabularCloudModel,
    EnsembleCloudModel.name: EnsembleCloudModel,
    XceptionCloudModel.name: XceptionCloudModel,
    VGG16Cifer10CloudModel.name: VGG16Cifer10CloudModel,
    VGG16CloudModel.name: VGG16CloudModel,
    VGG16Cifar100CloudModel.name: VGG16Cifar100CloudModel,
    EfficientNetCloudModel.name: EfficientNetCloudModel,
    InceptionCloudModel.name: InceptionCloudModel,
    DenseNetCloudModel.name: DenseNetCloudModel,
    SequenceClassificationLLMCloudModel.name: SequenceClassificationLLMCloudModel,
}