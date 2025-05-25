import keras, tensorflow as tf

from src.cloud.base import CloudModel
from src.cloud.vision import (XceptionCloudModel,
      ResNetEmbeddingCloudModel,
      VGG16CloudModel, VGG16Cifer10CloudModel, VGG16Cifar100CloudModel,
      InceptionCloudModel, EfficientNetCloudModel, DenseNetCloudModel,
        MobileNetCloudModel
      )
from src.cloud.tabular import EnsembleCloudModel, TabularCloudModel, NeuralNetCloudModel
from src.cloud.llm import SequenceClassificationLLMCloudModel, BertLLMCloudModel

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
    MobileNetCloudModel.name: MobileNetCloudModel,
    SequenceClassificationLLMCloudModel.name: SequenceClassificationLLMCloudModel,
    BertLLMCloudModel.name: BertLLMCloudModel
}

class CloudModelManager:
    def __init__(self):
        self.current_model_name = None
        self.current_model = None

    def get_model(self, model_name):
        if self.current_model_name != model_name:
            # Clean up previous model
            if self.current_model is not None:
                del self.current_model
                import gc; gc.collect()
                keras.backend.clear_session()
            # Load new model
            self.current_model = CLOUD_MODELS[model_name]()
            self.current_model_name = model_name
        return self.current_model

    def predict(self, model_name, batch):
        model = self.get_model(model_name)
        # Ensure input is a tensor
        batch_tensor = tf.convert_to_tensor(batch)
        return model.predict(batch_tensor)