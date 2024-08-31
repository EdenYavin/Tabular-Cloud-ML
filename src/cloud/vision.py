from keras.api.applications import ResNet50V2, VGG16, EfficientNetB2
from keras.api.applications.resnet50 import preprocess_input, decode_predictions
from keras.api.applications.vgg16 import preprocess_input, decode_predictions
from keras import Model
from keras.src.layers import GlobalAveragePooling2D, AveragePooling2D


from src.cloud.base import CloudModels

class ResNetEmbeddingCloudModel:
    name = "resnet_embedding"

    def __init__(self, **kwargs):
        self.output_shape = kwargs.get('output_shape', 2048)  # Default embedding size for ResNet50
        self.model = self.get_model()

    def get_model(self):
        base_model = ResNet50V2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Global average pooling to obtain embedding
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def predict(self, X):
        """Predict using the ResNet model to get embeddings."""
        return self.model.predict(X, verbose=None)

    def fit(self, X_train, y_train):
        # This model does not require fitting as it's using pretrained ResNet
        pass

    def evaluate(self, X, y):
        """Evaluate the model by using a downstream classifier after getting embeddings."""
        return -1, -1


class EfficientNetB2CloudModels(CloudModels):
    name = "efficientnet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = self.get_model()

    def fit(self, X_train, y_train):
        pass

    def get_model(self):
        # Load the pretrained ResNet50 model with ImageNet weights
        model = EfficientNetB2(weights='imagenet')
        return model

    def predict(self, X, **kwargs):
        # Ensure the input is properly preprocessed for ResNet50
        X = preprocess_input(X)
        predictions = self.models.predict(X, verbose=None)
        return predictions

    def evaluate(self, X, y):
        return -1, -1