from keras.api.applications import ResNet50V2, VGG16, EfficientNetB2
from keras.api.applications.resnet50 import preprocess_input, decode_predictions
from keras.api.applications.vgg16 import preprocess_input, decode_predictions
from keras import Model
from keras.src.layers import GlobalAveragePooling2D, AveragePooling2D
import numpy as np
from PIL import Image
import cv2


from src.cloud.base import CloudModels
from src.utils.constansts import CANVAS_PATH


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

    def fit(self, X_train, y_train, **kwargs):
        # This model does not require fitting as it's using pretrained ResNet
        pass

    def evaluate(self, X, y, **kwargs):
        """Evaluate the model by using a downstream classifier after getting embeddings."""
        return -1, -1


class EfficientNetB2CloudModels(CloudModels):
    name = "efficientnet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = self.get_model()

    def fit(self, X_train, y_train, **kwargs):
        pass

    def get_model(self):
        # Load the pretrained ResNet50 model with ImageNet weights
        model = EfficientNetB2(weights='imagenet')
        return model

    def predict(self, X):
        # Ensure the input is properly preprocessed for ResNet50
        X = preprocess_input(X)
        predictions = self.models.predict(X, verbose=None)
        return predictions

    def evaluate(self, X, y):
        return -1, -1



class ImagePatchEfficientCloudModel(EfficientNetB2CloudModels):

    name = "patch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas = self.get_canvas()
        self.model = EfficientNetB2(weights='imagenet')

    def get_canvas(self):
        # Load the image from the input path
        image = Image.open(CANVAS_PATH)
        # Convert the image to a NumPy array
        canvas = np.array(image)

        canvas_resized = cv2.resize(canvas, (260, 260))

        return canvas_resized


    def predict(self, matrix_input):

        matrix_input_resized = cv2.resize(matrix_input[0], (260, 260))

        patch_canvas = np.clip(self.canvas + matrix_input_resized, 0, 255)

        # Preprocess the image for EfficientNet
        preprocessed_image = preprocess_input(patch_canvas)

        # Expand dimensions to match the input shape of the model
        input_tensor = np.expand_dims(preprocessed_image, axis=0)

        # Get predictions
        predictions = self.model.predict(input_tensor)

        return predictions