import pandas as pd
from keras.src.applications import resnet
from keras.src.applications.resnet import preprocess_input

from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from src.utils.helpers import create_image_from_number, expand_matrix_to_img_size


class ImageEmbedding(nn.Module):

    name = "image_embedding"

    def __init__(self, base_model='resnet50'):
        super(ImageEmbedding, self).__init__()

        # Load a pre-trained ResNet model
        if base_model == 'resnet50':
            self.model = resnet.ResNet50(weights="imagenet", include_top=False)
        elif base_model == 'resnet101':
            self.model = resnet.ResNet101(weights="imagenet", include_top=False)
        elif base_model == 'resnet152':
            self.model = resnet.ResNet152(weights="imagenet", include_top=False)
        else:
            raise ValueError("Unsupported ResNet model")

        self.input_shape = (224, 224)
        self.output_shape = (7, 7, 2048)



    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape(1,224,224,3)
        # Extract embeddings
        image = expand_matrix_to_img_size(x, self.input_shape)
        embeddings = self.model(preprocess_input(image))
        return embeddings.numpy()[0]


class StringEmbeddings(nn.Module):
    name = "string_embedding"

    def __init__(self, model_path):
        super(StringEmbeddings, self).__init__()
        # Load the pre-trained Word2Vec model
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.vector_size = self.model.vector_size

    def embed_word(self, word):
        # Get the embedding for a single word
        if word in self.model:
            return self.model[word]
        else:
            # Return a zero vector if the word is not in the vocabulary
            return np.zeros(self.vector_size)

    def forward(self, words):
        # Get embeddings for a list of words
        embeddings = [self.embed_word(word) for word in words]
        return torch.tensor(embeddings, dtype=torch.float32)


class NumericalTableEmbeddings(nn.Module):

    name = "numerical_table_embedding"

    def __init__(self, image_size=(224, 224), font_size=80):
        super(NumericalTableEmbeddings, self).__init__()
        # Load pre-trained ResNet model and remove the final classification layer
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last FC layer
        self.image_size = image_size
        self.font_size = font_size
        self.output_shape = (7, 7, 2048)

        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(image_size),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Normalization like ResNet expects
        ])

    # Function to pass an image through ResNet and get the embedding
    def get_embedding_from_image(self, img):
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.resnet(img_tensor).squeeze()  # Get 1D embedding
        return embedding

    # Function to get the row embedding by averaging all column embeddings
    def get_row_embedding(self, row):
        embeddings = []
        for value in row:
            value = int(value) # Round to make the image more clear
            img = create_image_from_number(value, image_size=self.image_size, font_size=self.font_size)
            embedding = self.get_embedding_from_image(img)
            embeddings.append(embedding)

        # Stack embeddings and compute the mean
        embeddings = torch.stack(embeddings)
        return torch.mean(embeddings, dim=0)

    # Function to process an entire dataframe and return row embeddings
    def forward(self, matrix):

        if type(matrix) is pd.DataFrame:
            matrix = matrix.to_numpy()

        row_embeddings = []
        for row in matrix:
            row_embedding = self.get_row_embedding(row)
            row_embeddings.append(row_embedding)

        # Convert to tensor
        return torch.stack(row_embeddings)


