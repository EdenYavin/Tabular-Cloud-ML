from keras.src.applications import resnet
from keras.src.applications.resnet import preprocess_input
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import numpy as np

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
        embeddings = self.model(preprocess_input(x))
        return embeddings.numpy()[0]


class TabularEmbedding(nn.Module):
    name = "tabular_embedding"

    def __init__(self, model_path):
        super(TabularEmbedding, self).__init__()
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