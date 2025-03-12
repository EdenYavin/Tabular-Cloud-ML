import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import DistilBertTokenizer, DistilBertModel

from src.cloud.base import CloudModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def post_process(embeddings: np.ndarray):
    """
    Pad or truncate the embedding to be in the shape of (1000, embedding_dimension),
    the same shape of vision cloud models.
    """
    target_shape = 1000
    current_shape = embeddings.shape[1]

    if current_shape < target_shape:
        # Pad with zeros if the current shape is less than the target shape
        processed_embeddings = np.pad(embeddings, ((0, 0), (0, target_shape - current_shape)), mode='constant')
    else:
        # Truncate if the current shape is greater than the target shape
        processed_embeddings = embeddings[:target_shape]

    return processed_embeddings

def preprocess(X: np.ndarray) -> list[str]:
    X = np.log(np.mean(X, axis=(1, 2, 3))[...,np.newaxis])
    X_str = [str(x).replace("[", "").replace("]", "").strip(",") for x in X]
    return X_str

class SequenceClassificationLLMCloudModel(CloudModel):
    name = "sequence_classification_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(device)  # Adjust num_labels
        self.labels = None
        self.output_shape = (1, 1000)
        self.input_shape = (2,2)

    def predict(self, X: np.ndarray, batch_size: int = 32):

        X_str = preprocess(X)
        all_embeddings = []

        # Process in smaller batches
        for i in range(0, len(X_str), batch_size):
            batch = X_str[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)

            # Forward pass with hidden states output enabled
            outputs = self.model(**inputs, output_hidden_states=True)

            # Extract last hidden states and [CLS] embeddings
            last_hidden_states = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_states[:, 0, :]

            # Append processed embeddings to the list
            all_embeddings.append(cls_embeddings.detach().cpu().numpy())

        # Concatenate all embeddings from batches
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        return post_process(all_embeddings)

    def fit(self, X_train, y_train, **kwargs):
        self.labels = kwargs.get("labels")

    def evaluate(self, X, y) -> tuple:
        return -1, -1


class BertLLMCloudModel(CloudModel):
    name = "next_token_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
        self.labels = None
        self.output_shape = (1, 1000)
        self.input_shape = (2,2)

    def predict(self, X, batch_size: int = 32):
        X_str = preprocess(X)
        all_embeddings = []
        # Process in smaller batches
        for i in range(0, len(X_str), batch_size):
            batch = X_str[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)

            # Forward pass through model
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract last hidden states and [CLS] embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # Append processed embeddings to the list
            all_embeddings.append(cls_embeddings.detach().cpu().numpy())

        # Concatenate all embeddings from batches
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        return post_process(all_embeddings)

    def fit(self, X_train, y_train, **kwargs):
        self.labels = kwargs.get("labels")

    def evaluate(self, X, y) -> tuple:
        return -1, -1
