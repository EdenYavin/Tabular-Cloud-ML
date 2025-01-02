import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tqdm import tqdm
from pydantic import BaseModel, ConfigDict

class Batch:

    def __init__(self, X:pd.DataFrame | NDArray[np.float_], y: list | NDArray[np.int8], size=1):
        self.data = X
        self.y = y
        self.size = size
        self.start = 0
        self.end = size
        self.progress = tqdm(total=len(X), unit="samples", position=0, leave=True, desc="Batches")

    def accumulate(self, item: NDArray[np.float_] | pd.DataFrame):
        self.data.append(item)

    def is_empty(self) -> bool:
        return len(self.data) == 0

    def is_full(self) -> bool:
        return len(self.data) == self.size

    def __iter__(self):
        return self

    def __next__(self):
        self.end = min(self.start + self.size, len(self.data))
        data = self.data[self.start:self.end]
        y = self.y[self.start:self.end]
        if self.start >= len(self.data):
            raise StopIteration

        self.start += self.size
        self.progress.update(len(data))
        return np.vstack(data), np.vstack(y)


class IIMFeatures(BaseModel):
    # Enable arbitrary types in the model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    features: NDArray[np.float_]
    labels: NDArray[np.float_] | list[float]

class EmbeddingBaselineFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    embeddings: NDArray[np.float_]
    labels: NDArray[np.float_] | list[float]

class PredictionBaselineFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    predictions: NDArray[np.float_]
    labels: NDArray[np.float_] | list[float]

class PredictionsDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    train_iim_features: IIMFeatures
    train_embeddings: EmbeddingBaselineFeatures
    train_predictions: PredictionBaselineFeatures
    test_iim_features: IIMFeatures
    test_embeddings: EmbeddingBaselineFeatures
    test_predictions: PredictionBaselineFeatures