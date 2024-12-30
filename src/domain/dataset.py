import numpy as np
from numpy.typing import NDArray

from pydantic import BaseModel, ConfigDict


class PredictionsData(BaseModel):
    # Enable arbitrary types in the model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: NDArray[np.float_]
    predictions: NDArray[np.float_]
    predictions_and_embeddings: NDArray[np.float_]
    labels: NDArray[np.float_] | list[float]

class PredictionsDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_data: PredictionsData
    test_data: PredictionsData