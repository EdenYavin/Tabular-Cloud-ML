from typing import Union
from xgboost import XGBClassifier

from src.internal_model.base import NeuralNetworkInternalModel, TabularInternalModel
from src.internal_model.model import DenseInternalModel, BiggerDense, LSTMIIM
from src.utils.constansts import IIM_MODELS

class InternalInferenceModelFactory:

    @staticmethod
    def get_model(**kwargs) -> Union[NeuralNetworkInternalModel, TabularInternalModel]:

        iim = kwargs.get("type", IIM_MODELS.XGBOOST)

        if iim == IIM_MODELS.XGBOOST:
            return TabularInternalModel(**dict(model=XGBClassifier(), **kwargs))

        elif iim == IIM_MODELS.LSTM:
            return LSTMIIM(**kwargs)

        elif iim == IIM_MODELS.BIGGER_NEURAL_NET:
            return BiggerDense(**kwargs)
        else:
            return DenseInternalModel(**kwargs)