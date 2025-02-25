from typing import Union

from xgboost import XGBClassifier

from src.internal_model.model import NeuralNetworkInternalModel, TabularInternalModel, DenseInternalModel
from src.utils.constansts import IIM_MODELS


class InternalInferenceModelFactory:

    @staticmethod
    def get_model(**kwargs) -> Union[NeuralNetworkInternalModel, TabularInternalModel]:

        iim = kwargs.get("type", IIM_MODELS.XGBOOST)

        if iim == IIM_MODELS.XGBOOST:
            return TabularInternalModel(**dict(model=XGBClassifier(), **kwargs))

        else:
            return DenseInternalModel(**kwargs)
