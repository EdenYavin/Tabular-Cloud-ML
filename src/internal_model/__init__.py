from typing import Union
from xgboost import XGBClassifier

from src.internal_model.base import NeuralNetworkInternalModel, TabularInternalModel
from src.internal_model.model import DenseInternalModel, BiggerDense, LSTMIIM, TransformerIIM, GatedFiLMConditionedIIM, GatedCloudIIM, FiLMConditionedIIM, DeepSetsReconstructionIIM, FlexibleSINClassifier
from src.utils.constansts import IIM_MODELS

class InternalInferenceModelFactory:

    @staticmethod
    def get_model(**kwargs) -> Union[NeuralNetworkInternalModel, TabularInternalModel]:

        iim = kwargs.get("type", IIM_MODELS.XGBOOST)

        if iim == IIM_MODELS.XGBOOST:
            return TabularInternalModel(**dict(model=XGBClassifier(), **kwargs))

        elif iim == IIM_MODELS.LSTM:
            return LSTMIIM(**kwargs)

        elif iim == IIM_MODELS.TRANSFORMER:
            return TransformerIIM(**kwargs)

        elif iim == IIM_MODELS.BIGGER_NEURAL_NET:
            return BiggerDense(**kwargs)

        elif iim == IIM_MODELS.GATED_FILM_CONDITIONED:
            return GatedFiLMConditionedIIM(**kwargs)

        elif iim == IIM_MODELS.GATED:
            return GatedCloudIIM(**kwargs)

        elif iim == IIM_MODELS.FILM_CONDITIONED:
            return FiLMConditionedIIM(**kwargs)

        elif iim == IIM_MODELS.DEEPSET:
            return DeepSetsReconstructionIIM(**kwargs)

        elif iim == IIM_MODELS.FLEXIBLE_SIN:
            return FlexibleSINClassifier(**kwargs)

        else:
            return DenseInternalModel(**kwargs)