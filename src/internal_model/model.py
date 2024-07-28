from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from src.utils.constansts import CONFIG_IMM_NAME_TOKEN

class TabularInternalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        config = kwargs.get("config")
        self.name = config[CONFIG_IMM_NAME_TOKEN]
        self.model = XGBClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)
