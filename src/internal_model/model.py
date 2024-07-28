from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
import numpy as np

class TabularInternalModel(BaseEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = XGBClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)