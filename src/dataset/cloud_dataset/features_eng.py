from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.base import TransformerMixin


class FeatureReduction(TransformerMixin):

    def __init__(self, **kwargs):

        n_components = kwargs.get('n_components', 'auto')
        method = kwargs.get('method', "none")

        methods = {
            "guassian": GaussianRandomProjection(n_components, random_state=42),
            "agglomeration": FeatureAgglomeration(n_clusters=n_components),
        }

        self.method = methods.get(method, None)


    def fit(self, X, y=None):
        if not self.method:
            return self

        self.method.fit(X)
        return self

    def transform(self, X, y=None):
        if not self.method:
            return X

        return self.method.transform(X)