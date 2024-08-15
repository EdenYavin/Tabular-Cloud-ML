from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration

def random_projection_feature_engineering(X, n_components='auto'):
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp = rp.fit_transform(X)
    return X_rp, rp


# For new data
def transform_new_data(X_new, rp_model):
    return rp_model.transform(X_new)




def feature_agglomeration_engineering(X, n_clusters=20):

    if X.shape[1] < n_clusters:
        n_clusters = X.shape[1]

    agg = FeatureAgglomeration(n_clusters=n_clusters)
    X_agg = agg.fit_transform(X)
    return X_agg, agg

# Usage
# For new data
def transform_new_data(X_new, agg_model):
    return agg_model.transform(X_new)