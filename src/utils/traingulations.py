from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import numpy as np


class TriangulationTransformer:
    """
    Handles the mathematical transformation of embeddings for triangulation.
    """

    @staticmethod
    def compute_differential(target_embedding: np.ndarray, anchor_embeddings: np.ndarray) -> np.ndarray:
        """
        Computes the differential vectors: [Enc(Sample) - Enc(Anchor_i)]
        Returns a flattened vector of differentials.
        """
        # target: (1, D), anchors: (N, D) -> diffs: (N, D)
        diffs = target_embedding - anchor_embeddings
        return diffs.flatten()

    @staticmethod
    def compute_concatenation(target_embedding: np.ndarray, anchor_embeddings: np.ndarray) -> np.ndarray:
        """
        Original Method: Simply stacks the sample embedding with the anchor embeddings.
        Returns flattened [Enc(Sample), Enc(Anchor_1), ..., Enc(Anchor_N)]
        """
        # We flatten both and stack them
        return np.hstack([target_embedding.flatten(), anchor_embeddings.flatten()])

    @staticmethod
    def compute_cosine_distances(target_embedding: np.ndarray, anchor_embeddings: np.ndarray) -> np.ndarray:
        """
        Alternative: Computes cosine similarity scores (scalars) instead of full vectors.
        Useful if the embedding dimension is too large.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Result shape (1, N) -> flatten to (N,)
        scores = cosine_similarity(target_embedding, anchor_embeddings)
        return scores.flatten()

def get_class_representative_samples(embeddings, labels):
    """
    Selects one representative sample from each of two classes.

    The function calculates the centroid for each class and finds the sample
    closest to that centroid.

    Args:
        embeddings (np.ndarray): The matrix of embeddings.
        labels (np.ndarray): The array of class labels corresponding to the embeddings.

    Returns:
        np.ndarray: An array containing the two representative embeddings.
    """
    representative_samples = []

    # Get the unique class labels
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError("This function is designed for exactly two classes.")

    for label in unique_labels:
        # Get the embeddings for the current class
        class_embeddings = embeddings[labels.argmax(axis=1) == label]

        # Calculate the centroid (mean) of the class embeddings
        centroid = np.mean(class_embeddings, axis=0)

        # Calculate the distance from each point in the class to the centroid
        distances = euclidean_distances(class_embeddings, centroid.reshape(1, -1))

        # Find the index of the point closest to the centroid
        closest_point_index = np.argmin(distances)

        # Get the representative sample
        representative_sample = class_embeddings[closest_point_index]
        representative_samples.append(representative_sample)

    return np.array(representative_samples)


def get_triangulation_samples_clustering(n_samples, embeddings):
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10) # Set n_init explicitly
    kmeans.fit(embeddings)

    # Find the embedding closest to each centroid
    triangulation_samples = []
    for i in range(n_samples):
        # Get the centroid
        centroid = kmeans.cluster_centers_[i]

        # Find the index of the closest embedding in the original data
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_embedding_index = np.argmin(distances)

        triangulation_samples.append(embeddings[closest_embedding_index])

    return np.array(triangulation_samples)
