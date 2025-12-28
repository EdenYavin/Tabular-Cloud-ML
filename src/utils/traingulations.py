from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
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


def get_dense_and_distant_anchors(embeddings, labels, n_samples=2, density_percentile=60):
    """
    Selects triangulation anchors for each class that are:
    1. In high-density regions (representative).
    2. Maximally distant from each other (stable triangulation).

    Args:
        embeddings (np.ndarray): The matrix of embeddings.
        labels (np.ndarray): The array of class labels (can be 1D or one-hot encoded).
        n_samples (int): Number of anchors to select PER CLASS.
        density_percentile (int): Percentile of density to keep (e.g., 60 means top 60% densest).

    Returns:
        np.ndarray: An array containing the selected anchors for all classes combined.
    """

    # Handle One-Hot Encoding if necessary
    if labels.ndim > 1:
        y_labels = np.argmax(labels, axis=1)
    else:
        y_labels = labels

    unique_classes = np.unique(y_labels)
    all_anchors = []

    for cls in unique_classes:
        # Filter embeddings for the current class
        class_embeddings = embeddings[y_labels == cls]

        # --- Step 1: Density Estimation ---
        # Estimate density using KNN distance (lower distance = higher density)
        n_neighbors = min(len(class_embeddings), 10)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(class_embeddings)
        distances, _ = nbrs.kneighbors(class_embeddings)

        # Mean distance to neighbors (excluding self at index 0)
        avg_neighbor_dist = distances[:, 1:].mean(axis=1)

        # Determine cutoff for "high density"
        cutoff = np.percentile(avg_neighbor_dist, density_percentile)

        # Indices of candidates that are dense enough
        candidate_indices = np.where(avg_neighbor_dist <= cutoff)[0]
        candidate_embeddings = class_embeddings[candidate_indices]

        # Fallback: If filtering removes too many points, use all class data
        if len(candidate_embeddings) < n_samples:
            candidate_embeddings = class_embeddings
            candidate_indices = np.arange(len(class_embeddings))
            # Recalculate dists for fallback
            avg_neighbor_dist = avg_neighbor_dist

        # --- Step 2: Furthest Point Sampling (FPS) ---
        selected_indices_local = []

        # 2a. First Anchor: Pick the absolute densest point (min neighbor dist)
        # We need the index relative to 'candidate_embeddings'
        # But avg_neighbor_dist corresponds to original class_embeddings indices
        # So we look up the densest value among the candidates
        densest_candidate_idx = np.argmin(avg_neighbor_dist[candidate_indices])
        selected_indices_local.append(densest_candidate_idx)

        # Distance buffer: Min dist from every candidate to the CURRENT set of selected points
        min_dists = euclidean_distances(
            candidate_embeddings[densest_candidate_idx].reshape(1, -1),
            candidate_embeddings
        ).flatten()

        # 2b. Subsequent Anchors: Pick point maximizing distance to current set
        for _ in range(n_samples - 1):
            next_idx = np.argmax(min_dists)
            selected_indices_local.append(next_idx)

            # Update minimum distances
            new_dists = euclidean_distances(
                candidate_embeddings[next_idx].reshape(1, -1),
                candidate_embeddings
            ).flatten()
            min_dists = np.minimum(min_dists, new_dists)

        # Append found anchors for this class
        all_anchors.append(candidate_embeddings[selected_indices_local])

    # Stack all class anchors into one array
    return np.vstack(all_anchors)