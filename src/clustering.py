"""
Semantic clustering of chunks using embeddings.

Groups similar chunks together using k-means or other clustering algorithms.
"""

import numpy as np
from typing import List, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering
from .models import Chunk, Cluster


def compute_centroid(embeddings: List[List[float]]) -> List[float]:
    """
    Compute the centroid (mean) of a set of embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Centroid embedding vector
    """
    if not embeddings:
        return []

    arr = np.array(embeddings)
    centroid = np.mean(arr, axis=0)
    return centroid.tolist()


class ClusterEngine:
    """Clusters chunks based on embedding similarity."""

    def __init__(
        self,
        num_clusters: int = 10,
        method: str = "kmeans",
        random_state: int = 42
    ):
        """
        Initialize clustering engine.

        Args:
            num_clusters: Number of clusters to form
            method: Clustering algorithm ("kmeans", "hierarchical")
            random_state: Random seed for reproducibility
        """
        self.num_clusters = num_clusters
        self.method = method
        self.random_state = random_state

    def cluster_chunks(self, chunks: List[Chunk]) -> List[Cluster]:
        """
        Cluster chunks and assign cluster IDs.

        Modifies chunks in-place to set cluster_id.

        Args:
            chunks: List of chunks with embeddings

        Returns:
            List of Cluster objects
        """
        if not chunks:
            return []

        # Handle edge case: fewer chunks than clusters
        effective_clusters = min(self.num_clusters, len(chunks))

        # Extract embeddings as numpy array
        embeddings = np.array([chunk.embedding for chunk in chunks])

        # Perform clustering
        if self.method == "kmeans":
            clusterer = KMeans(
                n_clusters=effective_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.method == "hierarchical":
            clusterer = AgglomerativeClustering(
                n_clusters=effective_clusters,
                metric='euclidean',
                linkage='ward'
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Fit and predict cluster labels
        labels = clusterer.fit_predict(embeddings)

        # Assign cluster IDs to chunks
        for chunk, label in zip(chunks, labels):
            chunk.cluster_id = int(label)

        # Build Cluster objects
        clusters = []
        for cluster_id in range(effective_clusters):
            # Get chunks in this cluster
            cluster_chunks = [
                chunk for chunk in chunks if chunk.cluster_id == cluster_id
            ]

            # Get chunk IDs
            chunk_ids = [chunk.chunk_id for chunk in cluster_chunks]

            # Compute centroid
            cluster_embeddings = [chunk.embedding for chunk in cluster_chunks]
            centroid = compute_centroid(cluster_embeddings)

            cluster = Cluster(
                cluster_id=cluster_id,
                chunk_ids=chunk_ids,
                centroid=centroid,
                summary=None  # Set during summarization
            )

            clusters.append(cluster)

        return clusters


def cluster_documents(
    chunks: List[Chunk],
    num_clusters: int = 10,
    method: str = "kmeans",
    random_state: int = 42
) -> List[Cluster]:
    """
    Convenience function to cluster chunks.

    Args:
        chunks: List of chunks with embeddings
        num_clusters: Target number of clusters
        method: Clustering algorithm
        random_state: Random seed for reproducibility

    Returns:
        List of Cluster objects
    """
    engine = ClusterEngine(
        num_clusters=num_clusters,
        method=method,
        random_state=random_state
    )
    return engine.cluster_chunks(chunks)
