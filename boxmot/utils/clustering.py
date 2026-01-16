# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Clustering utilities for person cropping."""

import numpy as np


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """
    Cluster embeddings using K-means algorithm.
    
    Args:
        embeddings: Array of shape (N, D) where N is number of samples and D is embedding dimension
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        Array of cluster labels for each embedding
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for clustering. "
            "Install it with: pip install scikit-learn"
        )
    
    if len(embeddings) < n_clusters:
        raise ValueError(
            f"Number of embeddings ({len(embeddings)}) is less than "
            f"number of clusters ({n_clusters})"
        )
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return labels
