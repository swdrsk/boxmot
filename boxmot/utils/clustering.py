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


def get_diverse_samples(
    embeddings: np.ndarray, 
    n_samples: int, 
    random_state: int = 42
) -> np.ndarray:
    """
    Select diverse samples from embeddings by clustering them and picking samples closest to centroids.
    
    Args:
        embeddings: Array of shape (N, D)
        n_samples: Number of samples to select
        random_state: Random seed
        
    Returns:
        Indices of the selected samples in the original array
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
    except ImportError:
        raise ImportError("scikit-learn is required for clustering.")
        
    n_total = len(embeddings)
    if n_total <= n_samples:
        return np.arange(n_total)
        
    # Cluster into n_samples clusters to find representative centroids
    kmeans = KMeans(n_clusters=n_samples, random_state=random_state, n_init=10)
    kmeans.fit(embeddings)
    
    # Find the real sample closest to each centroid
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    
    # If we have duplicates (rare but possible if n_samples close to n_total), fill up
    unique_indices = np.unique(closest_indices)
    if len(unique_indices) < n_samples:
        remaining = list(set(range(n_total)) - set(unique_indices))
        needed = n_samples - len(unique_indices)
        if len(remaining) > 0:
            extra_indices = np.random.choice(remaining, min(needed, len(remaining)), replace=False)
            unique_indices = np.concatenate([unique_indices, extra_indices])
            
    return np.sort(unique_indices)
