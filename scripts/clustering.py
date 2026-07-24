import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

def find_optimal_clusters(
    features,
    min_clusters=2,
    max_clusters=10,
    random_state=57
):
    """
    Automatically determine the best number of clusters
    using Silhouette Score.

    Returns
    -------
    best_k
    """

    n_samples = len(features)

    if n_samples < min_clusters:
        return 1

    max_clusters = min(max_clusters, n_samples - 1)

    best_score = -1
    best_k = min_clusters

    for k in range(min_clusters, max_clusters + 1):

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        labels = model.fit_predict(features)

        if len(np.unique(labels)) == 1:
            continue

        score = silhouette_score(features, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k

def cluster_samples(
    features,
    n_clusters=None,
    random_state=57
):
    """
    Cluster latent features.

    Returns
    -------
    cluster_ids
    model
    """

    if n_clusters is None:

        n_clusters = find_optimal_clusters(
            features,
            random_state=random_state
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    cluster_ids = model.fit_predict(features)

    return cluster_ids, model

def compute_cluster_quality(
    features,
    cluster_ids
):
    """
    Compute clustering quality metrics.
    """

    if len(np.unique(cluster_ids)) < 2:

        return {
            "silhouette": 0,
            "davies_bouldin": np.inf
        }

    silhouette = silhouette_score(
        features,
        cluster_ids
    )

    db = davies_bouldin_score(
        features,
        cluster_ids
    )

    return {
        "silhouette": silhouette,
        "davies_bouldin": db
    }

def compute_cluster_centers(
    model
):
    """
    Return cluster centers.
    """

    return model.cluster_centers_

def assign_new_samples(
    model,
    features
):
    """
    Assign unseen samples to existing clusters.
    """

    return model.predict(features)
