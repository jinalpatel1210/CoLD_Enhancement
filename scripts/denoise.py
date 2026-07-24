import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from .reliability import compute_agreement
from .classify import create_classifier
from .config import Config

VALID_CLUSTER_METHODS = ("kmeans", "gmm")

def align_clusters(cluster_ids, labels, n_classes):
    overlap = np.zeros((n_classes, n_classes), dtype=np.int64)
    for c, l in zip(cluster_ids, labels):
        if 0 <= c < n_classes and 0 <= l < n_classes:
            overlap[c, l] += 1
    row, col = linear_sum_assignment(overlap.max() - overlap)
    mapping = np.arange(n_classes)
    for r, c in zip(row, col):
        mapping[r] = c
    return mapping[cluster_ids]

def cluster_latent(feat, y_ref, n_classes, method, random_state):
    if method == "kmeans":
        km = KMeans(n_clusters=n_classes, random_state=random_state)
        km.fit(feat)
        labels = km.labels_
    else:
        gmm = GaussianMixture(
            n_components=n_classes, random_state=random_state, covariance_type="full",
        )
        labels = gmm.fit_predict(feat)
    return align_clusters(labels, y_ref, n_classes)

def cluster_labels_from_embedding(clf, z_fit, z_eval, y_train, n_classes, method, random_state):
    clf.fit(z_fit, y_train)
    proba = clf.predict_proba(z_eval)[:, :-1]
    return cluster_latent(proba, y_train, n_classes, method, random_state)

def purify_by_consistency(results, n_subsets):
    n = results[0].shape[0]
    keep_indices = list(range(n))
    for i in range(n):
        values = [results[j][i] for j in range(1, n_subsets + 2)]
        if len(set(values)) != 1:
            if results[0][i] != results[n_subsets + 1][i]:
                keep_indices.remove(i)
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_indices] = True
    return keep_mask

def recover_candidates(keep_mask, cluster_labels):
    """
    Find discarded samples having 100% agreement between
    the Global cluster and all Local view clusters.

    Parameters
    ----------
    keep_mask : np.ndarray
        Boolean mask returned by CoLD purification.

    cluster_labels : list
        Output cluster labels from CoLD.
        Format:
        [
            global_cluster,
            local_cluster_1,
            local_cluster_2,
            ...
            local_cluster_n,
            noisy_labels
        ]

    Returns
    -------
    candidate_indices : np.ndarray
        Indices of discarded samples whose agreement = 1.0

    agreement_labels : np.ndarray
        Corresponding agreement labels (global cluster labels)
    """

    global_cluster = cluster_labels[0]

        # all local views
    local_clusters = cluster_labels[1:-1]

    candidate_indices = []
    agreement_labels = []

    discarded_indices = np.where(~keep_mask)[0]

    for idx in discarded_indices:

        global_label = global_cluster[idx]

        agree = True

        for local in local_clusters:
            if local[idx] != global_label:
                agree = False
                break

        if agree:
            candidate_indices.append(idx)
            agreement_labels.append(global_label)

    return np.array(candidate_indices), np.array(agreement_labels)


class Denoiser:
    def __init__(self, config):
        self.cfg = config
        if config.cluster_method not in VALID_CLUSTER_METHODS:
            raise ValueError(f"cluster_method must be {VALID_CLUSTER_METHODS}")

    def purify(self, z_global, z_views, y_noisy, y_clean):
        K = self.cfg.n_classes
        rs = self.cfg.cluster_random_state
        method = self.cfg.cluster_method

        clf = create_classifier(self.cfg)
        agg = cluster_labels_from_embedding(
            clf, z_global, z_global, y_noisy, K, method, rs,
        )
        results = [agg]

        for z_view in z_views:
            view_clf = create_classifier(self.cfg)

            cluster = cluster_labels_from_embedding(
                view_clf,
                z_view,
                z_view,
                y_noisy,
                K,
                method,
                rs,
            )

            results.append(cluster)

        # Compute agreement BEFORE appending noisy labels
        agreement, agreement_labels = compute_agreement(results)

        # Append noisy labels for original CoLD purification
        results.append(y_noisy)
        keep_mask = purify_by_consistency(results, self.cfg.n_subsets)

        discard_mask = ~keep_mask

        discard_indices = np.where(discard_mask)[0]
        candidate_indices, _ = recover_candidates(
    keep_mask,
    results
)
        candidate_mask = np.zeros(len(y_noisy), dtype=bool)

        candidate_mask[recover_candidates(
            keep_mask,
            results
        )[0]] = True

        print("\n" + "=" * 65)
        print("STEP 1 : ORIGINAL CoLD PURIFICATION")
        print("=" * 65)
        print(f"Total Training Samples : {len(y_noisy)}")
        print(f"Samples Kept by CoLD   : {keep_mask.sum()}")
        print(f"Samples Discarded      : {len(discard_indices)}")
        print("=" * 65)
        
        self._save_results(
    y_clean,
    y_noisy,
    keep_mask,
    results,
    agreement,
    candidate_mask,
)
        return keep_mask, {
    "cluster_labels": results,
    "agreement": agreement,
    "agreement_labels": results[0], 
    "discard_indices": discard_indices,  
    "candidate_indices": candidate_indices,
}

    def _save_results(
    self,
    y_clean,
    y_noisy,
    keep_mask,
    results,
    agreement,
    candidate_mask,
):
        out_dir = os.path.join(self.cfg.results_dir, self.cfg.run_name)
        os.makedirs(out_dir, exist_ok=True)
        agreement_column = agreement.copy()

        df = pd.DataFrame({
    "y_clean": y_clean,
    "y_noisy": y_noisy,
    "keep": keep_mask,
    "agreement": agreement,
    "agg_cluster": results[0],
    "is_truly_noisy": y_clean != y_noisy,
    "agreement_candidate": candidate_mask,
})
        for j in range(self.cfg.n_subsets):
            df[f"view{j + 1}_cluster"] = results[j + 1]
        path = os.path.join(
            out_dir, f"denoising_{self.cfg.cluster_method}.csv",
        )
        df.to_csv(path, index=False)

