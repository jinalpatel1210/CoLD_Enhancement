import numpy as np

from .clustering import cluster_samples
from .sieve import sieve_filter

class RecoveryModule:
    """
    Recover medium reliability samples.
    """

    def __init__(
        self,
        reliability_threshold=0.70,
        neighbour_weight=0.40,
        reliability_weight=0.60,
        k_neighbors=5,
        consistency_threshold=0.80,
        random_state=57
    ):

        self.reliability_threshold = reliability_threshold

        self.neighbour_weight = neighbour_weight

        self.reliability_weight = reliability_weight

        self.k_neighbors = k_neighbors

        self.consistency_threshold = consistency_threshold

        self.random_state = random_state

    def recover(
        self,
        features,
        labels,
        reliability
    ):
        """
        Parameters
        ----------
        features
            latent features of medium samples

        labels
            predicted labels

        reliability
            reliability score

        Returns
        -------
        dict
        """

        if len(features) == 0:

            return {

                "recovered_mask": np.array([], dtype=bool),

                "recovered_labels": np.array([], dtype=int),

                "cluster_ids": np.array([], dtype=int),

                "consistency": np.array([]),

                "recovery_score": np.array([])

            }

        cluster_ids, model = cluster_samples(
            features,
            random_state=self.random_state
        )

        sieve = sieve_filter(
            features,
            cluster_ids,
            labels,
            k=self.k_neighbors,
            threshold=self.consistency_threshold
        )

        consistency = sieve["consistency"]

        recovery_score = (

            self.reliability_weight * reliability +

            self.neighbour_weight * consistency

        )

        recovered_mask = (

            recovery_score >= self.reliability_threshold

        )

        recovered_labels = labels.copy()

        return {

            "recovered_mask": recovered_mask,

            "recovered_labels": recovered_labels,

            "cluster_ids": cluster_ids,

            "consistency": consistency,

            "recovery_score": recovery_score

        }
