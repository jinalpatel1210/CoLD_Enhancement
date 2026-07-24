import numpy as np

def compute_agreement(results):
    """
    Parameters
    ----------
    results : list
        [global_prediction, view1, view2, ..., viewN]

    Returns
    -------
    agreement
    weighted_vote
    """

    n = results[0].shape[0]
    agreement = np.zeros(n)
    weighted_vote = np.zeros(n, dtype=int)

    n_views = len(results) - 1

    global_weight = 0.40
    local_weight = 0.60 / n_views

    weights = [global_weight] + [local_weight] * n_views

    for i in range(n):

        score = {}

        for j, pred in enumerate(results):

            label = pred[i]

            score[label] = score.get(label, 0.0) + weights[j]

        weighted_vote[i] = max(score, key=score.get)

        agreement[i] = score[weighted_vote[i]]

    return agreement, weighted_vote

