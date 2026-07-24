import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class FeatureReorderer:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.order_ = None

    def fit(self, X):
        n_features = X.shape[1]
        if not self.enabled or n_features <= 2:
            self.order_ = np.arange(n_features)
            return self
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.abs(np.nan_to_num(corr, nan=0.0))
        np.fill_diagonal(corr, 0.0)
        self.order_ = np.asarray(self._mst_dfs_order(corr), dtype=int)
        return self

    def transform(self, X):
        if self.order_ is None:
            raise RuntimeError("FeatureReorderer must be fit before transform.")
        return X[:, self.order_]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @staticmethod
    def _mst_dfs_order(corr):
        n = corr.shape[0]
        neg = corr.max() - corr
        np.fill_diagonal(neg, 0.0)
        neg = neg + 1e-9
        np.fill_diagonal(neg, 0.0)
        mst = minimum_spanning_tree(csr_matrix(np.triu(neg))).toarray()
        adj = [[] for _ in range(n)]
        rows, cols = np.nonzero(mst)
        for i, j in zip(rows, cols):
            adj[i].append(j)
            adj[j].append(i)
        iu = np.triu_indices(n, k=1)
        if len(iu[0]) == 0:
            return list(range(n))
        best = np.argmax(corr[iu])
        start = int(iu[0][best])
        visited = [False] * n
        order = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            order.append(node)
            for nb in sorted(adj[node], key=lambda x: corr[node, x]):
                if not visited[nb]:
                    stack.append(nb)
        for i in range(n):
            if not visited[i]:
                order.append(i)
        return order

