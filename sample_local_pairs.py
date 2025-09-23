import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors


def sample_local_pairs(X, k):

    n = X.shape[0]

    if n <= 1:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)
    
    k_eff = min(k, max(1, n - 1))

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto")

    nbrs.fit(X)

    dists_all, inds_all = nbrs.kneighbors(X)

    neighbor_inds = inds_all[:, 1:]

    neighbor_dists = dists_all[:, 1:]

    pairs = np.vstack([np.repeat(np.arange(n), k_eff), neighbor_inds.ravel()]).T

    dist_vals = neighbor_dists.ravel()

    maxd = dist_vals.max() if dist_vals.size > 0 else 1.0

    normalized = dist_vals / (maxd if maxd != 0 else 1.0)
    
    return pairs.astype(int), normalized.astype(float)