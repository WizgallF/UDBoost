"""
    Extra functions useful for Distns
"""
import numpy as np
import math
from numba import njit, prange, float64, vectorize, guvectorize
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from .distn import RegressionDistn

# pylint: disable=too-few-public-methods


def SurvivalDistnClass(Dist: RegressionDistn):
    """
    Creates a new dist class from a given dist. The new class has its implemented scores

    Parameters:
        Dist (RegressionDistn): a Regression distribution with censored scores implemented.

    Output:
        SurvivalDistn class, this is only used for Survival regression
    """

    class SurvivalDistn(Dist):
        # Stores the original distribution for pickling purposes
        _basedist = Dist
        scores = (
            Dist.censored_scores
        )  # set to the censored versions of the scores implemented for dist

        def fit(Y):
            """
            Parameters:
                Y : a object with keys {time, event}, each containing an array
            """
            return Dist.fit(Y["Time"])

    return SurvivalDistn

@njit(parallel=True, fastmath=True)
def rbf_kernel_and_grad_numba(X, gamma):
    """
    Compute the RBF kernel matrix and its gradient wrt the first argument using Numba.

    Parameters:
        X (np.ndarray): Input array of shape (n, d).
        gamma (float): RBF kernel parameter.

    Returns:
        K (np.ndarray): Kernel matrix of shape (n, n).
        dK (np.ndarray): Gradient tensor of shape (n, n, d), where
                         dK[i, j, k] = ∂/∂X[i, k] [exp(-gamma * ||X[i] - X[j]||^2)].
    """
    n, d = X.shape
    K = np.empty((n, n), dtype=np.float64)
    dK = np.empty((n, n, d), dtype=np.float64)
    for i in prange(n):
        for j in range(n):
            # Compute squared distance
            sq_dist = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                sq_dist += diff * diff
            # Kernel value
            k_val = math.exp(-gamma * sq_dist)
            K[i, j] = k_val
            # Gradient wrt X[i]
            for k in range(d):
                diff = X[i, k] - X[j, k]
                dK[i, j, k] = -2.0 * gamma * diff * k_val
    return K, dK


def compute_leaf_boxes(tree: DecisionTreeRegressor, feature_dim: int):
    """
    For each leaf node in `tree`, compute the axis-aligned [min, max] box
    in R^feature_dim that the leaf covers.
    Returns:
        leaf_box: dict mapping leaf_id -> (lower_bounds[feature_dim], upper_bounds[feature_dim])
    """
    t = tree.tree_
    # node -> (lo_bounds, hi_bounds)
    boxes = {0: (np.full(feature_dim, -np.inf),
                 np.full(feature_dim,  np.inf))}
    leaf_box = {}

    def recurse(node_id):
        lo, hi = boxes[node_id]
        # if this node is a leaf
        if t.children_left[node_id] == t.children_right[node_id]:
            leaf_box[node_id] = (lo.copy(), hi.copy())
            return
        feat = t.feature[node_id]
        thr  = t.threshold[node_id]
        # left child: x[feat] <= thr
        lo_l, hi_l = lo.copy(), hi.copy()
        hi_l[feat] = min(hi[feat], thr)
        boxes[t.children_left[node_id]] = (lo_l, hi_l)
        recurse(t.children_left[node_id])
        # right child: x[feat] > thr
        lo_r, hi_r = lo.copy(), hi.copy()
        lo_r[feat] = max(lo[feat], thr)
        boxes[t.children_right[node_id]] = (lo_r, hi_r)
        recurse(t.children_right[node_id])

    recurse(0)
    return leaf_box


def leaf_volume_density_vec(tree, X):
    """
    Vectorized leaf‐volume density: p_hat[i] ∝ n_leaf(i) / vol(leaf(i)),
    with mean(p_hat)=1, but using only NumPy indexing (no Python loops).
    """
    n, d = X.shape

    # 1) which leaf each row lands in
    leaf_ids = tree.apply(X)            # shape (n,)

    # 2) count per leaf, and get unique ids
    unique, counts = np.unique(leaf_ids, return_counts=True)
    # build a dense map from leaf_id -> count
    max_id = unique.max()
    count_map = np.zeros(max_id+1, int)
    count_map[unique] = counts         # now count_map[leaf_ids] gives counts per row

    # 3) build bounding‐boxes once (you still need the recursion for that)
    #    but we can then compute volumes for only the unique leaves
    boxes = compute_leaf_boxes(tree, d)   # your existing function
    # volumes in a dense array
    vol_map = np.empty(max_id+1, float)
    for lid in unique:
        lo, hi = boxes[lid]
        span = hi - lo
        span[np.isinf(span)] = 1e6
        vol_map[lid] = np.prod(span)

    # 4) density per sample, vectorized
    p_hat = count_map[leaf_ids] / (vol_map[leaf_ids] + 1e-12)

    # 5) normalize to mean=1
    p_hat *= (n / p_hat.sum())

    return p_hat

def epistemic_scaling(Dist, knn=10):
    """
    Scales the epistemic uncertainty of a distribution to the range [0, 1].

    Parameters:
        Dist (RegressionDistn): A regression distribution.
        Uncertainties (dict): A dictionary containing uncertainties.

    Returns:
        Scaled uncertainties.
    """
    uncertainties = Dist.pred_uncertainty()
    epistemic = uncertainties["epistemic"]
    aleatoric = uncertainties["aleatoric"]
    X_test = Dist.X_test
    X_train = Dist.X_train
    tree = Dist.leaf_tree
    if tree is not None and X_train is not None and X_test is not None:
        eps = 1e-12

        # 1) leaf‐volume weight
        p_hat = leaf_volume_density_vec(tree, X_test)
        w_vol = np.log(p_hat + eps) * 0.1

        # Find k nearest neighbors of each X_test in X_train
        k = min(knn, X_train.shape[0])  # choose k, e.g., 10 or less if not enough samples
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train)
        # For each test point, get indices of k nearest neighbors in X_train
        _, indices = nbrs.kneighbors(X_test)
        # Compute mean of k nearest neighbors for each test point
        x_train_mean = np.mean(X_train[indices], axis=1)  # shape (n_test, d)
        diffs = X_test - x_train_mean               # (n_test, d)
        # if d>1, you could sum over dims, but here we do per‐row dot diag:
        # m2[i] = sum_j (diffs[i,j]**2 / aleatoric[i])
        # which for 1‐D is just (diffs[:,0]**2 / aleatoric)
        m2 = np.einsum('ij,i->i', diffs**2, 1.0/(aleatoric + eps))

        w_maha = np.sqrt(np.exp(-10 * m2) )            # (n_test,)

        # 3) combine
        w_comb = w_vol * w_maha
        
        scaled_uncertainties = {
            "epistemic": w_comb * epistemic,
            "aleatoric": aleatoric,
            "predictive": w_comb * epistemic + aleatoric,
        }
    return scaled_uncertainties