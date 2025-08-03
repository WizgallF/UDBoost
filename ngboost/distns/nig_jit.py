
import numpy as np
import math
from numba import njit, prange, float64, vectorize, guvectorize
from sklearn.tree import DecisionTreeRegressor

@njit(fastmath=True)
def digamma(x):
    result = 0.0
    while x < 6.0:
        result -= 1.0/x
        x += 1.0
    inv = 1.0/x
    inv2 = inv*inv
    result += math.log(x) - 0.5*inv - inv2*(1/12 - inv2*(1/120 - inv2*(1/252)))
    return result

@njit(fastmath=True)
def trigamma(x):
    result = 0.0
    while x < 6.0:
        result += 1.0/(x*x)
        x += 1.0
    inv = 1.0/x
    inv2 = inv*inv
    result += inv + 0.5*inv2 + inv2*inv2*(1/6 - inv2*(1/30 + inv2/42))
    return result


@njit(parallel=True, fastmath=True, cache=True)
def d_score_numba(Y, mu, lam, alpha, beta,
                  evid_strength, kl_strength):
    n = Y.shape[0]
    grads = np.empty((n, 4), np.float64)
    eps = 1e-8
    mu0, lam0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0

    for i in prange(n):
        yi, mui = Y[i], mu[i]
        lami, alphai, betai = lam[i], alpha[i], beta[i]

        # ── clip & common terms ───────────────────────
        if lami   < eps:     lami   = eps
        if alphai < 1.0+eps: alphai = 1.0+eps
        if betai  < eps:     betai  = eps

        nu    = 2.0 * alphai
        two_b = 2.0 * betai
        Om    = two_b * (1.0 + lami)
        resid = yi - mui
        r2    = resid * resid
        term  = lami*r2 + Om
        inv_l = 1.0 / lami
        inv_O = 1.0 / Om

        # ── NLL gradients ─────────────────────────────
        d_mu    = lami*(nu+1.0)*(-resid)/term
        d_lam   = -0.5*inv_l \
                  - alphai*two_b*inv_O \
                  + (alphai+0.5)*r2/term
        d_alpha = -math.log(Om) + math.log(term) \
                  + digamma(alphai) - digamma(alphai+0.5)
        d_beta  = -alphai/betai \
                  + (alphai+0.5)*(two_b*(1.0+lami)/two_b)/term

        # ── evidential reg (per‐sample) ───────────────
        sgn       = 1.0 if resid >= 0.0 else -1.0
        ev_mu     = -sgn*(2.0*alphai + lami)
        ev_lam    = abs(resid)
        ev_alpha  = 2.0*abs(resid)

        # ── KL reg (per‐sample) ───────────────────────
        gk_mu    = alpha0*lami*(mui - mu0)/betai
        gk_lam   = -0.5*inv_l + alpha0*((r2/(2.0*betai)) + 1.0/lam0)
        gk_alpha = (alphai-alpha0)*trigamma(alphai) + (beta0/betai - 1.0)
        gk_beta  = alpha0*(1.0/betai - beta0/(betai*betai)) \
                   - alphai/(betai*betai) \
                   - alpha0*lami*r2/(2.0*betai*betai)

        # ── combine & regs ────────────────────────────
        d_mu    += evid_strength*ev_mu   + kl_strength*gk_mu
        d_lam   += evid_strength*ev_lam  + kl_strength*gk_lam
        d_alpha += evid_strength*ev_alpha+ kl_strength*gk_alpha
        d_beta  +=                  kl_strength*gk_beta

        # ── chain‐rule back to raw parameters ────────
        grads[i, 0] = d_mu
        grads[i, 1] = d_lam   * lami
        grads[i, 2] = d_alpha*(alphai-1.0)
        grads[i, 3] = d_beta  * betai

    return grads


@njit(parallel=True, fastmath=True)
def full_score_numba(Y, mu, lam, alpha, evid_s, kl_s):
    n = Y.shape[0]
    out = np.empty(n, dtype=np.float64)

    mu0, lam0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0

    sum_evid = 0.0
    sum_kl = 0.0

    # First pass: compute per-sample NLL, accumulate evid/kl
    for i in prange(n):
        yi = Y[i]
        mui = mu[i]

        lami = lam[i] if lam[i] > 1e-8 else 1e-8
        alphai = alpha[i] if alpha[i] > 1.0 + 1e-8 else 1.0 + 1e-8

        # NLL (Student-t form, beta=1.0 assumed)
        nu = 2.0 * alphai
        Om = 2.0 * (1.0 + lami)
        resid = yi - mui
        term = lami * resid * resid + Om

        part1 = 0.5 * (math.log(math.pi) - math.log(lami))
        part2 = -alphai * math.log(Om)
        part3 = (alphai + 0.5) * math.log(term)
        part4 = math.lgamma(alphai) - math.lgamma(alphai + 0.5)

        nll_i = part1 + part2 + part3 + part4
        out[i] = nll_i

        # Evidential regularizer (summed for all samples)
        sum_evid += abs(resid) * (2.0 * alphai + lami)

        # KL regularizer (summed for all samples, beta=1.0 fixed)
        betai = 1.0
        t1 = 0.5 * math.log(lam0 / lami)
        t2 = alpha0 * math.log(betai / beta0)
        t3 = -math.lgamma(alphai) + math.lgamma(alpha0)
        t4 = (alphai - alpha0) * digamma(alphai)
        t5 = alpha0 * lami * (mui - mu0) ** 2 / (2.0 * betai)
        t6 = alpha0 * (lami / lam0 - 1.0)
        t7 = alphai * (beta0 / betai - 1.0)

        sum_kl += t1 + t2 + t3 + t4 + t5 + t6 + t7

    # Average regularizers
    mean_evid = sum_evid / n
    mean_kl = sum_kl / n

    # Second pass: add same regularizer to each element
    for i in prange(n):
        out[i] += evid_s * mean_evid + kl_s * mean_kl

    return out


@njit(parallel=True, fastmath=True)
def compute_diag_fim(grads):
    n, d = grads.shape
    fim_diag = np.empty((n, d), dtype=np.float64)
    for i in prange(n):
        for j in range(d):
            fim_diag[i, j] = grads[i, j] ** 2 + 1e-5  # add ridge to diagonal
    return fim_diag


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