# ============================
# End-to-end DAG + NGBoost demo
# ============================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_diabetes
import pandas as pd

# --- NGBoost ---
from ngboost import NGBRegressor
from ngboost.distns import NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD

# --- DAGMA (causal discovery on real data) ---
from dagma.linear import DagmaLinear

# --- Synthetic DAG generator (Castle) ---
from castle.datasets import DAG
from castle.datasets.simulator import IIDSimulation


# -----------------------
# Helpers: training + eval
# -----------------------
def fit_ngboost(
    X_train, y_train, X_test, y_test,
    use_svgd=True,
    n_estimators=400,
    learning_rate=0.1,
    max_depth=4,
    verbose=True,
):
    """
    Fit NGBoost with Normal-Inverse-Gamma target, z-scoring y for stability.
    Returns: y_pred (original scale), Pearson r, trained model.
    """
    # z-score target (very important for NIG stability)
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-12
    y_train_z = (y_train - y_mean) / y_std

    score = NIGLogScoreSVGD if use_svgd else NIGLogScore
    ngb = NGBRegressor(
        Dist=NormalInverseGamma,
        Score=NIGLogScore,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        alpha=0.001,
        verbose=verbose,
        natural_gradient=True,
        epistemic_scaling=True, 
        metadistribution_method="evidential_regression"   # Enable epistemic scaling
        )
    ngb.fit(X_train, y_train_z)

    # predict in z-space and back-transform
    y_pred_z = ngb.predict(X_test)
    y_pred = y_pred_z * y_std + y_mean

    r, _ = pearsonr(y_pred, y_test)
    return y_pred, float(r), ngb


def plot_pred_vs_true(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_learned_dag(
    adj_matrix,
    feature_names=None,
    threshold=None,          # keep edges with |w| >= threshold
    top_k=None,              # OR keep top-k by |w|
    title="Learned graph",
    figsize=(12, 6),
    seed=0,
    edge_labels=False,
    add_heatmap=True,        # show heatmap next to the graph
    heatmap_pruned=False,    # heatmap over pruned edges if True; else full W
    heatmap_cmap="coolwarm",
):
    """
    Plot a (possibly weighted) directed graph from an adjacency matrix,
    and (optionally) a heatmap of the adjacency.

    Parameters
    ----------
    adj_matrix : (d, d) array-like
        Weighted adjacency; W[i, j] is weight of i -> j.
    feature_names : list[str] or None
        Node names; defaults to ["x0", ..., "x{d-1}"].
    threshold : float or None
        If set, keep edges with |weight| >= threshold.
    top_k : int or None
        If set (and threshold is None), keep top-k edges by |weight|.
    title : str
        Figure title (applies to whole figure).
    edge_labels : bool
        If True, draw numeric weights on edges.
    add_heatmap : bool
        If True, show a heatmap of the adjacency next to the graph.
    heatmap_pruned : bool
        If True, the heatmap shows only the retained edges (others = 0).
        If False, the heatmap shows the full adjacency.
    heatmap_cmap : str
        Matplotlib colormap for the heatmap.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    W = np.asarray(adj_matrix, dtype=float)
    d = W.shape[0]
    assert W.shape == (d, d), "adj_matrix must be square"

    names = feature_names if feature_names is not None else [f"x{i}" for i in range(d)]

    # --- collect all nonzero off-diagonal edges ---
    edges_all = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = W[i, j]
            if np.isfinite(w) and abs(w) > 0:
                edges_all.append((i, j, float(w)))

    if not edges_all:
        print("No edges to plot.")
        return

    # --- prune by threshold or top-k ---
    edges = edges_all
    if threshold is not None:
        edges = [e for e in edges if abs(e[2]) >= threshold]
    elif top_k is not None and top_k < len(edges):
        edges = sorted(edges, key=lambda e: abs(e[2]), reverse=True)[:top_k]

    if not edges:
        print("No edges left after pruning.")
        return

    # --- build pruned adjacency (for optional heatmap) ---
    W_pruned = np.zeros_like(W)
    for i, j, w in edges:
        W_pruned[i, j] = w

    # --- create figure layout ---
    if add_heatmap:
        fig, (ax_hm, ax_g) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.1, 1.2]))
    else:
        fig, ax_g = plt.subplots(1, 1, figsize=(figsize[0] * 0.6, figsize[1]))
        ax_hm = None

    # --- draw graph ---
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    pos = nx.spring_layout(G, seed=seed)

    # edge widths & colors by |weight|
    ws = np.array([abs(G[u][v]["weight"]) for u, v in G.edges()])
    if ws.size > 0 and ws.max() > 0:
        widths = 1.0 + 4.0 * (ws - ws.min()) / (ws.max() - ws.min() + 1e-12)
        colors = (ws - ws.min()) / (ws.max() - ws.min() + 1e-12)
    else:
        widths = np.ones_like(ws)
        colors = np.zeros_like(ws)

    nx.draw_networkx_nodes(G, pos, ax=ax_g, node_color="#e8f0fe",
                           edgecolors="#4e5d78", node_size=900)
    nx.draw_networkx_labels(G, pos, ax=ax_g,
                            labels={i: n for i, n in enumerate(names)}, font_size=10)

    nx.draw_networkx_edges(
        G, pos, ax=ax_g,
        arrowstyle="-|>", arrowsize=16,
        width=widths.tolist(),
        edge_color=colors, edge_cmap=plt.cm.magma,
        connectionstyle="arc3,rad=0.05",
    )

    if edge_labels:
        lbls = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, ax=ax_g, edge_labels=lbls,
                                     font_size=8, label_pos=0.5)

    ax_g.set_title("Directed graph")
    ax_g.axis("off")

    # --- draw heatmap (optional) ---
    if add_heatmap:
        H = W_pruned if heatmap_pruned else W
        vmax = np.max(np.abs(H)) + 1e-12
        im = ax_hm.imshow(H, cmap=heatmap_cmap, vmin=-vmax, vmax=vmax)
        ax_hm.set_xticks(range(d))
        ax_hm.set_yticks(range(d))
        ax_hm.set_xticklabels(names, rotation=45, ha="right")
        ax_hm.set_yticklabels(names)
        ax_hm.set_title("Adjacency heatmap" + (" (pruned)" if heatmap_pruned else " (full)"))
        cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("edge weight", rotation=90)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# =========================================
# Part A: Synthetic data from a random DAG
# =========================================
def run_synthetic_demo(
    d=10, exp_edges=2, n=500, noise_scale=0.5,
    test_size=0.1, seed=42,
):
    print("\n=== Synthetic DAG Demo (Castle) ===")
    # 1) sample a random weighted DAG
    W = DAG.erdos_renyi(d, exp_edges)

    # 2) simulate IID linear-Gaussian data from this DAG
    sim = IIDSimulation(
    W=W,                # your weighted adjacency matrix
    n=500,             # number of samples
    method='linear',    # 'linear' or 'nonlinear'
    sem_type='gauss',   # noise type: 'gauss', 'exp', 'gumbel', etc.
    noise_scale=0.5     # noise standard deviation
    )
    X_all = sim.X  # shape (n, d)

    # 3) pick a node with at least one parent as target
    child_nodes = np.where(W.sum(axis=0) != 0)[0]
    if len(child_nodes) == 0:
        raise RuntimeError("Random graph had no edges—try different exp_edges.")
    target_idx = child_nodes[-1]
    parents = np.where(W[:, target_idx] != 0)[0]
    print(f"Target node: {target_idx}, parents: {parents}")

    y = X_all[:, target_idx]
    X = np.delete(X_all, target_idx, axis=1)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # quick baseline
    bl = LinearRegression().fit(X_train, y_train)
    r_bl, _ = pearsonr(bl.predict(X_test), y_test)
    print(f"Baseline LinearRegression Pearson r = {r_bl:.3f}")

    # NGBoost (NIG + SVGD)
    y_pred, r, _ = fit_ngboost(X_train, y_train, X_test, y_test, use_svgd=True)
    print(f"NGBoost(NIG, SVGD) Pearson r = {r:.3f}")

    # plot
    plot_pred_vs_true(y_test, y_pred, f"Synthetic DAG — NGBoost r={r:.3f}")


# =========================================================
# Part B: California Housing — DAGMA + NGBoost on parents
# =========================================================
def _fit_dagma_linear(Z, reg_w=1e-2, max_iter=2000, lr=1e-2, T=None, w_threshold=None, seed=0):
    """
    Tries both common DAGMA APIs:
      • model.fit(Z, reg_w=..., max_iter=..., lr=..., T=...)
      • model.fit(Z, lambda1=..., max_iter=..., lr=..., checkpoint=...)
    Returns W_hat (numpy array).
    """
    model = DagmaLinear(loss_type="l2", verbose=False)
    try:
        # Newer API
        kwargs = dict(reg_w=reg_w, max_iter=max_iter, lr=lr)
        if T is not None:
            kwargs["T"] = T
        if w_threshold is not None:
            kwargs["w_threshold"] = w_threshold
        W_hat = model.fit(Z, **kwargs)
        return W_hat
    except TypeError:
        # Older API (lambda1 / checkpoint)
        kwargs = dict(lambda1=reg_w, max_iter=max_iter, lr=lr, checkpoint=200)
        if w_threshold is not None:
            kwargs["w_threshold"] = w_threshold
        W_hat = model.fit(Z, **kwargs)
        return W_hat


# ---------- The dataset-agnostic demo ----------

def dagma_demo(
    df: pd.DataFrame,
    target_name: str,
    include_target: bool = True,
    n_subsample: int | None = 8000,
    reg_w: float = 2e-2,
    w_threshold: float = 0.05,
    test_size: float = 0.2,
    seed: int = 42,
    use_svgd: bool = True,
):
    """
    Learn a DAG with DAGMA on (standardized) columns of `df`,
    then train NGBoost on (a) all features and (b) parents-only.

    Returns:
        dict with W_hat, parents, r_all, r_parents (if any), models, etc.
    """
    assert target_name in df.columns, f"{target_name=} not in df."

    feature_names = [c for c in df.columns if c != target_name]

    if include_target:
        var_names = feature_names + [target_name]
        Z_raw = df[var_names].to_numpy()
    else:
        var_names = feature_names
        Z_raw = df[feature_names].to_numpy()

    # z-score for DAG learning
    Z = StandardScaler().fit_transform(Z_raw)

    # optional subsample for speed
    rng = np.random.default_rng(seed)
    if n_subsample is not None and Z.shape[0] > n_subsample:
        idx = rng.choice(Z.shape[0], size=n_subsample, replace=False)
        Z = Z[idx]

    # learn DAG
    W_hat = _fit_dagma_linear(
        Z,
        reg_w=reg_w,
        max_iter=2000,
        lr=0.01,
        T=8,
        w_threshold=None,  # we threshold only for analysis/plot below
        seed=seed,
    )

    # plot adjacency
    plot_learned_dag(
        W_hat,
        feature_names=var_names,
        threshold=w_threshold,
        title=f"DAGMA learned weighted adjacency ({target_name})",
    )

    # prune tiny edges for parent extraction
    W_thr = W_hat.copy()
    if w_threshold is not None and w_threshold > 0:
        W_thr[np.abs(W_thr) < w_threshold] = 0.0

    # parents of target (only if target included in DAG)
    parents = []
    if include_target:
        t_idx = len(var_names) - 1
        p_idx = np.where(W_thr[:, t_idx] != 0.0)[0]
        parents = [var_names[j] for j in p_idx if var_names[j] != target_name]
        print("Parents of target above threshold:", parents)
    else:
        print("Target not included in DAG learning; skipping parents-of-target extraction.")

    # regression matrices on original scale
    X_full = df[feature_names].to_numpy()
    y = df[target_name].to_numpy()

    X_train_A, X_test_A, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=seed
    )

    # baseline LR (all features)
    lr = LinearRegression().fit(X_train_A, y_train)
    r_lr, _ = pearsonr(lr.predict(X_test_A), y_test)
    print(f"[All features] LinearRegression Pearson r = {r_lr:.3f}")

    # NGBoost on all features
    y_pred_all, r_all, ngb_all = fit_ngboost(
        X_train_A, y_train, X_test_A, y_test, use_svgd=use_svgd
    )
    print(f"[All features] NGBoost(NIG, {'SVGD' if use_svgd else 'LogScore'}) r = {r_all:.3f}")
    plot_pred_vs_true(y_test, y_pred_all, f"NGBoost — all features (r={r_all:.3f})")

    # NGBoost on parents only (if any)
    r_par, y_pred_par, ngb_par, X_train_B, X_test_B = None, None, None, None, None
    if include_target and len(parents) > 0:
        parent_feat_idx = [feature_names.index(p) for p in parents if p in feature_names]
        X_parent = X_full[:, parent_feat_idx]
        X_train_B, X_test_B, _, _ = train_test_split(
            X_parent, y, test_size=test_size, random_state=seed
        )
        y_pred_par, r_par, ngb_par = fit_ngboost(
            X_train_B, y_train, X_test_B, y_test, use_svgd=use_svgd
        )
        print(f"[Parents only] NGBoost r = {r_par:.3f}")
        plot_pred_vs_true(y_test, y_pred_par, f"NGBoost — parents only (r={r_par:.3f})")
    else:
        print("No non-empty parent set (or target excluded). Skipping parents-only model.")

    return {
        "W_hat": W_hat,
        "feature_names": feature_names,
        "var_names": var_names,
        "parents": parents,
        "r_all": r_all,
        "r_lr": r_lr,
        "r_parents": r_par,
        "ngb_all": ngb_all,
        "ngb_parents": ngb_par,
        "X_train_parents": X_train_B,
        "X_test_parents": X_test_B,
    }


# ---------- Example main ----------

if __name__ == "__main__":
    # choose your dataset here:
    # 1) California Housing
    cali = fetch_california_housing(as_frame=True)
    df_cali = cali.frame.rename(columns={"MedHouseVal": "MedHouseVal"})
    dagma_demo(
        df=df_cali,
        target_name="MedHouseVal",
        include_target=True,
        n_subsample=8000,     # None to use all rows
        reg_w=0.01,
        w_threshold=0.01,
        test_size=0.2,
        seed=42,
        use_svgd=False,
    )

    # 2) Diabetes (same function, different DF/target)
    diab = load_diabetes(as_frame=True)
    df_diab = pd.concat([diab.data, diab.target.rename("target")], axis=1)
    dagma_demo(
        df=df_diab,
        target_name="target",
        include_target=True,
        n_subsample=None,     # small dataset
        reg_w=0.01,
        w_threshold=0.01,
        test_size=0.2,
        seed=42,
        use_svgd=False,
    )
