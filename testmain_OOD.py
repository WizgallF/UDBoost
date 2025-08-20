#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature-wise OOD detection with NGBoost (NIG) + SHAP on variance components.

- Train on Wine Quality RED.
- Build OOD by swapping selected feature columns with WHITE.
- Use SHAP TreeExplainer (chain rule over NIG params) to explain:
  predictive, epistemic, aleatoric variances.

Requirements:
    pip install ngboost shap xgboost pandas numpy scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ngboost import NGBRegressor
from ngboost.distns import NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD
 # your custom NIGLogScore also works

# -------------------------------
# Data loading & prep
# -------------------------------

def load_wine_quality():
    """Load UCI Wine Quality (red & white)."""
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    red = pd.read_csv(url_red, sep=";")
    white = pd.read_csv(url_white, sep=";")
    # unify column names (they already match in UCI release)
    return red, white

def split_Xy(df, target="quality"):
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(float).values
    return X, y

# -------------------------------
# NGBoost training
# -------------------------------

def train_ngb_nig(X_train, y_train_z, random_state=42):
    ngb = NGBRegressor(
        Dist=NormalInverseGamma,
        Score=NIGLogScoreSVGD,
        n_estimators=300,
        learning_rate=0.1,
        natural_gradient=True,
        verbose=True,
        random_state=random_state,
        epistemic_scaling=True
    )
    ngb.fit(X_train.values, y_train_z)
    return ngb

# -------------------------------
# Chain-rule SHAP over NIG params
# (standalone adaptation of your method)
# -------------------------------

def _get_ngb_nig_params(model, X):
    """Return dict of params mu, lam, alpha, beta from model.pred_dist(X)."""
    dist = model.pred_dist(X.values)
    params = dist.params  # dict-like
    # keep np arrays aligned with X rows
    mu    = np.asarray(params["mu"])
    lam   = np.asarray(params["lam"])
    alpha = np.asarray(params["alpha"])
    beta  = np.asarray(params["beta"])
    return mu, lam, alpha, beta

def _predict_param(model, X, which="lam"):
    """Convenience: predict a single NIG param for KernelExplainer fallback."""
    mu, lam, alpha, beta = _get_ngb_nig_params(model, X)
    if which == "mu":    return mu
    if which == "lam":   return lam
    if which == "alpha": return alpha
    if which == "beta":  return beta
    raise ValueError(which)

def feature_importance_variance_ngb(model, X, feature_names=None, show_plots=True, top_n=None, title=None):
    """
    Compute SHAP values for predictive / epistemic / aleatoric variance
    via TreeExplainer over NIG parameters + chain rule, as in your function.

    If TreeExplainer(model, model_output=k) fails for NGBoost, we fallback
    to KernelExplainer per-parameter.
    """
    # 1) get params
    mu, lam, alpha, beta = _get_ngb_nig_params(model, X)

    # 2) your variance definitions (matching your provided function)
    var_pred = beta / (lam * (alpha - 1))              # predictive
    var_epi  = beta / (lam * (alpha - 1)**2)           # epistemic
    var_alea = beta / (alpha - 1)                      # aleatoric

    # 3) derivatives (chain rule)
    dpred_dbeta  = 1.0 / (lam * (alpha - 1))
    dpred_dalpha = -beta / (lam * (alpha - 1)**2)
    dpred_dlam   = -beta / (lam**2 * (alpha - 1))

    depi_dbeta   = 1.0 / (lam * (alpha - 1)**2)
    depi_dalpha  = -2.0 * beta / (lam * (alpha - 1)**3)
    depi_dlam    = -beta / (lam**2 * (alpha - 1)**2)

    dalea_dbeta  = 1.0 / (alpha - 1)
    dalea_dalpha = -beta / (alpha - 1)**2

    # 4) build (param-wise) SHAP explainers
    # First try TreeExplainer(model, model_output=k)
    use_kernel_fallback = False
    try:
        expl_lam   = shap.TreeExplainer(model, model_output=1)
        sh_lam     = expl_lam.shap_values(X)
        expl_alpha = shap.TreeExplainer(model, model_output=2)
        sh_alpha   = expl_alpha.shap_values(X)
        expl_beta  = shap.TreeExplainer(model, model_output=3)
        sh_beta    = expl_beta.shap_values(X)
    except Exception:
        use_kernel_fallback = True

    if use_kernel_fallback:
        print("HI")
        # KernelExplainer fallback – slower; subsample background
        bg = X.sample(min(200, len(X)), random_state=0)
        f_lam   = lambda Z: _predict_param(model, pd.DataFrame(Z, columns=X.columns), "lam")
        f_alpha = lambda Z: _predict_param(model, pd.DataFrame(Z, columns=X.columns), "alpha")
        f_beta  = lambda Z: _predict_param(model, pd.DataFrame(Z, columns=X.columns), "beta")

        ex_lam   = shap.KernelExplainer(f_lam,   bg, link="identity")
        ex_alpha = shap.KernelExplainer(f_alpha, bg, link="identity")
        ex_beta  = shap.KernelExplainer(f_beta,  bg, link="identity")

        sh_lam   = ex_lam.shap_values(X, nsamples="auto")
        sh_alpha = ex_alpha.shap_values(X, nsamples="auto")
        sh_beta  = ex_beta.shap_values(X, nsamples="auto")

    # 5) chain rule (broadcast across feature axis)
    sh_lam   = np.asarray(sh_lam)
    sh_alpha = np.asarray(sh_alpha)
    sh_beta  = np.asarray(sh_beta)

    shap_pred = (dpred_dbeta[:, None]  * sh_beta
               + dpred_dalpha[:, None] * sh_alpha
               + dpred_dlam[:,   None] * sh_lam)

    shap_epi  = (depi_dbeta[:,  None]  * sh_beta
               + depi_dalpha[:, None]  * sh_alpha
               + depi_dlam[:,   None]  * sh_lam)

    shap_alea = (dalea_dbeta[:, None]  * sh_beta
               + dalea_dalpha[:, None] * sh_alpha)

    # 6) visuals (optional)
    if show_plots:
        max_display = top_n if top_n is not None else min(12, X.shape[1])
        shap.summary_plot(shap_epi, X, feature_names=feature_names, show=True, max_display=max_display)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{title}.png", dpi=300)

    return {
        "shap_pred": shap_pred,
        "shap_epi":  shap_epi,
        "shap_alea": shap_alea,
        "var_pred":  var_pred,
        "var_epi":   var_epi,
        "var_alea":  var_alea,
    }

# -------------------------------
# OOD builder: feature-wise swap
# -------------------------------

def make_feature_swap_ood(X_base, X_source, cols_ood, random_state=0):
    """
    Return a copy of X_base where columns in cols_ood are replaced
    by rows sampled (with permutation) from X_source (same columns/names).
    """
    rng = np.random.default_rng(random_state)
    X_ood = X_base.copy()
    idx_src = rng.choice(len(X_source), size=len(X_base), replace=True)
    for c in cols_ood:
        X_ood[c] = X_source.iloc[idx_src][c].values
    return X_ood


# -------------------------------
# Main experiment
# -------------------------------

def main():
    np.random.seed(42)

    # 1) data
    red, white = load_wine_quality()
    X_red, y_red = split_Xy(red, target="quality")
    X_white, _   = split_Xy(white, target="quality")

    # 2) train/test on RED (z-score the target)
    X_train, X_test, y_train, y_test = train_test_split(
        X_red, y_red, test_size=0.25, random_state=42
    )
    scaler_y = StandardScaler()
    y_train_z = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # 3) NGBoost (NIG)
    ngb = train_ngb_nig(X_train, y_train_z, random_state=42)

    # 4) Feature-wise OOD by swapping a few columns from WHITE
    #    pick 3 varied features, or choose your own list
    cols_ood = ["residual sugar", "chlorides", "sulphates"]
    X_test_ood = make_feature_swap_ood(X_test, X_white, cols_ood, random_state=0)

    print(f"Swapped OOD columns: {cols_ood}")

    # 5) SHAP on epistemic (and the others) — IN-DISTRIBUTION
    res_in = feature_importance_variance_ngb(
        model=ngb,
        X=X_test,
        feature_names=X_test.columns,
        show_plots=True,
        top_n=12,
        title="SHAP unaltered"

    )

    # 6) SHAP on OOD set
    res_ood = feature_importance_variance_ngb(
        model=ngb,
        X=X_test_ood,
        feature_names=X_test_ood.columns,
        show_plots=True,
        top_n=12,
        title="SHAP with OOD cols"
    )

    # 7) Compare epistemic distributions (should increase under OOD)
    epi_in  = res_in["var_epi"]
    epi_ood = res_ood["var_epi"]

    plt.figure(figsize=(7,4))
    plt.hist(epi_in,  bins=40, alpha=0.6, density=True, label="IN (Red test)")
    plt.hist(epi_ood, bins=40, alpha=0.6, density=True, label="OOD (feature swap from White)")
    plt.xlabel("Epistemic variance")
    plt.ylabel("Density")
    plt.title("Epistemic variance shift under feature-wise OOD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 8) Quick sanity: point predictions on test
    y_pred_z = ngb.predict(X_test.values)
    y_pred   = scaler_y.inverse_transform(y_pred_z.reshape(-1,1)).ravel()

    from scipy.stats import pearsonr
    r,_ = pearsonr(y_pred, y_test)
    print(f"[IN] Pearson r (point prediction): {r:.3f}")

if __name__ == "__main__":
    main()
