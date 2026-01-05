from __future__ import annotations

import pathlib
import sys

import numpy as np


def _ensure_local_udboost_import():
    repo_root = pathlib.Path(__file__).resolve().parent
    src = repo_root / "UDBoost" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _make_synthetic_regression(n: int = 4000, d: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y_clean = np.sin(X[:, 0]) + 0.5 * X[:, 1] - 0.25 * X[:, 2] ** 2
    noise = (0.1 + 0.2 * (np.abs(X[:, 0]) > 1.0)) * rng.normal(size=n)
    y = (y_clean + noise).astype(np.float32)

    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


def resnet_bagging_ensemble_demo():
    try:
        import torch
    except ImportError:
        print("Skipping ResNet bagging demo: torch not installed.")
        return

    _ensure_local_udboost_import()

    from udboost.nn import (
        TabularEDLRegressorResNet,
        bagging_ensemble_predict,
        disentangle_regression_uncertainty,
    )

    X_train, y_train, X_test, y_test = _make_synthetic_regression(n=3000, d=24, seed=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_fn():
        return TabularEDLRegressorResNet(
            in_features=X_train.shape[1],
            d=256,
            n_blocks=4,
            dropout=0.05,
            norm="layernorm",
            out_features=1,
        )

    fit_kwargs = dict(
        n_epochs=50,
        batch_size=512,
        lr=2e-3,
        weight_decay=1e-4,
        device=device,
        loss="edl_nig",
        edl_reg_weight=1e-2,
        verbose=False,
    )

    ens_pred = bagging_ensemble_predict(
        model_fn,
        X_train,
        y_train,
        X_test,
        n_members=8,
        sample_fraction=0.8,
        replace=True,
        rng=0,
        fit_kwargs=fit_kwargs,
        predict_kwargs=dict(device=device),
    )
    unc = disentangle_regression_uncertainty(ens_pred, mode="moment")

    mean = unc["mean"].reshape(-1)
    ale = unc["aleatoric"].reshape(-1) if unc["aleatoric"] is not None else None
    epi = unc["epistemic"].reshape(-1)
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))

    print("\n[ResNet Bagging Ensemble]")
    print(f"device={device}  rmse={rmse:.4f}")
    if ale is not None:
        print(f"aleatoric var: mean={float(ale.mean()):.6f}  p95={float(np.quantile(ale, 0.95)):.6f}")
    print(f"epistemic var: mean={float(epi.mean()):.6f}  p95={float(np.quantile(epi, 0.95)):.6f}")


def edl_resnet_demo():
    try:
        import torch
    except ImportError:
        print("Skipping EDL-MLP demo: torch not installed.")
        return

    _ensure_local_udboost_import()

    from udboost.nn import TabularEDLRegressorResNet, edl_nig_uncertainty, fit_regressor_torch

    X_train, y_train, X_test, y_test = _make_synthetic_regression(n=4000, d=16, seed=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabularEDLRegressorResNet(
        in_features=X_train.shape[1],
        d=256,
        n_blocks=4,
        dropout=0.05,
        norm="layernorm",
        out_features=1,
    )

    model = fit_regressor_torch(
        model,
        X_train,
        y_train,
        n_epochs=80,
        batch_size=512,
        lr=2e-3,
        weight_decay=1e-4,
        device=device,
        loss="edl_nig",
        edl_reg_weight=1e-2,
        verbose=False,
    )

    X_t = torch.as_tensor(X_test, device=device, dtype=torch.float32)
    with torch.no_grad():
        mu, v, alpha, beta = model(X_t)
        unc_t = edl_nig_uncertainty(mu, v, alpha, beta)

    mean = unc_t["mean"].detach().cpu().numpy().reshape(-1)
    ale = unc_t["aleatoric"].detach().cpu().numpy().reshape(-1)
    epi = unc_t["epistemic"].detach().cpu().numpy().reshape(-1)
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))

    print("\n[EDL-ResNet (Amini et al. 2020)]")
    print(f"device={device}  rmse={rmse:.4f}")
    print(f"aleatoric var: mean={float(ale.mean()):.6f}  p95={float(np.quantile(ale, 0.95)):.6f}")
    print(f"epistemic var: mean={float(epi.mean()):.6f}  p95={float(np.quantile(epi, 0.95)):.6f}")


def resnet_laplace_demo():
    try:
        import torch
    except ImportError:
        print("Skipping ResNet Laplace demo: torch not installed.")
        return

    _ensure_local_udboost_import()

    from udboost.nn import (
        TabularRegressorResNet,
        disentangle_regression_uncertainty,
        fit_laplace_diagonal_ensemble_predict,
    )

    X_train, y_train, X_test, y_test = _make_synthetic_regression(n=3000, d=24, seed=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_fn():
        return TabularRegressorResNet(
            in_features=X_train.shape[1],
            d=256,
            n_blocks=4,
            dropout=0.05,
            norm="layernorm",
            out_features=1,
        )

    ens_pred = fit_laplace_diagonal_ensemble_predict(
        model_fn,
        X_train,
        y_train,
        X_test,
        n_members=12,
        fit_kwargs=dict(
            n_epochs=50,
            batch_size=512,
            lr=2e-3,
            weight_decay=1e-4,
            device=device,
            loss="mse",
            verbose=False,
        ),
        laplace_kwargs=dict(
            device=device,
            loss="mse",
            prior_precision=1.0,
            temperature=1.0,
            batch_size_train=512,
            batch_size_pred=4096,
        ),
    )
    unc = disentangle_regression_uncertainty(ens_pred, mode="moment")

    mean = unc["mean"].reshape(-1)
    epi = unc["epistemic"].reshape(-1)
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))

    print("\n[ResNet Laplace Diagonal Ensemble]")
    print(f"device={device}  rmse={rmse:.4f}")
    print(f"epistemic var: mean={float(epi.mean()):.6f}  p95={float(np.quantile(epi, 0.95)):.6f}")


def resnet_mc_dropout_demo():
    try:
        import torch
    except ImportError:
        print("Skipping ResNet MC Dropout demo: torch not installed.")
        return

    _ensure_local_udboost_import()

    from udboost.nn import (
        TabularRegressorResNet,
        disentangle_regression_uncertainty,
        fit_mc_dropout_ensemble_predict,
    )

    X_train, y_train, X_test, y_test = _make_synthetic_regression(n=3000, d=24, seed=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_fn():
        return TabularRegressorResNet(
            in_features=X_train.shape[1],
            d=256,
            n_blocks=4,
            dropout=0.05,
            norm="layernorm",
            out_features=1,
        )

    ens_pred = fit_mc_dropout_ensemble_predict(
        model_fn,
        X_train,
        y_train,
        X_test,
        n_members=30,
        fit_kwargs=dict(
            n_epochs=50,
            batch_size=512,
            lr=2e-3,
            weight_decay=1e-4,
            device=device,
            loss="mse",
            verbose=False,
        ),
        predict_kwargs=dict(
            device=device,
            batch_size=4096,
        ),
    )
    unc = disentangle_regression_uncertainty(ens_pred, mode="moment")

    mean = unc["mean"].reshape(-1)
    epi = unc["epistemic"].reshape(-1)
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))

    print("\n[ResNet MC Dropout Ensemble]")
    print(f"device={device}  rmse={rmse:.4f}")
    print(f"epistemic var: mean={float(epi.mean()):.6f}  p95={float(np.quantile(epi, 0.95)):.6f}")


def resnet_deup_demo():
    try:
        import torch
    except ImportError:
        print("Skipping ResNet DEUP demo: torch not installed.")
        return

    _ensure_local_udboost_import()

    from udboost.nn import (
        TabularRegressorMLP,
        TabularRegressorResNet,
        deup_fit_regressor_torch,
        deup_predict_regressor_torch,
    )

    X_train, y_train, X_test, y_test = _make_synthetic_regression(n=3000, d=24, seed=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predictor_fn():
        return TabularRegressorResNet(
            in_features=X_train.shape[1],
            d=256,
            n_blocks=4,
            dropout=0.0,
            norm="layernorm",
            out_features=1,
        )

    def uncertainty_model_fn(in_features: int):
        return TabularRegressorMLP(
            in_features=in_features,
            hidden_features=(256, 256),
            dropout=0.0,
            norm="layernorm",
            out_features=1,
        )

    def aleatoric_var_fn(X):
        scale = 0.1 + 0.2 * (np.abs(X[:, 0]) > 1.0)
        return (scale * scale).astype(np.float32)

    deup = deup_fit_regressor_torch(
        predictor_fn,
        uncertainty_model_fn,
        X_train,
        y_train,
        n_folds=5,
        rng=0,
        include_mean_feature=False,
        target="epistemic",
        aleatoric_mode="from_fn",
        aleatoric_var_fn=aleatoric_var_fn,
        fit_kwargs_predictor=dict(
            n_epochs=40,
            batch_size=512,
            lr=2e-3,
            weight_decay=1e-4,
            device=device,
            loss="mse",
            verbose=False,
        ),
        fit_kwargs_uncertainty=dict(
            n_epochs=40,
            batch_size=512,
            lr=2e-3,
            weight_decay=1e-4,
            device=device,
            loss="mse",
            verbose=False,
        ),
    )

    unc = deup_predict_regressor_torch(deup, X_test, device=device, aleatoric_var_fn=aleatoric_var_fn)
    mean = unc["mean"].reshape(-1)
    epi = unc["epistemic"].reshape(-1)
    ale = unc["aleatoric"].reshape(-1) if unc["aleatoric"] is not None else None
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))

    print("\n[ResNet DEUP (direct epistemic prediction)]")
    print(f"device={device}  rmse={rmse:.4f}")
    if ale is not None:
        print(f"aleatoric var: mean={float(ale.mean()):.6f}  p95={float(np.quantile(ale, 0.95)):.6f}")
    print(f"epistemic var: mean={float(epi.mean()):.6f}  p95={float(np.quantile(epi, 0.95)):.6f}")


if __name__ == "__main__":
    resnet_bagging_ensemble_demo()
    resnet_laplace_demo()
    resnet_mc_dropout_demo()
    edl_resnet_demo()
    resnet_deup_demo()
