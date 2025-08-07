# --- Local imports --- # 
# - NGBoost - #
from sklearn.linear_model import LinearRegression
from ngboost import NGBRegressor
from ngboost import NGBEnsembleRegressor
from ngboost.distns import (
    Normal, 
    NormalInverseGamma, 
    NIGLogScore, 
    NIGLogScoreSVGD
)

# - Benchmark imports -
from benchmark.synthetic_data import SyntheticDataGenerator
from benchmark.benchmark import BenchmarkUncertainty
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from castle.datasets.simulator import IIDSimulation

# --- External imports --- #
# - Numpy import - #
import numpy as np
from castle.datasets import DAG, simulator                
                                


# --- Generating datasets --- #
SyntheticDataGenerator = SyntheticDataGenerator()

# - Aleatoric Benchmark Data - #
dataset = SyntheticDataGenerator.gen_1d_synthetic_benchmark(
    n_samples=1000,
    noise_levels=[0.1, 0.25, 1, 1.5, 0.8],
    data_densities=[0.01, 0.1, 0.5, 0.6, 0],
    random_seed=42,
    func=lambda x: (
        2 * np.sin(18 * x) +
        0.5 * np.cos(15 * x) +
        np.sin(2 * np.pi * x) * np.exp(-x) +
        0.3 * np.sin(18 * x + 0.5) +
        0.2 * np.cos(23 * x - 1.0) +
        0.1 * x**2
    ),
    normalized_y=False
)
dataset.plot_1d_syn_benchmark(show=True)

# --- Training and evaluating the NGBRegressor on both datasets --- #
# - Change your regressor here - #
# - NIG version - #

y = dataset.y

    # Step 3: Normalize the target
z = (y - y.mean()) / y.std()
dataset.y = z

regressor = NGBRegressor(
    Dist=NormalInverseGamma,
    Score=NIGLogScore,
    n_estimators=500,
    learning_rate=0.1,
    max_depth=4,
    alpha=0.001,
    verbose=True,
    natural_gradient=True,
    epistemic_scaling=True, 
    metadistribution_method="evidential_regression"   # Enable epistemic scaling
)
    
regressor.fit(dataset.X.reshape(-1, 1), dataset.y)
uncertainty_levi = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1))
#uncertainty_momentum = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1), mode='bayesian_mean')
# --- Benchmarking the uncertainty quantification methods --- #
benchmark = BenchmarkUncertainty()
benchmark.benchmark_uncertainty(dataset, uncertainty_levi)
#benchmark.benchmark_uncertainty(dataset, uncertainty_momentum)


###
# 1. Sample a random DAG
d = 10                       # total number of variables
exp_edges = 2                # expected parents per node
W = DAG.erdos_renyi(d, exp_edges)      # weighted adjacency matrix

# 2. Use the built-in IID simulator to get a continuous dataset
#    – n: number of samples
#    – func: "linear" (default) means Xj = ∑ W_ij · Xi + ε_j
#    – scale: noise standard deviation
sim = IIDSimulation(
    W=W,                # your weighted adjacency matrix
    n=500,             # number of samples
    method='linear',    # 'linear' or 'nonlinear'
    sem_type='gauss',   # noise type: 'gauss', 'exp', 'gumbel', etc.
    noise_scale=0.5     # noise standard deviation
)

X = sim.X  # shape (5000, d)
# 3. Split into regression features & label
child_nodes = np.where(W.sum(axis=0) != 0)[0]
print("nodes with ≥1 parent:", child_nodes)
if len(child_nodes)==0:
    raise RuntimeError("Your random graph had no edges!")
    
# Pick the last such node
target_idx = child_nodes[-1]
print("→ using target_idx =", target_idx, "with parents", np.where(W[:,target_idx]!=0)[0])
# choose node‐(d-1) as your “y”
y = X[:, target_idx]        # continuous target
X_features = np.delete(X, target_idx, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.1, random_state=42
)

lr = LinearRegression().fit(X_features, y)
y_pred_lr = lr.predict(X_features)
r_lr, _ = pearsonr(y_pred_lr, y)
print(f"LinearRegression r² ~ {r_lr**2:.3f},  Pearson r = {r_lr:.3f}")
# 4) Fit your NGBoost on the *scaled* targets
ngb_diab = NGBRegressor(
    Dist=NormalInverseGamma, 
    Score=NIGLogScoreSVGD,
    n_estimators=400,
    max_depth=4,
    natural_gradient=True,
    learning_rate=0.1,
    verbose=False,
    metadistribution_method="evidential_regression"    
)
ngb_diab.fit(X_train, y_train)

y_mean = y_train.mean()
y_std  = y_train.std()  
# Scale the targets to z-space
y_train_z = (y_train - y_mean) / y_std
# 5) Predict (in z-space) and back-transform to original units
y_pred_z = ngb_diab.predict(X_test)
y_pred   = y_pred_z * y_std + y_mean

# 6) Evaluate & plot on original scale
r, _ = pearsonr(y_pred, y_test)
print(f"Diabetes test Pearson r = {r:.3f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
mn, mx = y_test.min(), y_test.max()
plt.plot([mn,mx], [mn,mx], 'k--', lw=1)
plt.xlabel("True Target")
plt.ylabel("NGBoost Prediction")
plt.title(f"Diabetes (150/50 split) — Pearson r = {r:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()
