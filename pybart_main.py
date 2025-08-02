# --- External imports --- #
import numpy as np
from bartz.BART import gbart

# --- Benchmark imports ---
from benchmark.synthetic_data import SyntheticDataGenerator
from benchmark.benchmark import BenchmarkUncertainty

# --- Generate the dataset --- #
SyntheticDataGenerator = SyntheticDataGenerator()

dataset = SyntheticDataGenerator.gen_1d_synthetic_benchmark(
    n_samples=1000,
    noise_levels=[0.05, 0.05, 0.5, 0.75, 0.2],
    data_densities=[0.01, 0.05, 0.5, 0.6, 0.3],
    random_seed=42,
    func=lambda x: (
        2 * np.sin(12 * x) +
        0.5 * np.cos(15 * x) +
        np.sin(2 * np.pi * x) * np.exp(-x) +
        0.3 * np.sin(18 * x + 0.5) +
        0.2 * np.cos(23 * x - 1.0) +
        0.1 * x**2
    ),
    normalized_y=True
)

dataset.plot_1d_syn_benchmark(show=True)
dataset.dataspace = np.linspace(-1, 2, 1000)  # Ensure dataspace is defined for predictions
# --- Replace NGBoost with BART --- #
from stochtree import BARTModel
X_train = dataset.X.reshape(-1, 1)
y_train = dataset.y
X_test = dataset.dataspace.reshape(-1, 1)
model = BARTModel()
model.sample(
    X_train, y_train,
    num_mcmc=1000,
    mean_forest_params={'num_trees': 50},  # enable heteroskedasticity
    variance_forest_params={'num_trees': 50},  # enable heteroskedasticity
    X_test=X_test
)
# --- Predictive mean and uncertainty estimation --- #
# Predictive mean over posterior samples (ndraws, n_points)
mean_samples, variance_samples = model.predict(X_test)  # shape: (num_mcmc, n_test)
print(f"Shape of mean_samples: {mean_samples.shape}, variance_samples: {variance_samples.shape}")
# --- 1. Predictive Mean ---
pred_mean = mean_samples.mean(axis=1)

# --- 2. Epistemic Uncertainty ---
epistemic_std = mean_samples.var(axis=1)

# --- 3. Aleatoric Uncertainty ---
aleatoric_std = variance_samples.mean(axis=0)

# --- 4. Total Uncertainty ---
total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)


# Predictive mean and posterior draws
#y_preds = np.array([regressor.predict(dataset.dataspace.reshape(-1, 1)) for _ in range(100)])
#pred_mean = y_preds.mean(axis=0)
#pred_std = y_preds.std(axis=0)  # Epistemic uncertainty from BART
#
## Optional: Add aleatoric uncertainty estimation if needed
#
## --- Combine into uncertainty object (mean ± std) --- #
## Format to match expected benchmark input
uncertainty = {
    "mean": pred_mean,
    "epistemic": epistemic_std,
    "aleatoric": aleatoric_std,  
    "predictive": total_std  # If you want to keep it simple
}
#
## --- Benchmarking --- #
benchmark = BenchmarkUncertainty()
benchmark.benchmark_uncertainty(dataset, uncertainty)
