# --- Local imports --- # 
# - NGBoost - #
from ngboost import NGBRegressor
from ngboost import NGBEnsembleRegressor
from ngboost.distns import Normal, NormalInverseGamma, NIGLogScore

# - Benchmark imports -
from benchmark.synthetic_data import SyntheticDataGenerator
from benchmark.benchmark import BenchmarkUncertainty


# --- External imports --- #
# - Numpy import - #
import numpy as np


# --- Generating datasets --- #
SyntheticDataGenerator = SyntheticDataGenerator()

# - Aleatoric Benchmark Data - #
dataset = SyntheticDataGenerator.gen_1d_synthetic_benchmark(
    n_samples=1000,
    noise_levels=[0.5, 0, 3, 5, 0.8],
    data_densities=[0.01, 0.1, 0.3, 0.4, 0.005],
    random_seed=42,
    func=lambda x: (
        2 * np.sin(18 * x) +
        0.5 * np.cos(15 * x) +
        np.sin(2 * np.pi * x) * np.exp(-x) +
        0.3 * np.sin(18 * x + 0.5) +
        0.2 * np.cos(23 * x - 1.0) +
        15 * x**2
    ),
    normalized_y=False
)
dataset.plot_1d_syn_benchmark(show=True)

# --- Training and evaluating the NGBRegressor on both datasets --- #
# - Change your regressor here - #
regressor = NGBEnsembleRegressor(n_regressors=20, n_estimators=120, ensemble_method='SGBL', langevin_noise_scale=0.1)
regressor.fit(dataset.X.reshape(-1, 1), dataset.y)
uncertainty_levi = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1), mode='bayesian_mean')
uncertainty_levi['epistemic'] = uncertainty_levi['epistemic'] * 500  # Scale epistemic uncertainty for better visualization
#uncertainty_momentum = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1), mode='bayesian_mean')
# --- Benchmarking the uncertainty quantification methods --- #
benchmark = BenchmarkUncertainty()
benchmark.benchmark_uncertainty(dataset, uncertainty_levi)
#benchmark.benchmark_uncertainty(dataset, uncertainty_momentum)
