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
dataset = SyntheticDataGenerator.gen_synthetic_benchmark(
    n_samples=1000,
    noise_levels=[0, 0.0, 0.5, 0.75, 0.2],
    data_densities=[0.02, 0.05, 0.5, 0.6, 0.3],
    random_seed=42,
    func=lambda x: (
        2 * np.sin(8 * x) +
        0.5 * np.cos(10 * x) +
        np.sin(2 * np.pi * x) * np.exp(-x) +
        0.3 * np.sin(15 * x + 0.5) +
        0.2 * np.cos(20 * x - 1.0) +
        0.1 * x**2
    ),
    normalized_y=True
)
dataset.plot(show=True)

# --- Training and evaluating the NGBRegressor on both datasets --- #
# - Change your regressor here - #
# - NGBoost version - #
regressor = NGBRegressor(
    Dist=NormalInverseGamma,
    Score=NIGLogScore,
    n_estimators=500,
    learning_rate=0.01,
    verbose=True,
    natural_gradient=True,
)
    
regressor.fit(dataset.X.reshape(-1, 1), dataset.y)
uncertainty = regressor.pred_uncertainty(dataset.X.reshape(-1, 1))

# - NGBEnsemble version - #
#regressor = NGBEnsembleRegressor(n_regressors=10)
#
#regressor.fit(dataset.X.reshape(-1, 1), dataset.y)
uncertainty = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1))


# --- Benchmarking the uncertainty quantification methods --- #
benchmark = BenchmarkUncertainty()
benchmark.benchmark_uncertainty(dataset, uncertainty)
