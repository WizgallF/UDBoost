# --- Local imports --- # 
# - NGBoost - #
from ngboost import NGBRegressor
from ngboost import NGBEnsembleRegressor
from ngboost.distns import Normal, NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD

# - Benchmark imports -
from benchmark.synthetic_data import SyntheticDataGenerator
from benchmark.benchmark import BenchmarkUncertainty

# Catboost for comparison
from catboost import CatBoostRegressor


# --- External imports --- #
# - Numpy import - #
import numpy as np


# --- Generating datasets --- #
SyntheticDataGenerator = SyntheticDataGenerator()

# - Aleatoric Benchmark Data - #
dataset = SyntheticDataGenerator.gen_1d_synthetic_benchmark(
    n_samples=1000,
    noise_levels=[0.2, 0.2, 1, 1.5, 0.8],
    data_densities=[0.2, 0.2, 0.3, 0.4, 0.005],
    random_seed=42,
    func=lambda x: (
        2 * np.sin(18 * x) +
        0.5 * np.cos(15 * x) +
        np.sin(2 * np.pi * x) * np.exp(-x) +
        0.3 * np.sin(18 * x + 0.5) +
        0.2 * np.cos(23 * x - 1.0) +
        5 * x**2
    ),
    normalized_y=False
)
dataset.plot_1d_syn_benchmark(show=True)

# --- Training and evaluating the NGBRegressor on both datasets --- #
# - Change your regressor here - #
regressor = NGBEnsembleRegressor(n_regressors=25, n_estimators=300, ensemble_method='bagging', langevin_noise_scale=1, learning_rate=0.005)
catboostregressors = []
for i in range(25):
    catboostreg = CatBoostRegressor(
        iterations=120,
        loss_function='RMSEWithUncertainty',
        langevin=True,
        verbose=False
    )
    catboostreg.fit(dataset.X.reshape(-1, 1), dataset.y)
    catboostregressors.append(catboostreg)

# Calculate the uncertainties for the CatBoost regressors
parameters = np.array([model.predict(dataset.dataspace.reshape(-1, 1), prediction_type='RMSEWithUncertainty') for model in catboostregressors])
mean_prediction = np.mean(parameters[:, :, 0], axis=0)
print('Mean prediction shape:', mean_prediction.shape)
aleatoric_uncertainty = np.mean(parameters[:, :, 1], axis=0)
print('Aleatoric uncertainty:', aleatoric_uncertainty.shape)
epistemic_uncertainty = np.var(parameters[:, :, 0], axis=0)
cat_uncertainty = {
            'mean': mean_prediction,
            'predictive': aleatoric_uncertainty + epistemic_uncertainty if 'aleatoric_uncertainty' in locals() else None,
            'predicitve_upper_bound': None,
            'predicitve_lower_bound': None,
            'aleatoric': aleatoric_uncertainty if 'aleatoric_uncertainty' in locals() else None,
            'epistemic': epistemic_uncertainty,
            'aleatoric_lower_bound': None,
            'aleatoric_upper_bound': None,
}
# - Fit the regressor on the dataset - #
regressor.fit(dataset.X.reshape(-1, 1), dataset.y)
uncertainty_levi = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1), mode='bayesian_mean')
#uncertainty_levi['epistemic'] = uncertainty_levi['epistemic'] # Scale epistemic uncertainty for better visualization
#uncertainty_momentum = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1), mode='bayesian_mean')
# --- Benchmarking the uncertainty quantification methods --- #
benchmark = BenchmarkUncertainty()
benchmark.benchmark_uncertainty(dataset, uncertainty_levi)
benchmark.benchmark_uncertainty(dataset, cat_uncertainty)
