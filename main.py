# --- Local imports --- # 
# - NGBoost - #
from ngboost import NGBRegressor
from ngboost import NGBEnsembleRegressor
from ngboost.distns import Normal, NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD

# - Benchmark imports -
from benchmark.synthetic_data import SyntheticDataGenerator
from benchmark.benchmark import BenchmarkUncertainty

# - SKLearn imports - #
from sklearn.tree import DecisionTreeRegressor


# Catboost for comparison
from catboost import CatBoostRegressor


# --- External imports --- #
# - Numpy import - #
import numpy as np


# --- Generating datasets --- #
SyntheticDataGenerator = SyntheticDataGenerator()

def calculate_catboost_uncertainty(dataset, n_models=20, iterations=200):
    catboostregressors = []
    for i in range(n_models):
        catboostreg = CatBoostRegressor(
            iterations=iterations,
            loss_function='RMSEWithUncertainty',
            langevin=True,
            posterior_sampling=True,
            random_seed=i,
            verbose=False
        )
        catboostreg.fit(dataset.X.reshape(-1, 1), dataset.y)
        catboostregressors.append(catboostreg)

    parameters = np.array([
        model.predict(dataset.dataspace.reshape(-1, 1), prediction_type='RMSEWithUncertainty')
        for model in catboostregressors
    ])
    mean_prediction = np.mean(parameters[:, :, 0], axis=0)
    aleatoric_uncertainty = np.mean(parameters[:, :, 1], axis=0)
    epistemic_uncertainty = np.var(parameters[:, :, 0], axis=0) * 100
    cat_uncertainty = {
        'mean': mean_prediction,
        'predictive': aleatoric_uncertainty + epistemic_uncertainty,
        'predicitve_upper_bound': None,
        'predicitve_lower_bound': None,
        'aleatoric': aleatoric_uncertainty,
        'epistemic': epistemic_uncertainty,
        'aleatoric_lower_bound': None,
        'aleatoric_upper_bound': None,
    }
    return cat_uncertainty

# - Aleatoric Benchmark Data - #
dataset = SyntheticDataGenerator.gen_1d_synthetic_benchmark(
    n_samples=1000,
    noise_levels=[0.2, 0.2, 2, 1, 0.8],
    data_densities=[0.2, 0.2, 0.3, 0.4, 0.1],
    random_seed=42,
    func=lambda x: (
        2 * np.sin(15 * x) +
        #0.5 * np.cos(15 * x) +
        #np.sin(2 * np.pi * x) * np.exp(-x) + 
        #0.3 * np.sin(18 * x + 0.5) +
        #0.2 * np.cos(23 * x - 1.0) +
        5 * x**2
    ),
    normalized_y=False
)
dataset.plot_1d_syn_benchmark(show=False)

# --- Training and evaluating the NGBRegressor on both datasets --- #
# - Change your regressor here - #


regressor = NGBRegressor(n_regressors=10, metadistribution_method='bagging', SGLB=False, n_estimators=200, min_samples_leaf=5, max_depth=3, bagging_frac=0.7)

# - Fit the regressor on the dataset - #
regressor.fit(dataset.X.reshape(-1, 1), dataset.y)
uncertainty_levi = regressor.pred_uncertainty(dataset.dataspace.reshape(-1, 1), mode='bayesian_kl')
uncertainty_levi['epistemic'] = uncertainty_levi['epistemic'] * 100 # Scale epistemic uncertainty for better visualization

# --- Benchmarking the uncertainty quantification methods --- #
benchmark = BenchmarkUncertainty()
benchmark.benchmark_uncertainty(dataset, uncertainty_levi)
cat_uncertainty = calculate_catboost_uncertainty(dataset)
benchmark.benchmark_uncertainty(dataset, cat_uncertainty)