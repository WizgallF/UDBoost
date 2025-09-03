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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

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
    n_samples=500,
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


# Define base regressor with fixed parameters
base_regressor = NGBRegressor(
    metadistribution_method='None',
    verbose=False,  # Reduce logging output
)

# Define hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300],
    'min_samples_leaf': [2, 5, 10, 20],
    'max_depth': [3],
    'learning_rate': [0.005, 0.01],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_regressor,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)
# Fit the model with hyperparameter optimization
random_search.fit(dataset.X.reshape(-1, 1), dataset.y)

# Get best parameters
best_params = random_search.best_params_
print("Best parameters found: ", best_params)

# Fit an ensemble using best parameters
ensemble_model = NGBRegressor(metadistribution_method='bagging', n_regressors=10, **best_params, bagging_frac=0.75, SGLB=False, verbose=False)
ensemble_model.fit(dataset.X.reshape(-1, 1), dataset.y)

# Use ensemble_model as best_model for further evaluation
best_model = ensemble_model
# - Fit the regressor on the dataset - #
best_model.fit(dataset.X.reshape(-1, 1), dataset.y)
uncertainty_levi = best_model.pred_uncertainty(dataset.dataspace.reshape(-1, 1), mode='bayesian_kl')
uncertainty_levi['epistemic'] = uncertainty_levi['epistemic'] # Scale epistemic uncertainty for better visualization

# --- Benchmarking the uncertainty quantification methods --- #
benchmark = BenchmarkUncertainty()
benchmark.benchmark_uncertainty(dataset, uncertainty_levi)
cat_uncertainty = calculate_catboost_uncertainty(dataset)
benchmark.benchmark_uncertainty(dataset, cat_uncertainty)