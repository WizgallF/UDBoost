import sys
import os

# Add the path to the local ngboost module
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ngboost")))
from ngboost import NGBRegressor
from ngboost import NGBEnsembleRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from ngboost.distns import Normal, NormalInverseGamma, NIGLogScore
import matplotlib.pyplot as plt
from benchmark.synthetic_data import SyntheticDataGenerator
from benchmark.benchmark import BenchmarkUncertainty
#Load Boston housing dataset
#data_url = "http://lib.stat.cmu.edu/datasets/boston"
#raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
#X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#Y = raw_df.values[1::2, 2]
# Standardize y and call the corresponding variable z
#X, y = load_diabetes(return_X_y=True, as_frame=True)
# Step 1: Generate the synthetic data
data_gen = SyntheticDataGenerator()
#df = data_gen.gen_aleatoric_benchmark(
#    n_samples=1000,
#    noise_levels=[0.0, 0.25, 0.5, 0.75, 1],
#    noise_scale=2,
#    random_seed=42,
#    plot=True,
#    func=lambda x: (x / 500)**3
#)
def train_and_evaluate(X, y):
    # Step 2: Extract features and target
    X = df[['x']].values
    y = df['y_noisy'].values

    # Step 3: Normalize the target
    z = (y - y.mean()) / y.std()

    # Step 4: Track indices for train/test split
    indices = np.arange(len(df))
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, z, indices, test_size=0.2, random_state=42
    )

    # Step 5: Get the true aleatoric stds only for the test set
    true_noise_std_test = df.loc[idx_test, 'true_noise_std'].values

    # Step 6: Fit the ensemble regressor
    ngb_ensemble = NGBEnsembleRegressor(n_regressors=10)
    ngb_ensemble.fit(X_train, Y_train)

    # Step 7: Get aleatoric predictions
    y_pred_all = ngb_ensemble.pred_uncertainty(X_test)
    return y_pred_all, X_test, Y_test, true_noise_std_test


# Step 8: Benchmark the aleatoric uncertainty
benchmark = BenchmarkUncertainty()
#benchmark.within_sample_aleatoric_uncertainty(
#    X=X_test,
#    y=Y_test,
#    true_aleatoric_uncertainty=true_noise_std_test,
#    aleatoric_uncertainty=y_pred_all['aleatoric'].values,
#    y_pred=y_pred_all['mean'].values
#)

df = data_gen.gen_epistemic_benchmark(n_samples=1000,
    data_densities=[1, 0.75, 0.5, 0.25, 0.1],
    noise_std=0.05,
    random_seed=42,
    plot=True,
    func=lambda x: x**3
)
# Step 9: Train and evaluate the epistemic uncertainty model
y_pred_all, X_test, Y_test, true_noise_std_test = train_and_evaluate(df[['x']].values, df['y_noisy'].values)
# Step 10: Benchmark the epistemic uncertainty
benchmark.within_sample_epistemic_uncertainty(
    X=X_test,
    y=Y_test,
    true_epistemic_uncertainty=true_noise_std_test,
    epistemic_uncertainty=y_pred_all['epistemic'].values,
    y_pred=y_pred_all['mean'].values
)

## Fit the Normal-Inverse-Gamma regressor
#ngb_nig = NGBRegressor(Dist=NormalInverseGamma,
#    Score=NIGLogScore,
#    n_estimators=500,
#    learning_rate=0.01, 
#    verbose=True,
#    natural_gradient=True,)
#ngb_nig.fit(X_train, Y_train)
#y_pred_nig = ngb_nig.predict(X_test)
#
## Calculate MSE
#mse_ensemble = mean_squared_error(Y_test, y_pred_ensemble)
#mse_nig = mean_squared_error(Y_test, y_pred_nig)

# Sort by true values for proper shading
#sorted_indices = np.argsort(Y_test)
#Y_sorted = Y_test[sorted_indices]
#y_pred_sorted = y_pred_ensemble[sorted_indices]
#std_sorted = np.sqrt(variance_ensemble[sorted_indices])

# Plot
#plt.figure(figsize=(10, 6))
#plt.scatter(Y_test, y_pred_ensemble, label=f'Ensemble Regressor (MSE={mse_ensemble:.2f})', alpha=0.6)
#plt.scatter(Y_test, y_pred_nig, label=f'NIG Regressor (MSE={mse_nig:.2f})', alpha=0.6)
#plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
#
## Plot shaded variance band
#plt.fill_between(
#    Y_test,
#    y_pred_ensemble - np.sqrt(variance_ensemble),
#    y_pred_ensemble + np.sqrt(variance_ensemble),
#    color='tab:blue',
#    alpha=0.2,
#    label='Ensemble ±1 std dev'
#)
#
#plt.xlabel('True Values')
#plt.ylabel('Predicted Values')
#plt.title('Comparison of NGB Ensemble vs NIG Regressor with Uncertainty')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()



#Y_preds = ngb.predict(X_test)
#Y_dists = ngb.pred_dist(X_test)
#
## test Mean Squared Error
#test_MSE = mean_squared_error(Y_preds, Y_test)
#print('Test MSE', test_MSE)
#
## test Negative Log Likelihood
#test_NLL = -Y_dists.logpdf(Y_test).mean()
#print('Test NLL', test_NLL)