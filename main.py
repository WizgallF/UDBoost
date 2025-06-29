import sys
import os

# Add the path to the local ngboost module
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ngboost")))
from ngboost import NGBRegressor
#from ngboost import NGBEnsembleRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from ngboost.distns import Normal, NormalInverseGamma, NIGLogScore
import matplotlib.pyplot as plt
#Load Boston housing dataset
#data_url = "http://lib.stat.cmu.edu/datasets/boston"
#raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
#X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#Y = raw_df.values[1::2, 2]
# Standardize y and call the corresponding variable z
X, y = load_diabetes(return_X_y=True, as_frame=True)
z = (y - y.mean()) / y.std()

X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size=0.2)

# Fit the ensemble regressor
#ngb_ensemble = NGBEnsembleRegressor(n_regressors=10)
#ngb_ensemble.fit(X_train, Y_train)
#y_pred_ensemble = ngb_ensemble.predict(X_test, average=False)
#print("Ensemble Regressor Predictions Shape:", y_pred_ensemble.shape)
## Get raw predictions from each regressor in the ensemble
#y_pred_all = ngb_ensemble.predict(X_test, average=False)  # shape: (n_samples, n_regressors)

# Compute mean prediction across regressors
#y_pred_ensemble = np.mean(y_pred_all, axis=1)

# Compute variance across regressors
#variance_ensemble = np.var(y_pred_all, axis=1)


# Fit the Normal-Inverse-Gamma regressor
ngb_nig = NGBRegressor(Dist=NormalInverseGamma,
    Score=NIGLogScore,
    n_estimators=500,
    learning_rate=0.01, 
    verbose=True,
    natural_gradient=True,)
ngb_nig.fit(X_train, Y_train)
y_pred_nig = ngb_nig.predict(X_test)

# Calculate MSE
#mse_ensemble = mean_squared_error(Y_test, y_pred_ensemble)
mse_nig = mean_squared_error(Y_test, y_pred_nig)

# Sort by true values for proper shading
sorted_indices = np.argsort(Y_test)
Y_sorted = Y_test.values[sorted_indices]
#y_pred_sorted = y_pred_ensemble[sorted_indices]
#std_sorted = np.sqrt(variance_ensemble[sorted_indices])

# Plot
plt.figure(figsize=(10, 6))
#plt.scatter(Y_test, y_pred_ensemble, label=f'Ensemble Regressor (MSE={mse_ensemble:.2f})', alpha=0.6)
plt.scatter(Y_test, y_pred_nig, label=f'NIG Regressor (MSE={mse_nig:.2f})', alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)

# Plot shaded variance band
plt.fill_between(
    Y_sorted,
    #y_pred_sorted - std_sorted,
    #y_pred_sorted + std_sorted,
    color='tab:blue',
    alpha=0.2,
    label='Ensemble ±1 std dev'
)

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of NGB Ensemble vs NIG Regressor with Uncertainty')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



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