"The NGBoost library API"
# pylint: disable=too-many-arguments

# SKlearn
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import resample
from sklearn.utils import check_random_state

# Numpy
import numpy as np


from ngboost.distns import (
    Bernoulli,
    ClassificationDistn,
    LogNormal,
    Normal,
    RegressionDistn,
)
from ngboost.distns.utils import SurvivalDistnClass
from ngboost.helpers import Y_from_censored
from ngboost.learners import default_tree_learner
from ngboost.manifold import manifold
from .ngboost_core import NGBoost
from ngboost.scores import LogScore


class NGBRegressor(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost regression models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates regression.
    Use this class if you want to predict an outcome that could take an
    infinite number of (ordered) values.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Normal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        validation_fraction: Proportion of training data to set
                             aside as validation data for early stopping.
        early_stopping_rounds:      The number of consecutive boosting iterations during which the
                                    loss has to increase before the algorithm stops early.
                                    Set to None to disable early stopping and validation.
                                    None enables running over the full data set.
        SGLB                   : whether to use Stochastic Gradient Langevin Boosting (SGLB) or not.
        langevin_noise_scale: the scale of the Langevin noise to add during training (only used if SGLB=True)

    Output:
        An NGBRegressor object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
        SGLB=False,
        langevin_noise_scale=1,
    ):
        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."

        if not hasattr(
            Dist, "scores"
        ):  # user is trying to use a dist that only has censored scores implemented
            Dist = Dist.uncensor(Score)

        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
            validation_fraction,
            early_stopping_rounds,
            SGLB=SGLB,
            langevin_noise_scale=langevin_noise_scale,
        )

        self._estimator_type = "regressor"

    def __getstate__(self):
        state = super().__getstate__()
        # Remove the unpicklable entries.
        if self.Dist.__name__ == "DistWithUncensoredScore":
            state["Dist"] = self.Dist.__base__
            state["uncensor"] = True
        return state

    def __setstate__(self, state_dict):
        if "uncensor" in state_dict.keys():
            state_dict["Dist"] = state_dict["Dist"].uncensor(state_dict["Score"])
        super().__setstate__(state_dict)

    def pred_uncertainty(self, X):
        """
        Predict the uncertainty of Y at the points X=x

        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p)
                in numeric format

        Output:
            A dict of numpy arrays of the uncertainty estimates of Y with keys:
                "mean": mean of the distribution
                "aleatoric": aleatoric uncertainty of the distribution
                "epistemic": epistemic uncertainty of the distribution
                "predictive": total uncertainty of the distribution
        """
        dist = self.pred_dist(X)
        if hasattr(self.Dist, "pred_uncertainty") & dist.is_EDL == True:
            return dist.pred_uncertainty()    
        else:
            raise NotImplementedError(
                "The distribution does not implement pred_uncertainty method."
            )

class NGBClassifier(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost classification models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates classification.
    Use this class if you want to predict an outcome that could take a discrete number of
    (unordered) values.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Bernoulli
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
    Output:
        An NGBClassifier object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=Bernoulli,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):
        assert issubclass(
            Dist, ClassificationDistn
        ), f"{Dist.__name__} is not useable for classification."
        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )
        self._estimator_type = "classifier"

    def predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : get the prediction at the specified number of boosting iterations

        Output:
            Numpy array of the estimates of P(Y=k|X=x). Will have shape (n, K)
        """
        return self.pred_dist(X, max_iter=max_iter).class_probs()

    def staged_predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x at multiple boosting iterations

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : largest number of boosting iterations to get the prediction for

        Output:
            A list of of the estimates of P(Y=k|X=x) of shape (n, K),
            one per boosting stage up to max_iter
        """
        return [
            dist.class_probs() for dist in self.staged_pred_dist(X, max_iter=max_iter)
        ]


class NGBSurvival(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost survival models.

    NGBSurvival is a wrapper for the generic NGBoost class that facilitates survival analysis.
    Use this class if you want to predict an outcome that could take an infinite number of
    (ordered) values, but right-censoring is present in the observed data.

     Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. LogNormal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
    Output:
        An NGBSurvival object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=LogNormal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):

        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."
        if not hasattr(Dist, "censored_scores"):
            raise ValueError(
                f"The {Dist.__name__} distribution does not have any censored scores implemented."
            )

        SurvivalDistn = SurvivalDistnClass(Dist)

        # assert issubclass(Dist, RegressionDistn), f'{Dist.__name__} is not useable for survival.'
        super().__init__(
            SurvivalDistn,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # Both of the below contain SurvivalDistn
        del state["Manifold"]
        state["_basedist"] = state["Dist"]._basedist
        del state["Dist"]
        return state

    def __setstate__(self, state_dict):
        # Recreate the object which could not be pickled
        state_dict["Dist"] = SurvivalDistnClass(state_dict["_basedist"])
        del state_dict["_basedist"]
        state_dict["Manifold"] = manifold(state_dict["Score"], state_dict["Dist"])
        self.__dict__ = state_dict

    # pylint: disable=too-many-positional-arguments
    def fit(self, X, T, E, X_val=None, T_val=None, E_val=None, **kwargs):
        """Fits an NGBoost survival model to the data.
        For additional parameters see ngboost.NGboost.fit

        Parameters:
            X                     : DataFrame object or List or
                                    numpy array of predictors (n x p) in Numeric format
            T                     : DataFrame object or List or
                                    numpy array of times to event or censoring (n) (floats).
            E                     : DataFrame object or List or
                                    numpy array of event indicators (n).
                                    E[i] = 1 <=> T[i] is the time of an event, else censoring time
            T_val                 : DataFrame object or List or
                                    validation-set times, in numeric format if any
            E_val                 : DataFrame object or List or
                                    validation-set event idicators, in numeric format if any
        """

        X = check_array(X, accept_sparse=True)

        if X_val is not None:
            X_val = check_array(X_val, accept_sparse=True)

        return super().fit(
            X,
            Y_from_censored(T, E),
            X_val=X_val,
            Y_val=Y_from_censored(T_val, E_val),
            **kwargs,
        )



class NGBEnsembleRegressor(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost ensemble regression models.

    NGBEnsembleRegressor is a wrapper for the generic NGBoost class that facilitates an ensemble of NGBoost models for regression.
    Use this class if you want to predict an outcome that could take an
    infinite number of (ordered) values using an ensemble of NGBoost models.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Normal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        validation_fraction: Proportion of training data to set
                             aside as validation data for early stopping.
        early_stopping_rounds:      The number of consecutive boosting iterations during which the
                                    loss has to increase before the algorithm stops early.
                                    Set to None to disable early stopping and validation.
                                    None enables running over the full data set.
        n_regressors    : the number of NGBoost regressors to fit in the ensemble
        ensemble_method : the method create the ensemble, options are
                            'stochastic_gradient_boosting' Creating an ensemble by fitting using a subsample of the data
                            'bagging' Creating an ensemble by fitting on random subsamples of the data
                            'SGLB' Creating an ensemble by fitting using a stochastic gradient boosting approach
        langevin_noise_scale: the scale of the Langevin noise to add during training (only used if ensemble_method='SGBL')
        bagging_frac    : the fraction of the data to use for each regressor in the ensemble when using bagging


    Output:
        An NGBRegressor object that can be fit.
    """
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
        n_regressors=10,
        ensemble_method="stochastic_gradient_boosting",
        bagging_frac=0.8,
        langevin_noise_scale=1,
        ):
            assert issubclass(
                Dist, RegressionDistn
            ), f"{Dist.__name__} is not useable for regression."

            if not hasattr(
                Dist, "scores"
            ):  # user is trying to use a dist that only has censored scores implemented
                Dist = Dist.uncensor(Score)

            super().__init__(
                Dist,
                Score,
                Base,
                natural_gradient,
                n_estimators,
                learning_rate,
                minibatch_frac,
                col_sample,
                verbose,
                verbose_eval,
                tol,
                random_state,
                validation_fraction,
                early_stopping_rounds,
            )

            self._estimator_type = "regressor"
            self.n_regressors = n_regressors
            self.ensemble_method = ensemble_method
            self.models = []
            self.bagging_frac = bagging_frac
            self.langevin_noise_scale = langevin_noise_scale

    def fit(self, X, y, **kwargs):
        """
        Fits an NGBoost ensemble regressor to the data.
        For additional parameters see ngboost.NGboost.fit

        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p) in Numeric format
            y : DataFrame object or List or numpy array of target values (n) in Numeric format

        Output:
            self : returns a list of fitted NGBRegressor models
        """
        if self.ensemble_method == "stochastic_gradient_boosting":
            rng = check_random_state(self.random_state)            
            for i in range(self.n_regressors):
                seed = rng.randint(0, 1e6)
                print(f"\n Fitting regressor [{i+1}/{self.n_regressors}] with seed {seed}")
                model = NGBRegressor(
                    Dist=self.Dist,
                    Score=self.Score,
                    Base=self.Base,
                    natural_gradient=self.natural_gradient,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=self.minibatch_frac,
                    col_sample=self.col_sample,
                    verbose=self.verbose,
                    verbose_eval=self.verbose_eval,
                    tol=self.tol,
                    random_state=seed,
                    SGLB=False,  # Use standard NGBoost for ensemble
                )
                model.fit(X, y, **kwargs)
                self.models.append(model)
        elif self.ensemble_method == "bagging":
            for i in range(self.n_regressors):
                print(f"\n Fitting regressor [{i+1}/{self.n_regressors}]")
                # Use a subsample of the data for bagging
                X_resampled, y_resampled = resample(
                    X, y, 
                    replace=False,  # subsample without replacement
                    n_samples=int(len(X) * self.bagging_frac),  # fraction of data
                    random_state=self.random_state
                )
                model = NGBRegressor(
                    Dist=self.Dist,
                    Score=self.Score,
                    Base=self.Base,
                    natural_gradient=self.natural_gradient,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=self.minibatch_frac,
                    col_sample=self.col_sample,
                    verbose=self.verbose,
                    verbose_eval=self.verbose_eval,
                    tol=self.tol,
                    random_state=self.random_state,
                    SGLB=False,  # Use standard NGBoost for bagging
                )
                model.fit(X_resampled, y_resampled, **kwargs)
                self.models.append(model)
        elif self.ensemble_method == "SGLB":
            for i in range(self.n_regressors):
                print(f"\n Fitting regressor [{i+1}/{self.n_regressors}]")
                model = NGBRegressor(
                    Dist=self.Dist,
                    Score=self.Score,
                    Base=self.Base,
                    natural_gradient=self.natural_gradient,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=self.minibatch_frac,
                    col_sample=self.col_sample,
                    verbose=self.verbose,
                    verbose_eval=self.verbose_eval,
                    tol=self.tol,
                    random_state=self.random_state,
                    SGLB=True,  # Use SGLB for ensemble
                    langevin_noise_scale=self.langevin_noise_scale,
                )
                model.fit(X, y, **kwargs)
                self.models.append(model)
        else:
            raise ValueError(f"Ensemble method {self.ensemble_method} not supported. Use 'stochastic_gradient_boosting' or 'SGLB' or 'bagging'.")
        return self
    
    def predict(self, X, average: bool = True):
        """
        Predicts the target values for the input data X using the ensemble of NGBoost models.
        For additional parameters see ngboost.NGboost.predict
        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p) in Numeric format

        Output:
            Numpy array of predicted target values (n)
        """
        if average:
            predictions = [model.predict(X) for model in self.models]
            return np.mean(predictions, axis=0)
        else:
            return np.array([model.predict(X) for model in self.models]).T
        
    def pred_dist(self, X, max_iter=None):
        """
        Predicts the distribution of the target values for the input data X using the ensemble of NGBoost models.
        For additional parameters see ngboost.NGboost.pred_dist

        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p) in Numeric format
            max_iter : get the prediction at the specified number of boosting iterations

        Output:
            A list of predicted distributions for each model in the ensemble
        """
        return [model.pred_dist(X, max_iter=max_iter) for model in self.models]
    
    def pred_uncertainty(self, X, mode: str = 'bayesian_kl'):
        """
        Computes the Bayesian uncertainty disentanglement of the NGBoost ensemble.

        This method calculates both epistemic (model) and aleatoric (data) uncertainties for the predictions made by the ensemble of models. It aggregates predictions and variances from all models in the ensemble to provide a comprehensive uncertainty estimate for each prediction.

        Parameters:
            X (array-like): Input data for which to compute uncertainties

        Returns:
            Dict: A DataFrame containing the following columns for each training sample:
                - mean_prediction: The mean prediction across all models.
                - aleatoric_uncertainty: The mean predicted variance (aleatoric uncertainty) across all models.
                - epistemic_uncertainty: The variance of predictions across all models (epistemic uncertainty).#
                - predictive_uncertainty: The sum of aleatoric and epistemic uncertainties, representing the total predictive uncertainty.
        """
        if mode == 'bayesian_mean':    
            predictions = np.array([model.predict(X) for model in self.models])
            print(predictions)
            parameters = np.array([model.pred_dist(X).params for model in self.models])
            mean_prediction = np.mean(predictions, axis=0)
            aleatoric_uncertainty = np.mean([param['scale'] for param in parameters], axis=0)
            epistemic_uncertainty = np.var(predictions, axis=0)
        elif mode == 'levi_simple':
            predictions = np.array([model.predict(X) for model in self.models])
            parameters = np.array([model.pred_dist(X).params for model in self.models])
            mean_prediction = np.mean(predictions, axis=0)
            aleatoric_uncertainty_lower_bound = np.min([param['scale'] for param in parameters], axis=0)
            aleatoric_uncertainty_upper_bound = np.max([param['scale'] for param in parameters], axis=0)
            # Extract all loc vectors
            locs = [param['loc'] for param in parameters]
            print(f"Number of models: {len(locs)}, Dimension of locs: {len(locs[0]) if locs else 0}")

            # Stack into array of shape (M, D) where M = # of models, D = dimension
            loc_matrix = np.stack(locs)  # shape: (M, D)
            print(f"Shape of loc_matrix: {loc_matrix.shape}")

            # Compute pairwise L2 distances (or differences)
            pairwise_diff = loc_matrix[:, None, :] - loc_matrix[None, :, :]  # shape: (M, M, D)
            print(f"Pairwise differences shape: {pairwise_diff.shape}")

            # Compute max difference across all model pairs and dimensions
            epistemic_uncertainty = np.max(np.abs(pairwise_diff), axis=(0, 1)) 
            print(f"Shape of epistemic_uncertainty: {epistemic_uncertainty.shape}")
        elif mode == 'bayesian_kl':
            predictions = np.array([model.predict(X) for model in self.models])
            parameters = np.array([model.pred_dist(X).params for model in self.models])
            mean_prediction = np.mean(predictions, axis=0)
            aleatoric_uncertainty = 0.5 * (1 + np.log(2*np.pi) + np.mean([np.log(param['scale']**2) for param in parameters], axis=0))
            mean_mu = np.mean([param['loc'] for param in parameters], axis=0)
            mean_sigma = np.mean([param['scale'] for param in parameters], axis=0)
            
            log_mean_sigma = np.mean(np.log([param['scale']**2 for param in parameters]), axis=0)
            mean_mu_deviation = np.mean([param['loc']**2 - mean_mu**2 for param in parameters], axis=0) / mean_sigma**2
            epistemic_uncertainty = 0.5 * (np.log(mean_sigma**2) - log_mean_sigma + mean_mu_deviation)
        elif mode == 'levi_kl':
            predictions = np.array([model.predict(X) for model in self.models])
            parameters = np.array([model.pred_dist(X).params for model in self.models])
            mean_prediction = np.mean(predictions, axis=0)
            # The agent with the minimum entropy
            aleatoric_uncertainty_lower_bound = np.min([0.5 * np.log(2 * np.pi * np.e * param['scale']**2) for param in parameters], axis=0)
            # The agent with the maximum entropy
            aleatoric_uncertainty_upper_bound = np.max([0.5 * np.log(2 * np.pi * np.e * param['scale']**2) for param in parameters], axis=0)
            # Extract all loc vectors
            locs = [param['loc'] for param in parameters]
            variances = [param['scale'] for param in parameters]
            divergences = []
            for model_p in parameters:
                for model_q in parameters:
                    divergence = np.log(model_q['scale']) - np.log(model_p['scale']) + (model_p['scale']**2 + (model_p['loc'] - model_q['loc'])**2) / (2 * model_q['scale']**2) - 0.5
                    divergences.append(divergence)
            divergences = np.array(divergences)
            print(f"Shape of divergences: {divergences.shape}")
            epistemic_uncertainty = np.max(divergences, axis=0)

        return {
            'mean': mean_prediction,
            'predictive': aleatoric_uncertainty + epistemic_uncertainty if 'aleatoric_uncertainty' in locals() else None,
            'predicitve_upper_bound': aleatoric_uncertainty_upper_bound + epistemic_uncertainty if 'aleatoric_uncertainty_upper_bound' in locals() else None,
            'predicitve_lower_bound': aleatoric_uncertainty_lower_bound + epistemic_uncertainty if 'aleatoric_uncertainty_lower_bound' in locals() else None,
            'aleatoric': aleatoric_uncertainty if 'aleatoric_uncertainty' in locals() else None,
            'epistemic': epistemic_uncertainty,
            'aleatoric_lower_bound': aleatoric_uncertainty_lower_bound if 'aleatoric_uncertainty_lower_bound' in locals() else None,
            'aleatoric_upper_bound': aleatoric_uncertainty_upper_bound if 'aleatoric_uncertainty_upper_bound' in locals() else None,
        }