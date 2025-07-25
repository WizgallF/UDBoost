"""The NGBoost library"""

# pylint: disable=line-too-long,too-many-instance-attributes,too-many-arguments
# pylint: disable=unused-argument,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=unused-variable,invalid-unary-operand-type,attribute-defined-outside-init
# pylint: disable=redundant-keyword-arg,protected-access,unnecessary-lambda-assignment
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_array, check_random_state, check_X_y

from ngboost.distns import MultivariateNormal, Normal, k_categorical
from ngboost.learners import default_tree_learner
from ngboost.manifold import manifold
from ngboost.scores import LogScore


class NGBoost:
    """
    Constructor for all NGBoost models.

    This class implements the methods that are common to all NGBoost models.
    Unless you are implementing a new kind of regression (e.g. interval-censored, etc.),
    you should probably use one of NGBRegressor, NGBClassifier, or NGBSurvival.

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
        random_state      : seed for reproducibility.
                            See https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        validation_fraction: Proportion of training data to set aside as validation data for early stopping.
        early_stopping_rounds: The number of consecutive boosting iterations during which the
                                    loss has to increase before the algorithm stops early.
                                    Set to None to disable early stopping and validation.
                                    None enables running over the full data set.


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
    ):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.Manifold = manifold(Score, Dist)
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.n_features = None
        self.base_models = []
        self.scalings = []
        self.col_idxs = []
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.best_val_loss_itr = None
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds

        if hasattr(self.Dist, "multi_output"):
            self.multi_output = self.Dist.multi_output
        else:
            self.multi_output = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["Manifold"]
        state["Dist_name"] = self.Dist.__name__
        if self.Dist.__name__ == "Categorical":
            del state["Dist"]
            state["K"] = self.Dist.n_params + 1
        elif self.Dist.__name__ == "MVN":
            del state["Dist"]
            state["K"] = (-3 + (9 + 8 * (self.Dist.n_params)) ** 0.5) / 2
        return state

    def __setstate__(self, state_dict):
        # Recreate the object which could not be pickled
        name_to_dist_dict = {"Categorical": k_categorical, "MVN": MultivariateNormal}
        if "K" in state_dict.keys():
            state_dict["Dist"] = name_to_dist_dict[state_dict["Dist_name"]](
                state_dict["K"]
            )
        state_dict["Manifold"] = manifold(state_dict["Score"], state_dict["Dist"])
        self.__dict__ = state_dict

    def fit_init_params_to_marginal(self, Y, sample_weight=None, iters=1000):
        self.init_params = self.Manifold.fit(
            Y
        )  # would be best to put sample weights here too

    def pred_param(self, X, max_iter=None):
        m, n = X.shape
        params = np.ones((m, self.Manifold.n_params)) * self.init_params
        for i, (models, s, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs)
        ):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X[:, col_idx]) for model in models]).T
            params -= self.learning_rate * resids * s
        return params

    def sample(self, X, Y, sample_weight, params):
        idxs = np.arange(len(Y))
        col_idx = np.arange(X.shape[1])

        if self.minibatch_frac != 1.0:
            sample_size = int(self.minibatch_frac * len(Y))
            idxs = self.random_state.choice(
                np.arange(len(Y)), sample_size, replace=False
            )

        if self.col_sample != 1.0:
            if self.col_sample > 0.0:
                col_size = max(1, int(self.col_sample * X.shape[1]))
            else:
                col_size = 0
            col_idx = self.random_state.choice(
                np.arange(X.shape[1]), col_size, replace=False
            )

        weight_batch = None if sample_weight is None else sample_weight[idxs]

        return (
            idxs,
            col_idx,
            X[idxs, :][:, col_idx],
            Y[idxs],
            weight_batch,
            params[idxs, :],
        )

    def fit_base(self, X, grads, sample_weight=None):
        if sample_weight is None:
            models = [clone(self.Base).fit(X, g) for g in grads.T]
        else:
            models = [
                clone(self.Base).fit(X, g, sample_weight=sample_weight) for g in grads.T
            ]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    # pylint: disable=too-many-positional-arguments
    def line_search(self, resids, start, Y, sample_weight=None, scale_init=1):
        D_init = self.Manifold(start.T)
        loss_init = D_init.total_score(Y, sample_weight)
        scale = scale_init

        # first scale up
        while True:
            scaled_resids = resids * scale
            D = self.Manifold((start - scaled_resids).T)
            loss = D.total_score(Y, sample_weight)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isfinite(loss) or loss > loss_init or scale > 256:
                break
            scale = scale * 2

        # then scale down
        while True:
            scaled_resids = resids * scale
            D = self.Manifold((start - scaled_resids).T)
            loss = D.total_score(Y, sample_weight)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if norm < self.tol:
                break
            if np.isfinite(loss) and loss < loss_init:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    # pylint: disable=too-many-positional-arguments
    def fit(
        self,
        X,
        Y,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):
        """
        Fits an NGBoost model to the data

        Parameters:
            X                     : DataFrame object or List or
                                    numpy array of predictors (n x p) in Numeric format
            Y                     : DataFrame object or List or numpy array of outcomes (n)
                                    in numeric format. Should be floats for regression and
                                    integers from 0 to K-1 for K-class classification
            X_val                 : DataFrame object or List or
                                    numpy array of validation-set predictors in numeric format
            Y_val                 : DataFrame object or List or
                                    numpy array of validation-set outcomes in numeric format
            sample_weight         : how much to weigh each example in the training set.
                                    numpy array of size (n) (defaults to None)
            val_sample_weight     : how much to weigh each example in the validation set.
                                    (defaults to None)
            train_loss_monitor    : a custom score or set of scores to track on the training set
                                    during training. Defaults to the score defined in the NGBoost
                                    constructor
            val_loss_monitor      : a custom score or set of scores to track on the validation set
                                    during training. Defaults to the score defined in the NGBoost
                                    constructor
            early_stopping_rounds : the number of consecutive boosting iterations during which
                                    the loss has to increase before the algorithm stops early.

        Output:
            A fit NGBRegressor object
        """

        self.base_models = []
        self.scalings = []
        self.col_idxs = []

        return self.partial_fit(
            X,
            Y,
            X_val=X_val,
            Y_val=Y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )

    # pylint: disable=too-many-positional-arguments
    def partial_fit(
        self,
        X,
        Y,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):
        """
        Fits an NGBoost model to the data appending base models to the existing ones.

        NOTE: This method is not yet fully tested and may not work as expected, for example,
        the first call to partial_fit will be the most signifcant and later calls will just
        retune the model to newer data at the cost of making it more expensive. Use with caution.

        Parameters:
            X                     : DataFrame object or List or
                                    numpy array of predictors (n x p) in Numeric format
            Y                     : DataFrame object or List or numpy array of outcomes (n)
                                    in numeric format. Should be floats for regression and
                                    integers from 0 to K-1 for K-class classification
            X_val                 : DataFrame object or List or
                                    numpy array of validation-set predictors in numeric format
            Y_val                 : DataFrame object or List or
                                    numpy array of validation-set outcomes in numeric format
            sample_weight         : how much to weigh each example in the training set.
                                    numpy array of size (n) (defaults to None)
            val_sample_weight     : how much to weigh each example in the validation set.
                                    (defaults to None)
            train_loss_monitor    : a custom score or set of scores to track on the training set
                                    during training. Defaults to the score defined in the NGBoost
                                    constructor
            val_loss_monitor      : a custom score or set of scores to track on the validation set
                                    during training. Defaults to the score defined in the NGBoost
                                    constructor
            early_stopping_rounds : the number of consecutive boosting iterations during which
                                    the loss has to increase before the algorithm stops early.

        Output:
            A fit NGBRegressor object
        """

        if len(self.base_models) != len(self.scalings) or len(self.base_models) != len(
            self.col_idxs
        ):
            raise RuntimeError(
                "Base models, scalings, and col_idxs are not the same length"
            )

        # if early stopping is specified, split X,Y and sample weights (if given) into training and validation sets
        # This will overwrite any X_val and Y_val values passed by the user directly.
        if self.early_stopping_rounds is not None:
            early_stopping_rounds = self.early_stopping_rounds
            if X_val is None and Y_val is None:
                print(
                    f"early_stopping_rounds is set to {early_stopping_rounds} but no validation set is provided creating val set with {self.validation_fraction} of the training data"
                )
                if sample_weight is None:
                    print(
                        "Creating validation set without sample weight similar to the training data"
                    )
                    X, X_val, Y, Y_val = train_test_split(
                        X,
                        Y,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )

                else:
                    print(
                        "Creating validation set with sample weight similar to the training data"
                    )
                    (
                        X,
                        X_val,
                        Y,
                        Y_val,
                        sample_weight,
                        val_sample_weight,
                    ) = train_test_split(
                        X,
                        Y,
                        sample_weight,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
            elif X_val is not None and Y_val is not None:
                if sample_weight is not None and val_sample_weight is None:
                    raise ValueError(
                        "Training data is passed with sample weights but the validation data is missing sample weights pass the appropriate val_sample_weights"
                    )
                if sample_weight is None and val_sample_weight is not None:
                    raise ValueError(
                        "sample weights mismatch between training and validation data check and pass the appropriate val_sample_weights"
                    )

                print("Using passed validation data to check for early stopping.")

            else:
                if (X_val is not None and Y_val is None) or (
                    X_val is None and Y_val is not None
                ):
                    raise ValueError(
                        "Inconsistent Validation data either X_val or Y_val is missing, check the data"
                    )

        if Y is None:
            raise ValueError("y cannot be None")

        X, Y = check_X_y(
            X,
            Y,
            accept_sparse=True,
            ensure_all_finite="allow-nan",
            multi_output=self.multi_output,
            y_numeric=True,
        )

        self.n_features = X.shape[1]

        loss_list = []
        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)

        if X_val is not None and Y_val is not None:
            X_val, Y_val = check_X_y(
                X_val,
                Y_val,
                accept_sparse=True,
                ensure_all_finite="allow-nan",
                multi_output=self.multi_output,
                y_numeric=True,
            )
            val_params = self.pred_param(X_val)
            val_loss_list = []
            best_val_loss = np.inf

        if not train_loss_monitor:
            train_loss_monitor = lambda D, Y, W: D.total_score(  # noqa: E731
                Y, sample_weight=W
            )

        if not val_loss_monitor:
            val_loss_monitor = lambda D, Y: D.total_score(  # noqa: E731
                Y, sample_weight=val_sample_weight
            )

        for itr in range(len(self.col_idxs), self.n_estimators + len(self.col_idxs)):
            _, col_idx, X_batch, Y_batch, weight_batch, P_batch = self.sample(
                X, Y, sample_weight, params
            )
            self.col_idxs.append(col_idx)

            D = self.Manifold(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch, weight_batch)]
            loss = loss_list[-1]
            grads = D.grad(Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads, weight_batch)
            scale = self.line_search(proj_grad, P_batch, Y_batch, weight_batch)

            params -= (
                self.learning_rate
                * scale
                * np.array([m.predict(X[:, col_idx]) for m in self.base_models[-1]]).T
            )

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= (
                    self.learning_rate
                    * scale
                    * np.array(
                        [m.predict(X_val[:, col_idx]) for m in self.base_models[-1]]
                    ).T
                )
                val_loss = val_loss_monitor(self.Manifold(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if val_loss < best_val_loss:
                    best_val_loss, self.best_val_loss_itr = val_loss, itr
                if (
                    early_stopping_rounds is not None
                    and len(val_loss_list) > early_stopping_rounds
                    and best_val_loss
                    < np.min(np.array(val_loss_list[-early_stopping_rounds:]))
                ):
                    if self.verbose:
                        print("== Early stopping achieved.")
                        print(
                            f"== Best iteration / VAL{self.best_val_loss_itr} (val_loss={best_val_loss:.4f})"
                        )
                    break

            if (
                self.verbose
                and int(self.verbose_eval) > 0
                and itr % int(self.verbose_eval) == 0
            ):
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(
                    f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                    f"norm={grad_norm:.4f}"
                )

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        self.evals_result = {}
        metric = self.Score.__name__.upper()
        self.evals_result["train"] = {metric: loss_list}
        if X_val is not None and Y_val is not None:
            self.evals_result["val"] = {metric: val_loss_list}

        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """
        Parameters
        ----------
        deep : Ignored. (for compatibility with sklearn)
        Returns
        ----------
        params : returns an dictionary of parameters.
        """
        params = {
            "Dist": self.Dist,
            "Score": self.Score,
            "Base": self.Base,
            "natural_gradient": self.natural_gradient,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "minibatch_frac": self.minibatch_frac,
            "col_sample": self.col_sample,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "validation_fraction": self.validation_fraction,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

        return params

    def score(self, X, Y):  # for sklearn
        return self.Manifold(self.pred_dist(X)._params).total_score(Y)

    def pred_dist(self, X, max_iter=None):
        """
        Predict the conditional distribution of Y at the points X=x

        Parameters:
            X         : DataFrame object or List or
                        numpy array of predictors (n x p) in numeric format.
            max_iter  : get the prediction at the specified number of boosting iterations

        Output:
            A NGBoost distribution object
        """

        X = check_array(X, accept_sparse=True, ensure_all_finite="allow-nan")

        params = np.asarray(self.pred_param(X, max_iter))
        dist = self.Dist(params.T)

        return dist

        
    def staged_pred_dist(self, X, max_iter=None):
        """
        Predict the conditional distribution of Y at the points X=x at multiple boosting iterations

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : largest number of boosting iterations to get the prediction for

        Output:
            A list of NGBoost distribution objects, one per boosting stage up to max_iter
        """
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s, col_idx) in enumerate(
            zip(self.base_models, self.scalings, self.col_idxs), start=1
        ):
            resids = np.array([model.predict(X[:, col_idx]) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(
                np.copy(params.T)
            )  # if the params aren't copied, param changes with stages carry over to dists
            predictions.append(dists)
            if max_iter and i == max_iter:
                break
        return predictions

    def predict(self, X, max_iter=None):
        """
        Point prediction of Y at the points X=x

        Parameters:
            X         : DataFrame object or List or numpy array of predictors (n x p)
                        in numeric Format
            max_iter  : get the prediction at the specified number of boosting iterations

        Output:
            Numpy array of the estimates of Y
        """

        X = check_array(X, accept_sparse=True, ensure_all_finite="allow-nan")

        return self.pred_dist(X, max_iter=max_iter).predict()

    def staged_predict(self, X, max_iter=None):
        """
        Point prediction of Y at the points X=x at multiple boosting iterations

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : largest number of boosting iterations to get the prediction for

        Output:
            A list of numpy arrays of the estimates of Y, one per boosting stage up to max_iter
        """
        return [dist.predict() for dist in self.staged_pred_dist(X, max_iter=max_iter)]

    @property
    def feature_importances_(self):
        """Return the feature importances for all parameters in the distribution
            (the higher, the more important the feature).

        Returns:
            feature_importances_ : array, shape = [n_params, n_features]
                The summation along second axis of this array is an array of ones,
                unless all trees are single node trees consisting of only the root
                node, in which case it will be an array of zeros.
        """
        # Check whether the model is fitted
        if not self.base_models:
            return None
        # Check whether the base model is DecisionTreeRegressor
        if not isinstance(self.base_models[0][0], DecisionTreeRegressor):
            return None
        # Reshape the base_models
        params_trees = zip(*self.base_models)

        # Get the feature_importances_ for all the params and all the trees
        all_params_importances = [
            [
                self._get_feature_importance(tree, tree_index)
                for tree_index, tree in enumerate(trees)
            ]
            for trees in params_trees
        ]

        if not all_params_importances:
            return np.zeros(
                (
                    len(self.base_models[0]),
                    self.base_models[0][0].n_features_,
                ),
                dtype=np.float64,
            )

        # Weighted average of importance by tree scaling factors
        all_params_importances = np.average(
            all_params_importances, axis=1, weights=self.scalings
        )
        return all_params_importances / np.sum(
            all_params_importances, axis=1, keepdims=True
        )

    def _get_feature_importance(self, tree, tree_index):
        tree_feature_importance = getattr(tree, "feature_importances_")
        total_feature_importance = np.zeros(self.n_features)
        total_feature_importance[self.col_idxs[tree_index]] = tree_feature_importance
        return total_feature_importance
