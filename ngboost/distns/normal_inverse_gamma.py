import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import gammaln, digamma, polygamma, psi
import scipy.stats as st

from .nig_jit import (
    d_score_numba,
    full_score_numba,
    compute_diag_fim,
    rbf_kernel_and_grad_numba,
    leaf_volume_density_vec,
)
import line_profiler

def softplus(x):
    """
    Numerically stable implementation of the softplus function.

    Parameters:
        x (float or np.ndarray): Input value(s).

    Returns:
        np.ndarray: Softplus-transformed values.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def positive(x, eps=1e-3):
    """
    Maps input to positive values using softplus, with a minimum epsilon shift.

    Parameters:
        x (float or np.ndarray): Input value(s).
        eps (float): Minimum value shift to ensure strict positivity.

    Returns:
        np.ndarray: Positive-transformed values.
    """
    return np.log1p(np.exp(x)) + eps

# Custom score for Normal-Inverse-Gamma.
class NIGLogScore(LogScore):
    """
    Custom scoring function for the Normal-Inverse-Gamma (NIG) distribution in NGBoost.

    Attributes:
        evid_strength (float): Coefficient for the evidential regularization term.
        kl_strength (float): Coefficient for KL divergence regularization.
    """
    
    lower_bound = None
    upper_bound = None
    evid_strength = 0.1
    kl_strength = 0.1

    @classmethod
    def set_params(cls, evid_strength=None, kl_strength=None):
        """
        Sets the regularization strengths for the NIG log score.

        Parameters:
            evid_strength (float): Evidential regularization strength.
            kl_strength (float): KL divergence regularization strength.
        """
        if evid_strength is not None:
            cls.evid_strength = evid_strength
        if kl_strength is not None:
            cls.kl_strength = kl_strength
    
    @classmethod
    def set_bounds(cls, lower, upper):
        """
        Set lower and upper bounds for predictions.

        Parameters:
            lower (float): Lower bound.
            upper (float): Upper bound.
        """
        cls.lower_bound = lower
        cls.upper_bound = upper

    def kl_divergence_nig(
            self, 
            mu, 
            lam, 
            alpha, 
            beta,
            mu0=0.0, 
            lam0=1.0, 
            alpha0=2.0, 
            beta0=1.0,
            eps=1e-8):
        """
        Compute the KL divergence between two NIG distributions.

        Parameters:
            mu, lam, alpha, beta: Current parameters.
            mu0, lam0, alpha0, beta0: Prior parameters.
            eps (float): Numerical stability threshold.

        Returns:
            float: Average KL divergence.
        """

        # Clip to avoid division by zero or log of zero
        lam = np.clip(lam, eps, 1e6)
        alpha = np.clip(alpha, 1.0 + eps, 1e6)
        beta = np.clip(beta, eps, 1e6)

        # Individual terms
        term1 = 0.5 * np.log(lam0 / lam)
        term2 = alpha0 * np.log(beta / beta0)
        term3 = -gammaln(alpha) + gammaln(alpha0)
        term4 = (alpha - alpha0) * digamma(alpha)
        term5 = alpha0 * (lam * (mu - mu0) ** 2 / (2 * beta))
        term6 = alpha0 * (lam / lam0 - 1)
        term7 = alpha * (beta0 / beta - 1)

        kl = term1 + term2 + term3 + term4 + term5 + term6 + term7
        return np.mean(kl)

    def evidential_regularizer(self, Y, mu, lam, alpha):
        """
        Computes an evidential penalty encouraging predictive uncertainty to match residual error.

        Parameters:
            Y (np.ndarray): Target values.
            mu, lam, alpha (np.ndarray): Model parameters.

        Returns:
            float: Average evidential penalty.
        """
        error = np.abs(Y - mu)
        penalty = error * (2 * alpha + lam)
        return np.mean(penalty)
    
    @line_profiler.profile
    def score(self, Y, params=None):
        """
        Computes the custom NIG loss for each data point using JIT-compiled scoring.

        Parameters:
            Y (np.ndarray): Target values.
            params (list or None): Optional list of model parameters.

        Returns:
            np.ndarray: Per-sample NIG loss values.
        """
        # 1) unpack parameters into arrays
        self._last_Y = Y
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
            mu, lam, alpha, beta = mu.ravel(), lam.ravel(), alpha.ravel(), beta.ravel()
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T

        # 2) (optional) pre-clip to avoid NaNs
        eps = 1e-8
        lam   = np.clip(lam,   eps, None)
        alpha = np.clip(alpha, 1.0 + eps, None)
        beta  = np.clip(beta,  eps, None)

        evid_strength = self.__class__.evid_strength
        kl_strength = self.__class__.kl_strength
        # 3) call the Numba ufunc — this returns an (n,) array of per-sample losses
        per_sample_losses = full_score_numba(
            Y.astype(np.float64),
            mu.astype(np.float64),
            lam.astype(np.float64),
            alpha.astype(np.float64),
            evid_strength,
            kl_strength
        )

        # 4) return the vector directly
        return per_sample_losses
    


    @line_profiler.profile
    def d_score(self, Y, params=None):
        """
        Computes the gradient of the NIG loss with respect to the model parameters.

        Parameters:
            Y (np.ndarray): Target values.
            params (list or None): Optional list of model parameters.

        Returns:
            np.ndarray: Per-sample gradient vectors.
        """
        # Unpack or use stored
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T
        
        evid_strength = self.__class__.evid_strength
        kl_strength = self.__class__.kl_strength
        # Stabilize
        grads = d_score_numba(Y.astype(np.float64),
                                 mu.astype(np.float64),
                                 lam.astype(np.float64),
                                 alpha.astype(np.float64),
                                 beta.astype(np.float64),
                                 evid_strength,
                                 kl_strength)
        self.current_grads = grads

        return grads
    
    @line_profiler.profile
    def metric(self, Y=None, params=None, diagonal: bool = False):
        """
        Estimates the Fisher Information Matrix (FIM) from gradients.

        Parameters:
            Y (np.ndarray): Target values.
            params (list): Model parameters.
            diagonal (bool): If True, compute only the diagonal of FIM.

        Returns:
            np.ndarray: Empirical FIM matrix or diagonals.
        """
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
            params = [mu, lam, alpha, beta]
        else:
            params = np.stack(params, axis=-1).T
        if Y is None:
            Y = self._last_Y

        grads = self.current_grads

        if diagonal:
            return compute_diag_fim(grads)
        else:
            # Full FIM
            return np.array([np.outer(g, g) + 1e-5*np.eye(g.shape[0]) for g in grads])
        

class NIGLogScoreSVGD(LogScore):
    """
    Custom scoring function for the Normal-Inverse-Gamma (NIG) distribution in NGBoost.

    Attributes:
        evid_strength (float): Coefficient for the evidential regularization term.
        kl_strength (float): Coefficient for KL divergence regularization.
    """
    
    lower_bound = None
    upper_bound = None
    evid_strength = 0.1
    kl_strength = 0.1
    length_scale = 0.1
    warmup = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_tree = None
        self.X_train = None

    def set_train_data(self, X):
        """Call this once before you .fit(), so we know X inside d_score."""
        self.X_train = X

    def set_prev_tree(self, tree):
        """Call this after each boosting iteration, passing the tree you just fit."""
        self.prev_tree = tree

    def set_counter(self, boosting_step):
        """
        Returns the number of steps in the current boosting step.
        This is used to track the number of boosting iterations.
        """
        self.boosting_step = boosting_step
        
    @classmethod
    def set_params(cls, evid_strength=None, kl_strength=None, length_scale=None, warmup=None):
        """
        Sets the regularization strengths for the NIG log score.

        Parameters:
            evid_strength (float): Evidential regularization strength.
            kl_strength (float): KL divergence regularization strength.
        """
        if evid_strength is not None:
            cls.evid_strength = evid_strength
        if kl_strength is not None:
            cls.kl_strength = kl_strength
        if length_scale is not None:
            cls.length_scale = length_scale
        if warmup is not None:
            cls.warmup = warmup

    @classmethod
    def set_bounds(cls, lower, upper):
        """
        Set lower and upper bounds for predictions.

        Parameters:
            lower (float): Lower bound.
            upper (float): Upper bound.
        """
        cls.lower_bound = lower
        cls.upper_bound = upper

    def kl_divergence_nig(
            self, 
            mu, 
            lam, 
            alpha, 
            beta,
            mu0=0.0, 
            lam0=1.0, 
            alpha0=2.0, 
            beta0=1.0,
            eps=1e-8):
        """
        Compute the KL divergence between two NIG distributions.

        Parameters:
            mu, lam, alpha, beta: Current parameters.
            mu0, lam0, alpha0, beta0: Prior parameters.
            eps (float): Numerical stability threshold.

        Returns:
            float: Average KL divergence.
        """

        # Clip to avoid division by zero or log of zero
        lam = np.clip(lam, eps, 1e6)
        alpha = np.clip(alpha, 1.0 + eps, 1e6)
        beta = np.clip(beta, eps, 1e6)

        # Individual terms
        term1 = 0.5 * np.log(lam0 / lam)
        term2 = alpha0 * np.log(beta / beta0)
        term3 = -gammaln(alpha) + gammaln(alpha0)
        term4 = (alpha - alpha0) * digamma(alpha)
        term5 = alpha0 * (lam * (mu - mu0) ** 2 / (2 * beta))
        term6 = alpha0 * (lam / lam0 - 1)
        term7 = alpha * (beta0 / beta - 1)

        kl = term1 + term2 + term3 + term4 + term5 + term6 + term7
        return np.mean(kl)

    def evidential_regularizer(self, Y, mu, lam, alpha):
        """
        Computes an evidential penalty encouraging predictive uncertainty to match residual error.

        Parameters:
            Y (np.ndarray): Target values.
            mu, lam, alpha (np.ndarray): Model parameters.

        Returns:
            float: Average evidential penalty.
        """
        error = np.abs(Y - mu)
        penalty = error * (2 * alpha + lam)
        return np.mean(penalty)
    
    @line_profiler.profile
    def score(self, Y, params=None):
        """
        Computes the custom NIG loss for each data point using JIT-compiled scoring.

        Parameters:
            Y (np.ndarray): Target values.
            params (list or None): Optional list of model parameters.

        Returns:
            np.ndarray: Per-sample NIG loss values.
        """
        # 1) unpack parameters into arrays
        self._last_Y = Y
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
            mu, lam, alpha, beta = mu.ravel(), lam.ravel(), alpha.ravel(), beta.ravel()
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T

        # 2) (optional) pre-clip to avoid NaNs
        eps = 1e-8
        lam   = np.clip(lam,   eps, None)
        alpha = np.clip(alpha, 1.0 + eps, None)
        beta  = np.clip(beta,  eps, None)

        evid_strength = self.__class__.evid_strength
        kl_strength = self.__class__.kl_strength
        # 3) call the Numba ufunc — this returns an (n,) array of per-sample losses
        per_sample_losses = full_score_numba(
            Y.astype(np.float64),
            mu.astype(np.float64),
            lam.astype(np.float64),
            alpha.astype(np.float64),
            evid_strength,
            kl_strength
        )

        # 4) return the vector directly
        return per_sample_losses
    


    @line_profiler.profile
    def d_score(self, Y, params=None):
    # 0) unpack
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T

        # 1) raw NIG gradients (size: n×4)
        raw_grads = d_score_numba(
            Y.astype(np.float64),
            mu.astype(np.float64),
            lam.astype(np.float64),
            alpha.astype(np.float64),
            beta.astype(np.float64),
            self.evid_strength,
            self.kl_strength
        )

        ## 2) warm-up
        if self.boosting_step < self.warmup:
            self.current_grads = raw_grads
            return raw_grads
        self.boosting_step += 1
        # 3) SVGD kernel in θ-space
        theta = np.stack([mu, lam, alpha, beta], axis=-1)  # (n,4)
        K_theta, dK_theta = rbf_kernel_and_grad_numba(theta, gamma=self.length_scale)
        n = theta.shape[0]

        ## 4) data‐space “Mahalanobis” kernel
        ##    here we use 1D X → for multi-D just replace with diff across dims
        #X      = self.X_train.ravel()                      # (n,)
        #epi    = self.pred_uncertainty()["epistemic"].ravel()  # (n,)
#
        ## broadcast to (n,n)
        #diff   = X[:, None] - X[None, :]                    # (n,n)
        #varsum = epi[:, None] + epi[None, :] + 1e-12        # (n,n)
        #K_data = np.exp(-0.5 * (diff*diff) / varsum)        # (n,n)

        # 5) combine kernels
        #K  = K_theta * K_data                               # elementwise
        #dK = dK_theta * K_data[:, :, None]                  # broadcast into last dim
        # 4) SVGD pseudo-residuals
        stein_grads = (K_theta.dot(raw_grads) + dK_theta.sum(axis=1)) / n

        
        # 5) leaf‐volume scaling *only* on raw_lam gradient:
        #if self.prev_tree is not None and self.X_train is not None:
        #    p_hat = leaf_volume_density_vec(self.prev_tree, self.X_train)  # (n,)
        #    print(f"p_hat: {p_hat.shape}, min: {np.min(p_hat)}, max: {np.max(p_hat)}")
        #    w     = 1.0/(p_hat + 1e-12) 
        #    stein_grads[:, 1] *= w
        
        #    mu      = self.mu.ravel()
        #    epi_var = self.pred_uncertainty()["epistemic"].ravel()
        #    X       = self.X_train.ravel()
        #    d2      = (X - mu)**2 / (epi_var + 1e-12)
        #    w_maha  = np.exp(-1 * d2)
        #    stein_grads[:,1] *= w_maha
        # 6) stash for metric
        self.current_grads = raw_grads
        self.current_K     = K_theta
        self.current_dK    = dK_theta

        return stein_grads

    def metric(self, Y=None, params=None, diagonal: bool = False):
        """
        Diagonal approximate Wasserstein-Newton operator (WGBoost eq 8–9)
          H_i = (1/M) ∑_j [ (g_j g_j^T) * K_ij^2  +  (dK_ij)(dK_ij)^T ]
        Returns either
          • shape (M,4) if diagonal=True, or
          • shape (M,4,4) if diagonal=False.
        """
        # 1) unpack current particles
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T


        if self.boosting_step < self.warmup:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
            if Y is None:
                Y = self._last_Y
            grads = self.current_grads

            if diagonal:
                return compute_diag_fim(grads)
            else:
                # Full FIM
                return np.array([np.outer(g, g) + 1e-5*np.eye(g.shape[0]) for g in grads])
    
        if Y is None:
            Y = self._last_Y
    
        # 2) raw SVGD-gradients from last d_score call
        grads = self.current_grads      # shape (M,4)
        M, d = grads.shape             # M = #particles, d = 4
    
        # 3) build Theta and get kernel + its grad
        Theta = np.column_stack([mu, lam, alpha, beta])  
        # K: (M,M), dK: (M,M,4)
        K, dK = rbf_kernel_and_grad_numba(Theta, self.length_scale)
    
        # 4) build the two tensor terms
        K2 = K**2
        # H1[i,k,l] = ∑_j K2[i,j] * grads[j,k] * grads[j,l]
        H1 = np.einsum('ij,jk,jl->ikl', K2, grads, grads)
        # H2[i,k,l] = ∑_j dK[i,j,k] * dK[i,j,l]
        H2 = np.einsum('ijk,ijl->ikl', dK, dK)
    
        # 5) combine, normalize, add tiny ridge
        H = (H1 + H2) / M
        ridge = 1e-6
        H += ridge * np.eye(d)[None, :, :]
    
        # 6) return diag or full
        if diagonal:
            # extract the k-th diagonal entry of each 4×4 block
            return np.diagonal(H, axis1=1, axis2=2)  # shape (M,4)
        else:
            return H  # shape (M,4,4)

class NormalInverseGamma(RegressionDistn):
    """
    Normal-Inverse-Gamma distribution implementation for NGBoost.

    Used to model both aleatoric and epistemic uncertainty.

    Attributes:
        mu (np.ndarray): Mean of the Normal component.
        lam (np.ndarray): Precision of the Normal component.
        alpha (np.ndarray): Shape parameter of Inverse Gamma.
        beta (np.ndarray): Scale parameter of Inverse Gamma.
    """
    n_params = 4  # Four parameters: μ, λ, α, β.
    scores = [NIGLogScore, NIGLogScoreSVGD]

    def __init__(self, params):
        """
        Initializes the NIG distribution from parameter array.

        Parameters:
            params (np.ndarray): Raw parameters [mu, raw_lam, raw_alpha, raw_beta].
        """
        self.mu    = params[0]
        self.lam   = np.exp(params[1])     # Avoid zero
        self.alpha = np.exp(params[2]) + 1      # Enforce α > 1
        self.beta  = np.exp(params[3])   
        self.is_EDL = True  # Avoid zero

        #print(f"Initialized NIG with params: mu={self.mu}, lam={self.lam}, alpha={self.alpha}, beta={self.beta}")
        #print(f"Mean beta: {np.mean(self.beta)}, Min: {np.min(self.beta)}, Max: {np.max(self.beta)}")

    @staticmethod
    def fit(Y):
        """
        Estimates initial parameters for NIG from observed data Y.

        Parameters:
            Y (np.ndarray): Target values.

        Returns:
            np.ndarray: Initial parameter estimates.
        """
        m = np.mean(Y)
        s = np.std(Y)
        return np.array([m, 0.0, np.log(1.0), np.log(s**2)])  

    def sample(self, m):
        """
        Samples m values from the predictive NIG distribution.

        Parameters:
            m (int): Number of samples per prediction.

        Returns:
            np.ndarray: Sampled values from the NIG distribution.
        """
        shape = self.mu.shape  # assuming vectorized μ
        # Sample sigma^2 from Inverse Gamma(α, β).
        sigma2 = 1 / np.random.gamma(self.alpha, 1 / self.beta, size=(m, *shape))
        samples = np.random.normal(self.mu, np.sqrt(sigma2 / self.lam))
        return samples

    def pred_uncertainty(self):
        """
        Computes predictive uncertainty (aleatoric + epistemic).

        Returns:
            dict: Dictionary containing:
                - "mean": predictive mean (mu),
                - "aleatoric": β / (α - 1),
                - "epistemic": β² / (λ(α - 1)²(α - 2)).
        """
        aleatoric = self.beta / (self.alpha - 1)
        epistemic = self.beta / (self.lam * (self.alpha - 1))
        return {
            "mean": self.mu, 
            "predictive": aleatoric + epistemic,
            "aleatoric": aleatoric, 
            "epistemic": epistemic
        }

    def pred_dist(self):
        """
        Returns the predictive parameters.

        Returns:
            tuple: (mu, lam, alpha, beta)
        """

        return self.mu, self.lam, self.alpha, self.beta 
    
    
    def metric(self, Y):
        """
        Computes the Fisher Information Matrix for parameter estimates.

        Parameters:
            Y (np.ndarray): Target values.

        Returns:
            np.ndarray: FIM matrix.
        """
        params = [self.mu, self.lam, self.alpha, self.beta]
        return NIGLogScore().metric(Y, params=params)
    
    @property
    def is_regression(self):
        """
        Indicates that this is a regression model.

        Returns:
            bool: Always True.
        """
        return True

    @property
    def params(self):
        """
        Returns a dictionary of NIG parameters.

        Returns:
            dict: {mu, lam, alpha, beta}
        """
        return {"mu": self.mu, "lam": self.lam, "alpha": self.alpha, "beta": self.beta}

    def mean(self):
        """
        Returns the predictive mean.

        Returns:
            np.ndarray: mu
        """
        return self.mu
    
    def var(self):
        """
        Computes total predictive variance.

        Returns:
            np.ndarray: Predictive variance (aleatoric + epistemic).
        """
        # Aleatoric variance (σ²) = β / (α - 1)
        aleatoric = self.beta / (self.alpha - 1)
        epistemic = self.beta / (self.lam * (self.alpha - 1))
        predictive = aleatoric + epistemic
        return predictive
        #return aleatoric
    
    def predict_variance(self, X):
        """
        Returns the predicted total variance per test input.

        Parameters:
            X (array-like): Input data.

        Returns:
            np.ndarray: Predictive variances.
        """
        dists = self.predict_dist(X)
        # call .var() on each to get aleatoric+epistemic
        return np.array([dist.var() for dist in dists])

    def logpdf(self, Y):
        """
        Computes the log probability density function using the implied Student-t marginal.

        Parameters:
            Y (np.ndarray): Target values.

        Returns:
            np.ndarray: Log-pdf values.
        """
        mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta

        # Degrees of freedom
        nu = 2 * alpha
        # Scale (variance of Student-t)
        var = beta / (lam * alpha)

        # Compute the log-pdf of the Student-t distribution
        coeff = gammaln((nu + 1) / 2) - gammaln(nu / 2)
        norm = -0.5 * np.log(nu * np.pi * var)
        sq_term = (Y - mu) ** 2 / (nu * var)
        log_prob = coeff + norm - 0.5 * (nu + 1) * np.log1p(sq_term)

        return log_prob
    
class NormalInverseGammaLeafOOD(NormalInverseGamma):
    """
    Same as NormalInverseGamma but at predict-time we down-weight
    the epistemic variance by 1/sqrt(leaf_volume_density) so that
    points in sparse leaves get larger uncertainty.
    """

    def __init__(self, params, *, leaf_tree=None, X_train=None, X_test=None):
        # call the base ctor (builds mu, lam, alpha, beta)
        super().__init__(params)
        # store the tree & the points we’ll predict on
        self.leaf_tree = leaf_tree
        self.X_train  = X_train
        self.X_test   = X_test
        
    def pred_uncertainty(self):
        aleatoric = self.beta / (self.alpha - 1)           # shape (n_test,)
        epistemic = self.beta / (self.lam * (self.alpha - 1))
        pred_var  = aleatoric + epistemic

        if self.leaf_tree is not None and self.X_train is not None and self.X_test is not None:
            eps = 1e-12

            # 1) leaf‐volume weight
            p_hat = leaf_volume_density_vec(self.leaf_tree, self.X_test)
            w_vol = np.log(p_hat + eps)

            # 2) “Mahalanobis” with diagonal cov = aleatoric
            #    here we assume X is 1‐D; for multi‐D you’d vectorize per‐feature
            x_train_mean = np.mean(self.X_train, axis=0)     # (d,)
            diffs = self.X_test - x_train_mean               # (n_test, d)
            # if d>1, you could sum over dims, but here we do per‐row dot diag:
            # m2[i] = sum_j (diffs[i,j]**2 / aleatoric[i])
            # which for 1‐D is just (diffs[:,0]**2 / aleatoric)
            m2 = np.einsum('ij,i->i', diffs**2, 1.0/(aleatoric + eps))

            w_maha = np.exp(-10 * m2)                       # (n_test,)

            # 3) combine
            print(f"w_vol: {w_vol}, w_maha: {w_maha}")
            w_comb = w_vol / w_maha

            # 4) inflate λ for low‐weight points
            self.lam = self.lam * (1.0/(w_comb + eps))

            # 5) recompute uncertainties
            aleatoric = self.beta / (self.alpha - 1)
            epistemic = self.beta / (self.lam * (self.alpha - 1))
            pred_var  = aleatoric + epistemic

        return {
            "mean":       self.mu,
            "aleatoric":  aleatoric,
            "epistemic":  epistemic,
            "predictive": pred_var
        }

            

    def predict_variance(self, X):
        """
        NGBoost’s sklearn‐API calls this for var estimates;
        we stash X so pred_uncertainty can see it.
        """
        self.X_query = X
        return self.pred_uncertainty()["predictive"]