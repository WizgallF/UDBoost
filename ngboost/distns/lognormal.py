import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf

# Normal distribution helpers
_inv_sqrt_2pi = 1.0 / jnp.sqrt(2.0 * jnp.pi)
_inv_sqrt2 = 1.0 / jnp.sqrt(2.0)

def norm_pdf(x):
    return _inv_sqrt_2pi * jnp.exp(-0.5 * x**2)

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x * _inv_sqrt2))

class LogNormalLogScoreCensored:
    def __init__(self, loc, scale, eps=1e-5):
        self.loc = loc
        self.scale = scale
        self.eps = eps

    @jit
    def score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = jnp.log(T)
        Z = (lT - self.loc) / self.scale

        # log CDF and log PDF of LogNormal
        log_cdf = jnp.log(norm_cdf(Z) + self.eps)
        log_pdf = jnp.log(norm_pdf(Z)) - jnp.log(self.scale) - lT

        cens = (1 - E) * log_cdf
        uncens = E * log_pdf
        return -(cens + uncens)

    @jit
    def d_score(self, Y):
        E = Y["Event"][:, None]
        T = Y["Time"]
        lT = jnp.log(T)
        Z = (lT - self.loc) / self.scale

        # gradients for uncensored
        d_uncens_loc = (self.loc - lT) / (self.scale**2)
        d_uncens_scale = 1.0 - ((self.loc - lT)**2) / (self.scale**2)
        D_uncens = jnp.stack([d_uncens_loc, d_uncens_scale], axis=1)

        # gradients for censored
        cdf = norm_cdf(Z)
        pdf = norm_pdf(Z)
        denom = cdf + self.eps
        d_cens_loc = -pdf / (denom * self.scale)
        d_cens_scale = -(Z * pdf) / denom
        D_cens = jnp.stack([d_cens_loc, d_cens_scale], axis=1)

        return (1 - E) * D_cens + E * D_uncens

    @jit
    def metric(self):
        n = self.loc.shape[0]
        FI = jnp.zeros((n, 2, 2))
        FI = FI.at[:, 0, 0].set(1.0 / (self.scale**2) + self.eps)
        FI = FI.at[:, 1, 1].set(2.0)
        return FI

class LogNormalCRPScoreCensored:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    @jit
    def score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = jnp.log(T)
        Z = (lT - self.loc) / self.scale

        # CRPS for uncensored
        crps_uncens = self.scale * (
            Z * (2 * norm_cdf(Z) - 1)
            + 2 * norm_pdf(Z)
            - 1 / jnp.sqrt(jnp.pi)
        )
        # CRPS for censored
        cdfZ = norm_cdf(Z)
        pdfZ = norm_pdf(Z)
        crps_cens = self.scale * (
            Z * cdfZ**2
            + 2 * cdfZ * pdfZ
            - norm_cdf(jnp.sqrt(2.0) * Z) / jnp.sqrt(jnp.pi)
        )
        return (1 - E) * crps_cens + E * crps_uncens

    @jit
    def d_score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = jnp.log(T)
        Z = (lT - self.loc) / self.scale

        cdfZ = norm_cdf(Z)
        pdfZ = norm_pdf(Z)
        sqrt2 = jnp.sqrt(2.0)

        # derivative w.r.t loc
        d0 = E * -(2 * cdfZ - 1) + (1 - E) * -(
            cdfZ**2
            + 2 * Z * cdfZ * pdfZ
            + 2 * pdfZ**2
            - 2 * cdfZ * pdfZ**2
            - jnp.sqrt(2.0 / jnp.pi) * norm_pdf(sqrt2 * Z)
        )
        # derivative w.r.t scale
        score_vals = self.score(Y)
        d1 = score_vals + (lT - self.loc) * d0

        return jnp.stack([d0, d1], axis=1)

    @jit
    def metric(self):
        n = self.loc.shape[0]
        I = jnp.zeros((n, 2, 2))
        I = I.at[:, 0, 0].set(2.0)
        I = I.at[:, 1, 1].set(self.scale**2)
        I = I / (2.0 * jnp.sqrt(jnp.pi))
        return I

class LogNormal:
    """
    Implements the log-normal distribution for NGBoost.
    """
    n_params = 2
    censored_scores = [LogNormalLogScoreCensored, LogNormalCRPScoreCensored]

    def __init__(self, params):
        self.loc = params[0]
        self.scale = jnp.exp(params[1])

    def score(self, Y):
        scorer = LogNormalLogScoreCensored(self.loc, self.scale)
        return scorer.score(Y)

    def d_score(self, Y):
        scorer = LogNormalLogScoreCensored(self.loc, self.scale)
        return scorer.d_score(Y)

    def full_score(self, Y, score_type='log'):
        if score_type == 'log':
            return LogNormalLogScoreCensored(self.loc, self.scale).score(Y)
        else:
            return LogNormalCRPScoreCensored(self.loc, self.scale).score(Y)

    def metric(self, score_type='log'):
        if score_type == 'log':
            return LogNormalLogScoreCensored(self.loc, self.scale).metric()
        else:
            return LogNormalCRPScoreCensored(self.loc, self.scale).metric()

    @staticmethod
    def fit(Y):
        # Method of moments on log-scale
        lY = jnp.log(Y)
        mu = jnp.mean(lY)
        sigma = jnp.std(lY)
        return jnp.array([mu, jnp.log(sigma)])