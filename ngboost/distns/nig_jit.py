import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import digamma, polygamma, gammaln

# Fast digamma and trigamma via JAX special functions
def trigamma(x):
    return polygamma(1, x)

@jit
def score_single(y, mu, lam, alpha, beta, evid_strength, kl_strength):
    eps = 1e-8
    mu0, lam0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0

    lami = jnp.maximum(lam, eps)
    alphai = jnp.maximum(alpha, 1.0 + eps)
    betai = jnp.maximum(beta, eps)

    nu = 2.0 * alphai
    two_b = 2.0 * betai
    Om = two_b * (1.0 + lami)
    resid = y - mu
    r2 = resid * resid
    term = lami * r2 + Om
    inv_l = 1.0 / lami
    inv_O = 1.0 / Om

    # NLL gradients
    d_mu = lami * (nu + 1.0) * (-resid) / term
    d_lam = -0.5 * inv_l - alphai * two_b * inv_O + (alphai + 0.5) * r2 / term
    d_alpha = -jnp.log(Om) + jnp.log(term) + digamma(alphai) - digamma(alphai + 0.5)
    d_beta = -alphai / betai + (alphai + 0.5) * (two_b * (1.0 + lami) / two_b) / term

    # Evidential reg
    sgn = jnp.where(resid >= 0.0, 1.0, -1.0)
    ev_mu = -sgn * (2.0 * alphai + lami)
    ev_lam = jnp.abs(resid)
    ev_alpha = 2.0 * jnp.abs(resid)

    # KL reg
    gk_mu = alpha0 * lami * (mu - mu0) / betai
    gk_lam = -0.5 * inv_l + alpha0 * (r2 / (2.0 * betai) + 1.0 / lam0)
    gk_alpha = (alphai - alpha0) * trigamma(alphai) + (beta0 / betai - 1.0)
    gk_beta = alpha0 * (1.0 / betai - beta0 / (betai * betai)) - alphai / (betai * betai) - alpha0 * lami * r2 / (2.0 * betai * betai)

    # Combine
    d_mu += evid_strength * ev_mu + kl_strength * gk_mu
    d_lam += evid_strength * ev_lam + kl_strength * gk_lam
    d_alpha += evid_strength * ev_alpha + kl_strength * gk_alpha
    d_beta += kl_strength * gk_beta

    # Chain rule back to raw
    return jnp.stack([d_mu, d_lam * lami, d_alpha * (alphai - 1.0), d_beta * betai])

@jit
def d_score_jax(Y, mu, lam, alpha, beta, evid_strength, kl_strength):
    # Vectorize over batch
    return vmap(score_single, in_axes=(0,0,0,0,0,None,None))(Y, mu, lam, alpha, beta, evid_strength, kl_strength)

@jit
def full_score_jax_single(y, mu, lam, alpha, evid_s, kl_s):
    eps = 1e-8
    mu0, lam0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0

    lami = jnp.maximum(lam, eps)
    alphai = jnp.maximum(alpha, 1.0 + eps)
    nu = 2.0 * alphai
    Om = 2.0 * (1.0 + lami)
    resid = y - mu
    term = lami * resid * resid + Om

    # NLL
    part1 = 0.5 * (jnp.log(jnp.pi) - jnp.log(lami))
    part2 = -alphai * jnp.log(Om)
    part3 = (alphai + 0.5) * jnp.log(term)
    part4 = gammaln(alphai) - gammaln(alphai + 0.5)
    nll = part1 + part2 + part3 + part4

    ev = jnp.abs(resid) * (2.0 * alphai + lami)

    # KL (beta fixed=1)
    betai = 1.0
    t1 = 0.5 * jnp.log(lam0 / lami)
    t2 = alpha0 * jnp.log(betai / beta0)
    t3 = -gammaln(alphai) + gammaln(alpha0)
    t4 = (alphai - alpha0) * digamma(alphai)
    t5 = alpha0 * lami * (mu - mu0)**2 / (2.0 * betai)
    t6 = alpha0 * (lami / lam0 - 1.0)
    t7 = alphai * (beta0 / betai - 1.0)
    kl = t1 + t2 + t3 + t4 + t5 + t6 + t7

    return nll, ev, kl

@jit
def full_score_jax(Y, mu, lam, alpha, evid_s, kl_s):
    # First pass
    nlls, evs, kls = vmap(full_score_jax_single, in_axes=(0,0,0,0,None,None))(Y, mu, lam, alpha, evid_s, kl_s)
    mean_ev = jnp.mean(evs)
    mean_kl = jnp.mean(kls)
    return nlls + evid_s * mean_ev + kl_s * mean_kl

@jit
def compute_diag_fim_jax(grads):
    return grads**2 + 1e-5
