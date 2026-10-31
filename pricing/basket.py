"""Monte Carlo for basket options on d correlated stocks (the optional extension).

Each stock follows its own geometric Brownian motion,

    S_i(T) = S0_i exp((r - sigma_i^2 / 2) T + sigma_i W_i(T)),

where the W_i are Brownian motions with correlation matrix C. We generate
correlated normals by the Cholesky factorisation C = L L^T: if z is a
vector of independent standard normals then L z has covariance L L^T = C.

The arithmetic basket call, with payoff max(mean of S_i(T) - K, 0), has no
closed form. The geometric basket call does, so we use it twice: to check
the simulation, and as a control variate for the arithmetic basket.
"""

import numpy as np


def equicorrelation_matrix(d, rho):
    """d by d correlation matrix with 1 on the diagonal and rho everywhere else."""
    return (1 - rho) * np.eye(d) + rho * np.ones((d, d))


def simulate_basket_terminal(S0, r, sigma, T, corr, n_paths, rng):
    """Simulate S_i(T) for all d stocks. Returns an array of shape (n_paths, d)."""
    S0 = np.asarray(S0, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d = len(S0)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_paths, d))
    # Each row of z @ L.T is L times a row of z, so it has covariance C.
    W = np.sqrt(T) * (z @ L.T)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W)


def basket_payoffs(S_T, K):
    """Arithmetic and geometric basket call payoffs for each simulated path."""
    arithmetic_mean = np.mean(S_T, axis=1)
    geometric_mean = np.exp(np.mean(np.log(S_T), axis=1))
    arithmetic = np.maximum(arithmetic_mean - K, 0.0)
    geometric = np.maximum(geometric_mean - K, 0.0)
    return arithmetic, geometric


def control_variate_estimate(Y, X, X_mean_exact):
    """Control variate estimate of E[Y] using X, whose mean we know exactly.

    We use Y_cv = Y - b (X - E[X]) with the variance minimising coefficient
    b = Cov(X, Y) / Var(X), estimated from the same samples. Since
    E[X - E[X]] = 0 the estimate is still (essentially) unbiased, and its
    variance is Var(Y) (1 - corr(X, Y)^2).

    Returns the estimate, its standard error and b.
    """
    covariance = np.cov(X, Y)
    b = covariance[0, 1] / covariance[0, 0]
    Y_cv = Y - b * (X - X_mean_exact)
    M = len(Y)
    return np.mean(Y_cv), np.std(Y_cv, ddof=1) / np.sqrt(M), b
