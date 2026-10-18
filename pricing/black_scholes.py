"""Closed form Black-Scholes prices and Greeks.

These formulas are the ground truth that every numerical method in the
project is measured against. All inputs can be numpy arrays.

Notation: S is the stock price, K the strike, r the risk free rate,
sigma the volatility and T the time to expiry (in years).
"""

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, r, sigma, T):
    """Return the usual d1 and d2 terms that appear in the Black-Scholes formulas."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, r, sigma, T):
    """European call price, S N(d1) - K exp(-rT) N(d2)."""
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def call_delta(S, K, r, sigma, T):
    """Delta of a European call, dV/dS = N(d1)."""
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1)


def call_gamma(S, K, r, sigma, T):
    """Gamma of a European call, d^2V/dS^2 = phi(d1) / (S sigma sqrt(T))."""
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def digital_price(S, K, r, sigma, T):
    """Cash-or-nothing digital call paying 1 if S(T) > K, price exp(-rT) N(d2)."""
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return np.exp(-r * T) * norm.cdf(d2)


def digital_delta(S, K, r, sigma, T):
    """Delta of the digital call, exp(-rT) phi(d2) / (S sigma sqrt(T))."""
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))


def digital_gamma(S, K, r, sigma, T):
    """Gamma of the digital call, -exp(-rT) phi(d2) d1 / (S^2 sigma^2 T)."""
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return -np.exp(-r * T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)


def geometric_basket_call_price(S0, K, r, sigma, T, corr):
    """Call on the geometric average of d correlated lognormal stocks.

    S0 and sigma are length d arrays and corr is the d by d correlation
    matrix of the Brownian motions. Because the log of the geometric average
    is the average of the logs, it is normally distributed:

        log G(T) ~ Normal(mu, v),
        mu = mean of [log S0_i + (r - sigma_i^2 / 2) T],
        v  = T * sigma^T corr sigma / d^2.

    The price exp(-rT) E[max(G(T) - K, 0)] then follows from the lognormal
    expectation formula, exactly as in the derivation of Black-Scholes.
    """
    S0 = np.asarray(S0, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d = len(S0)
    mu = np.mean(np.log(S0) + (r - 0.5 * sigma**2) * T)
    v = T * (sigma @ corr @ sigma) / d**2
    d2 = (mu - np.log(K)) / np.sqrt(v)
    d1 = d2 + np.sqrt(v)
    return np.exp(-r * T) * (np.exp(mu + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))
