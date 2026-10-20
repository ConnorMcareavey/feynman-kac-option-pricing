"""Monte Carlo simulation of the Black-Scholes stock price model.

Under the risk neutral measure the stock follows geometric Brownian motion,

    dS = r S dt + sigma S dB,

which has the exact solution

    S(t) = S(0) exp((r - sigma^2 / 2) t + sigma B(t)).

We can therefore sample S(T) exactly, or step the SDE forward with the
Euler-Maruyama method from Section 4.9 of the course notes,

    S_{n+1} = S_n + r S_n dt + sigma S_n dB_n,   dB_n = z_n sqrt(dt).
"""

import numpy as np


def sample_exact_terminal(S0, r, sigma, T, n_paths, rng):
    """Draw n_paths exact samples of S(T) in a single step."""
    z = rng.standard_normal(n_paths)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def simulate_terminal_prices(S0, r, sigma, T, n_steps, n_paths, rng):
    """Simulate S(T) along the same Brownian paths in two ways.

    Both the exact solution and the Euler-Maruyama approximation are driven
    by the same increments dB_n, so the only difference between them is the
    time stepping error. Only the current values are stored, so memory use
    does not grow with n_steps.

    Returns (S_exact, S_em), two arrays of length n_paths.
    """
    dt = T / n_steps
    S_exact = np.full(n_paths, float(S0))
    S_em = np.full(n_paths, float(S0))

    for n in range(n_steps):
        dB = np.sqrt(dt) * rng.standard_normal(n_paths)
        # Euler-Maruyama step
        S_em = S_em + r * S_em * dt + sigma * S_em * dB
        # Exact step: the solution formula applied over one interval of length dt
        S_exact = S_exact * np.exp((r - 0.5 * sigma**2) * dt + sigma * dB)

    return S_exact, S_em


def simulate_paths(S0, r, sigma, T, n_steps, n_paths, rng):
    """Like simulate_terminal_prices, but keep every time level (for plotting).

    Returns the time grid t and two arrays of shape (n_paths, n_steps + 1).
    """
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    S_exact = np.zeros((n_paths, n_steps + 1))
    S_em = np.zeros((n_paths, n_steps + 1))
    S_exact[:, 0] = S0
    S_em[:, 0] = S0

    for n in range(n_steps):
        dB = np.sqrt(dt) * rng.standard_normal(n_paths)
        S_em[:, n + 1] = S_em[:, n] + r * S_em[:, n] * dt + sigma * S_em[:, n] * dB
        S_exact[:, n + 1] = S_exact[:, n] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dB)

    return t, S_exact, S_em


def mc_estimate(samples):
    """Monte Carlo estimate and its standard error.

    The estimate is the sample mean, and the standard error is
    std / sqrt(M), as in Section 4.10 of the course notes.
    """
    M = len(samples)
    return np.mean(samples), np.std(samples, ddof=1) / np.sqrt(M)
