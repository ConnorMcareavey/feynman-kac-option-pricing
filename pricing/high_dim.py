"""High dimensional integration: tensor product grids against Monte Carlo.

The test problem is a call on the geometric average of d independent
stocks with identical parameters. Its price is a d dimensional integral
over the standard normal variables z_1, ..., z_d that drive the stocks,

    V = exp(-rT) E[ max(G(T) - K, 0) ],
    log G(T) = log S0 + (r - sigma^2 / 2) T + sigma sqrt(T) (z_1 + ... + z_d) / d,

and it also has a closed form (black_scholes.geometric_basket_call_price),
so both methods can be checked exactly in every dimension.
"""

import numpy as np
from scipy.special import gamma as gamma_function


def ball_volume_exact(d):
    """Volume of the unit ball in d dimensions, pi^(d/2) / Gamma(d/2 + 1)."""
    return np.pi ** (d / 2) / gamma_function(d / 2 + 1)


def ball_volume_mc(d, n_points, rng):
    """Hit or miss Monte Carlo estimate of the unit ball volume (as in exercise R4.2).

    Points are uniform in the cube [-1, 1]^d, which has volume 2^d. If a
    fraction p of them land inside the ball, the volume estimate is 2^d p,
    with standard error 2^d sqrt(p (1 - p) / n_points).
    """
    points = rng.uniform(-1.0, 1.0, size=(n_points, d))
    inside = np.sum(points**2, axis=1) <= 1.0
    p = np.mean(inside)
    volume = 2.0**d * p
    std_error = 2.0**d * np.sqrt(p * (1 - p) / n_points)
    return volume, std_error


def geometric_basket_grid(S0, K, r, sigma, T, d, n_per_dim, z_max=6.0):
    """Price the geometric basket call with a tensor product trapezoid rule.

    Each z_i is put on the same grid of n_per_dim points in [-z_max, z_max],
    so the full grid has n_per_dim ** d points. The weight of a grid point is
    the product of the one dimensional trapezoid weights times the standard
    normal densities. The arrays are built up one dimension at a time by
    broadcasting, which gives an array with d axes of length n_per_dim.
    """
    z = np.linspace(-z_max, z_max, n_per_dim)
    h = z[1] - z[0]
    w = np.full(n_per_dim, h)
    w[0] = h / 2
    w[-1] = h / 2
    w = w * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

    # z_sum holds z_1 + ... + z_d at every grid point and w_prod holds the weights.
    z_sum = z.copy()
    w_prod = w.copy()
    for i in range(1, d):
        z_sum = z_sum[..., np.newaxis] + z
        w_prod = w_prod[..., np.newaxis] * w

    log_G = np.log(S0) + (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z_sum / d
    payoff = np.maximum(np.exp(log_G) - K, 0.0)
    return np.exp(-r * T) * np.sum(payoff * w_prod)


def geometric_basket_mc(S0, K, r, sigma, T, d, n_paths, rng):
    """Plain Monte Carlo price of the geometric basket call and its standard error."""
    z = rng.standard_normal((n_paths, d))
    log_G = np.log(S0) + (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.mean(z, axis=1)
    discounted = np.exp(-r * T) * np.maximum(np.exp(log_G) - K, 0.0)
    return np.mean(discounted), np.std(discounted, ddof=1) / np.sqrt(n_paths)
