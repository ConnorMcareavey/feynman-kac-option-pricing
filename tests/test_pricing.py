"""Checks that every numerical method reproduces a known answer.

Run from the top folder of the repository with:  pytest

The Monte Carlo tests use fixed seeds, so they give the same result every
time, and their tolerances are a few standard errors wide.
"""

import numpy as np

from pricing import black_scholes as bs
from pricing import pde, sde, high_dim, basket

K, r, sigma, T, S0 = 100.0, 0.05, 0.2, 1.0, 100.0


def test_call_price_known_value():
    # Standard textbook value for S = K = 100, r = 5%, sigma = 20%, T = 1.
    assert abs(bs.call_price(S0, K, r, sigma, T) - 10.450583572185565) < 1e-10


def test_geometric_basket_with_one_stock_is_black_scholes():
    price = bs.geometric_basket_call_price([S0], K, r, [sigma], T, np.eye(1))
    assert abs(price - bs.call_price(S0, K, r, sigma, T)) < 1e-12


def test_crank_nicolson_rannacher_call():
    S, V = pde.price_heat_grid("call", K, r, sigma, T, 2.0, 1600, 160, 0.5, n_rannacher=2)
    assert abs(V[800] - bs.call_price(K, K, r, sigma, T)) < 5e-4


def test_crank_nicolson_rannacher_digital():
    S, V = pde.price_heat_grid("digital", K, r, sigma, T, 2.0, 1600, 160, 0.5, n_rannacher=4)
    window = (S > 80) & (S < 125)
    error = np.max(np.abs(V[window] - bs.digital_price(S[window], K, r, sigma, T)))
    assert error < 1e-4


def test_explicit_scheme_stability_limit():
    # On 200 intervals, lambda = 0.48 is stable and lambda = 0.52 blows up.
    n_x, x_max, tau_max = 200, 2.0, 0.5 * sigma**2 * T
    dx = 2 * x_max / n_x
    errors = []
    for lam in [0.48, 0.52]:
        n_tau = int(round(tau_max / (lam * dx**2)))
        S, V = pde.price_heat_grid("call", K, r, sigma, T, x_max, n_x, n_tau, 0.0)
        errors.append(np.max(np.abs(V - bs.call_price(S, K, r, sigma, T))))
    assert errors[0] < 0.02
    assert errors[1] > 1.0


def test_grid_greeks_for_call():
    S, V = pde.price_heat_grid("call", K, r, sigma, T, 2.0, 1600, 160, 0.5, n_rannacher=2)
    S_in, delta, gamma = pde.grid_greeks(S, V)
    window = (S_in > 80) & (S_in < 125)
    assert np.max(np.abs(delta[window] - bs.call_delta(S_in[window], K, r, sigma, T))) < 1e-4
    assert np.max(np.abs(gamma[window] - bs.call_gamma(S_in[window], K, r, sigma, T))) < 1e-5


def test_explicit_scheme_in_S_variable():
    S, V = pde.price_call_explicit_S(K, r, sigma, T, 300.0, 300, 4000)
    assert abs(V[100] - bs.call_price(K, K, r, sigma, T)) < 5e-3


def test_exact_monte_carlo_within_three_standard_errors():
    rng = np.random.default_rng(1)
    S_T = sde.sample_exact_terminal(S0, r, sigma, T, 200_000, rng)
    estimate, std_error = sde.mc_estimate(np.exp(-r * T) * np.maximum(S_T - K, 0.0))
    assert abs(estimate - bs.call_price(S0, K, r, sigma, T)) < 3 * std_error


def test_euler_maruyama_bias_is_first_order():
    # Halving the time step should roughly halve the (coupled) bias.
    rng = np.random.default_rng(2)
    biases = []
    for n_steps in [8, 16]:
        S_exact, S_em = sde.simulate_terminal_prices(S0, r, sigma, T, n_steps, 400_000, rng)
        diff = np.maximum(S_em - K, 0.0) - np.maximum(S_exact - K, 0.0)
        biases.append(np.mean(diff))
    ratio = biases[0] / biases[1]
    assert 1.4 < ratio < 2.8


def test_ball_volume_monte_carlo():
    rng = np.random.default_rng(3)
    for d in range(2, 7):
        volume, std_error = high_dim.ball_volume_mc(d, 200_000, rng)
        assert abs(volume - high_dim.ball_volume_exact(d)) < 4 * std_error


def test_geometric_basket_grid_and_monte_carlo():
    d = 2
    exact = bs.geometric_basket_call_price(np.full(d, S0), K, r, np.full(d, sigma), T, np.eye(d))
    grid_price = high_dim.geometric_basket_grid(S0, K, r, sigma, T, d, 513)
    assert abs(grid_price - exact) < 1e-4
    rng = np.random.default_rng(4)
    mc_price, std_error = high_dim.geometric_basket_mc(S0, K, r, sigma, T, d, 200_000, rng)
    assert abs(mc_price - exact) < 3 * std_error


def test_basket_simulation_and_control_variate():
    rng = np.random.default_rng(5)
    d, rho = 5, 0.3
    corr = basket.equicorrelation_matrix(d, rho)
    S0s, sigmas = np.full(d, S0), np.full(d, sigma)
    S_T = basket.simulate_basket_terminal(S0s, r, sigmas, T, corr, 200_000, rng)
    arithmetic, geometric = basket.basket_payoffs(S_T, K)
    discount = np.exp(-r * T)

    geometric_exact = bs.geometric_basket_call_price(S0s, K, r, sigmas, T, corr)
    estimate, std_error = sde.mc_estimate(discount * geometric)
    assert abs(estimate - geometric_exact) < 3 * std_error

    plain, plain_se = sde.mc_estimate(discount * arithmetic)
    cv, cv_se, b = basket.control_variate_estimate(discount * arithmetic, discount * geometric, geometric_exact)
    assert cv_se < plain_se / 5
    assert abs(cv - plain) < 3 * plain_se
