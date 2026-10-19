"""Finite difference solvers for the Black-Scholes PDE.

The main solver works on the heat equation u_tau = u_xx, which the
Black-Scholes PDE turns into after the change of variables

    x = ln(S / K),   tau = sigma^2 (T - t) / 2,   V = K exp(alpha x + beta tau) u,

with k = 2 r / sigma^2, alpha = -(k - 1) / 2 and beta = -(k + 1)^2 / 4.
The derivation is written out in notebooks/01_pde.ipynb.

For comparison there is also a small explicit solver that works directly
in the original stock price variable S.
"""

import numpy as np
from scipy.linalg import solve_banded


def heat_theta_scheme(u0, left, right, lam, theta, n_rannacher=0):
    """Solve u_tau = u_xx on a uniform grid with the theta scheme.

    u0 is the initial condition on the whole grid, including both boundary
    nodes. left and right are the boundary values at every time level, so
    each has length n_steps + 1. lam = dtau / dx^2 is the mesh ratio.

    theta = 0 gives explicit Euler, theta = 1 implicit Euler and
    theta = 0.5 Crank-Nicolson. The first n_rannacher steps are taken with
    theta = 1 (Rannacher smoothing) whatever theta is.

    Returns u on the whole grid at the final time.
    """
    n_steps = len(left) - 1
    m = len(u0) - 2  # number of interior nodes
    u = u0.copy()

    for n in range(n_steps):
        th = 1.0 if n < n_rannacher else theta

        # Right hand side from the known time level n:
        # (1 - 2(1 - theta) lam) u_j + (1 - theta) lam (u_{j-1} + u_{j+1})
        rhs = (1 - 2 * (1 - th) * lam) * u[1:-1] + (1 - th) * lam * (u[:-2] + u[2:])

        u_new = np.empty_like(u)
        u_new[0] = left[n + 1]
        u_new[-1] = right[n + 1]

        if th == 0:
            # Explicit Euler: nothing to solve.
            u_new[1:-1] = rhs
        else:
            # Implicit part: (1 + 2 theta lam) u_j - theta lam (u_{j-1} + u_{j+1}) = rhs.
            # The known boundary values at level n + 1 move to the right hand side.
            rhs[0] += th * lam * u_new[0]
            rhs[-1] += th * lam * u_new[-1]

            # Tridiagonal matrix stored by diagonals for scipy's banded solver:
            # row 0 is the super diagonal, row 1 the main diagonal, row 2 the sub diagonal.
            ab = np.zeros((3, m))
            ab[0, 1:] = -th * lam
            ab[1, :] = 1 + 2 * th * lam
            ab[2, :-1] = -th * lam
            u_new[1:-1] = solve_banded((1, 1), ab, rhs)

        u = u_new

    return u


def payoff(S, K, option):
    """Payoff at expiry for a European "call" or a cash-or-nothing "digital" call.

    The digital pays 1 if S > K and 0 if S < K. At a grid node sitting exactly
    on the strike we use the average of the two one-sided limits, 1/2, which
    is the same idea as the trapezoid rule at a jump (exercise R1.3).
    """
    if option == "call":
        return np.maximum(S - K, 0.0)
    elif option == "digital":
        values = np.where(S > K, 1.0, 0.0)
        on_strike = np.abs(S - K) < 1e-9 * K
        values[on_strike] = 0.5
        return values
    else:
        raise ValueError("option must be 'call' or 'digital'")


def upper_boundary(S_max, K, r, time_left, option):
    """Option value at the top of the grid, where the option is deep in the money.

    A deep in the money call behaves like a forward, S - K exp(-r(T - t)).
    A deep in the money digital is certain to pay 1, so it is worth exp(-r(T - t)).
    """
    if option == "call":
        return S_max - K * np.exp(-r * time_left)
    elif option == "digital":
        return np.exp(-r * time_left)
    else:
        raise ValueError("option must be 'call' or 'digital'")


def price_heat_grid(option, K, r, sigma, T, x_max, n_x, n_tau, theta, n_rannacher=0):
    """Price a European option by solving the transformed heat equation.

    The grid is x in [-x_max, x_max] with n_x intervals (n_x even, so that
    x = 0, i.e. S = K, is a grid node) and tau in [0, sigma^2 T / 2] with
    n_tau steps. The lower boundary value is V = 0, which is correct for both
    a call and a digital call as S tends to 0.

    Returns the stock prices S on the grid and the option values V at t = 0.
    """
    k = 2 * r / sigma**2
    alpha = -0.5 * (k - 1)
    beta = -0.25 * (k + 1) ** 2

    x = np.linspace(-x_max, x_max, n_x + 1)
    tau_max = 0.5 * sigma**2 * T
    tau = np.linspace(0.0, tau_max, n_tau + 1)
    dx = x[1] - x[0]
    dtau = tau[1] - tau[0]
    lam = dtau / dx**2
    S = K * np.exp(x)

    # Initial condition. V = K exp(alpha x + beta tau) u, and tau = 0 at expiry,
    # so u(x, 0) = payoff / K * exp(-alpha x).
    u0 = payoff(S, K, option) / K * np.exp(-alpha * x)

    # Boundary values for u at every time level, from the boundary values for V.
    time_left = 2 * tau / sigma**2  # calendar time to expiry, T - t
    left = np.zeros(n_tau + 1)
    V_top = upper_boundary(S[-1], K, r, time_left, option)
    right = V_top / K * np.exp(-alpha * x[-1] - beta * tau)

    u = heat_theta_scheme(u0, left, right, lam, theta, n_rannacher)

    # Undo the change of variables at t = 0, i.e. tau = tau_max.
    V = K * np.exp(alpha * x + beta * tau_max) * u
    return S, V


def grid_greeks(S, V):
    """Delta and gamma from option values on a grid that is uniform in x = ln(S / K).

    Central differences give V_x and V_xx. The chain rule with dx/dS = 1/S then gives

        delta = V_x / S,
        gamma = (V_xx - V_x) / S^2.

    The values are returned at the interior nodes only.
    """
    dx = np.log(S[1] / S[0])
    V_x = (V[2:] - V[:-2]) / (2 * dx)
    V_xx = (V[2:] - 2 * V[1:-1] + V[:-2]) / dx**2
    S_inner = S[1:-1]
    delta = V_x / S_inner
    gamma = (V_xx - V_x) / S_inner**2
    return S_inner, delta, gamma


def price_call_explicit_S(K, r, sigma, T, S_max, n_S, n_t):
    """Explicit Euler for the Black-Scholes PDE in the original variable S.

    The grid is S_j = j dS for j = 0, ..., n_S with dS = S_max / n_S, and we
    march backwards from expiry with n_t steps of size dt = T / n_t.
    Replacing the S derivatives by central differences gives, at an interior node,

        V_j(new) = a_j V_{j-1} + b_j V_j + c_j V_{j+1},
        a_j = dt (sigma^2 j^2 - r j) / 2,
        b_j = 1 - dt (sigma^2 j^2 + r),
        c_j = dt (sigma^2 j^2 + r j) / 2.

    Returns S and V at t = 0.
    """
    S = np.linspace(0.0, S_max, n_S + 1)
    dt = T / n_t
    V = np.maximum(S - K, 0.0)

    j = np.arange(1, n_S)
    a = 0.5 * dt * (sigma**2 * j**2 - r * j)
    b = 1 - dt * (sigma**2 * j**2 + r)
    c = 0.5 * dt * (sigma**2 * j**2 + r * j)

    for n in range(n_t):
        time_left = (n + 1) * dt
        V_new = np.empty_like(V)
        V_new[1:-1] = a * V[:-2] + b * V[1:-1] + c * V[2:]
        V_new[0] = 0.0
        V_new[-1] = S_max - K * np.exp(-r * time_left)
        V = V_new

    return S, V
