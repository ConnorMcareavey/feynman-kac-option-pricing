"""Numerical option pricing project.

The closed form prices and Greeks that serve as the ground truth are in
black_scholes.py. The finite difference solvers for the Black-Scholes PDE are
in pde.py, and the exact and Euler-Maruyama simulation of the stock price is
in sde.py. The comparison of tensor product grids with Monte Carlo in d
dimensions is in high_dim.py, and the correlated basket simulation with a
control variate (the optional extension) is in basket.py. The shared figure
style is in plot_style.py.
"""
