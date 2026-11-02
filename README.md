# feynman-kac-option-pricing
Numerical option pricing engine exploring the Feynman-Kac duality; comparing finite-difference PDE schemes and Euler-Maruyama SDE simulations.

**What this project does**

The Feynman-Kac formula says that the price of an option is both the solution of a partial differential equation and an expected value over random stock price paths. This project prices European options in both ways and measures every error against the exact Black-Scholes answer. On the PDE side we transform the Black-Scholes equation into the heat equation and solve it with the explicit, implicit and Crank-Nicolson schemes, including Rannacher smoothing and the Greeks. On the Monte Carlo side we simulate the stock exactly and with the Euler-Maruyama method, and separate the statistical error from the time stepping bias. We then ask how both methods scale with the number of underlying stocks, and find where Monte Carlo overtakes grid methods. An extension prices baskets of up to 20 correlated stocks with a geometric basket control variate. The project was written for COSC2500 (Numerical Methods in Computational Science) at The University of Queensland in Semester 2, 2026.

**Repository layout**

```
pricing/                  Python modules used by the notebooks
    black_scholes.py      closed form prices and Greeks
    pde.py                finite difference solvers (theta scheme, Rannacher smoothing)
    sde.py                exact and Euler-Maruyama simulation
    high_dim.py           tensor product grids against Monte Carlo in d dimensions
    basket.py             correlated basket Monte Carlo with a control variate
    plot_style.py         shared figure style
notebooks/
    00_overview.ipynb     introduction, Feynman-Kac, summary of findings, references
    01_pde.ipynb          Part 1: finite difference solution of the PDE
    02_sde.ipynb          Part 2: Monte Carlo simulation of the SDE
    03_high_dimensions.ipynb   Part 3: grids against Monte Carlo as the dimension grows
    04_basket_extension.ipynb  Part 4: basket options on many correlated stocks
figures/                  every figure, saved by the notebooks
report/
    main.tex              LaTeX source of the project report
    main.pdf              the compiled report
tests/test_pricing.py     checks of every method against known answers
```

**Running the code**

Install the dependencies with `pip install -r requirements.txt`. Run the tests from the top folder with `pytest`, which takes a few seconds. Then run the notebooks in order. Each one runs in well under a minute on a laptop and regenerates its figures, and every random number comes from a seeded generator, so the results are reproducible. The report in `report/main.pdf` is written up from the notebooks. To rebuild it, compile `report/main.tex` with any LaTeX distribution, for example `tectonic main.tex` from inside the `report` folder, or upload the repository to Overleaf. It reads its figures from the `figures` folder.

**Main results**

In one dimension the Crank-Nicolson solver with Rannacher smoothing prices a call to about $3 \times 10^{-6}$ in under half a second, an accuracy that plain Monte Carlo would need weeks to reach. Plain Crank-Nicolson with large time steps does not converge for a digital option at all, and two Rannacher steps are not enough for the digital's gamma, while four are. As the number of stocks grows, a second order grid's convergence rate per unit of work falls like $2/d$ while Monte Carlo keeps the rate $1/2$, and on a test problem with an exact answer the measured crossover is at about four to five dimensions for quadrature and three for a PDE grid. The full discussion is in the notebooks.
