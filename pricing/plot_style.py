"""Shared matplotlib settings so that every figure in the project looks the same."""

import matplotlib.pyplot as plt
from cycler import cycler

# Line colours, always used in this order so that a method keeps its colour
# from one figure to the next. The order was checked for colour blind safety.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Neutral colours for exact solutions, reference slopes and axes.
INK = "#0b0b0b"
GREY = "#898781"


def use_project_style():
    """Apply the project's plotting defaults to matplotlib."""
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.prop_cycle": cycler(color=COLOURS),
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "text.color": INK,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })
