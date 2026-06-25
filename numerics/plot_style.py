"""Shared matplotlib style for paper figures."""

import matplotlib as mpl


def apply_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.figsize": [6.0, 4.0],
    })


def viridis_by_size(n):
    """Return n colorblind-safe colors ordered by model size (small -> large)."""
    import matplotlib.cm as cm
    import numpy as np
    return [cm.viridis(x) for x in np.linspace(0.15, 0.95, n)]


def format_param_count(n):
    return f"N = {int(n):,}"
