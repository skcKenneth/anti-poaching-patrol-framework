"""Shared academic plotting style for all project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


def apply_academic_style() -> None:
    """Applies consistent publication-style plotting parameters."""
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def save_academic_figure(fig: plt.Figure, save_path: Path) -> None:
    """Saves figure with margins that reduce text overlaps."""
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
