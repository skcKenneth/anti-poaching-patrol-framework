"""Stability and bifurcation plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from visualizations.style_guide import apply_academic_style, save_academic_figure


def plot_stability_bifurcation(
    staff_range: NDArray[np.float64],
    stability_scores: NDArray[np.float64],
    threshold: float,
    current_staff: int,
) -> Path:
    """Plots Jacobian dominant-eigenvalue trajectory versus staffing.

    Args:
        staff_range: Staffing grid on x-axis.
        stability_scores: Dominant real eigenvalue values.
        threshold: Critical staffing red-line.
        current_staff: Current staffing level.
    """
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(11, 6.8), constrained_layout=True)
    ax.plot(staff_range, stability_scores, color="navy", lw=2.0, label=r"$\Re(\lambda_{\max})$")

    collapse_mask = staff_range < threshold
    safe_mask = ~collapse_mask
    ax.fill_between(staff_range, stability_scores, 0.0, where=collapse_mask, color="red", alpha=0.22, label="Collapse zone")
    ax.fill_between(staff_range, stability_scores, 0.0, where=safe_mask, color="green", alpha=0.12, label="Safe zone")

    ax.axhline(0.0, color="black", lw=1.0, linestyle=":")
    ax.axvline(threshold, color="red", linestyle="--", lw=1.6, label=f"Red-line threshold ({threshold:.1f})")
    ax.axvline(float(current_staff), color="green", linestyle="-", lw=1.6, label=f"Current staff ({current_staff})")

    ax.set_title("Jacobian Stability Bifurcation")
    ax.set_xlabel("Total personnel (staff)")
    ax.set_ylabel(r"Dominant eigenvalue real part, $\Re(\lambda_{\max})$")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "stability_bifurcation.png"
    save_academic_figure(fig, save_path)
    return save_path