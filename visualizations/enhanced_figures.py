"""Visualizations for the three new analytical contributions:

  1. Feedback loop convergence plots.
  2. Multi-objective Pareto frontier with knee-point identification.
  3. GBIF occurrence overlay on the WPP field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.typing import NDArray

from src.feedback_loop import ConvergenceRecord
from visualizations.style_guide import apply_academic_style, save_academic_figure


# ──────────────────────────────────────────────────────────────────────────────
# 1. Feedback loop convergence
# ──────────────────────────────────────────────────────────────────────────────

def plot_feedback_convergence(
    record: ConvergenceRecord,
    output_dir: Path,
) -> Path:
    """Four-panel convergence diagnostic for the self-consistent loop.

    Panels:
      (a) Relative deployment change per iteration (convergence criterion).
      (b) Equilibrium poacher activity Z* per iteration.
      (c) Objective value (weighted interception) per iteration.
      (d) NHPP baseline threat intensity λ₀ per iteration.
    """
    apply_academic_style()
    iters = np.arange(1, record.n_iterations + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        "Self-consistent feedback loop: convergence diagnostics",
        fontsize=13, fontweight="semibold",
    )

    # (a) relative change
    ax = axes[0, 0]
    ax.semilogy(iters, record.relative_change, "o-", color="#2166ac", linewidth=1.8,
                markersize=5, label=r"$\|\tau_k - \tau_{k-1}\| / \|\tau_{k-1}\|$")
    ax.axhline(1e-3, color="grey", linestyle="--", linewidth=1, label="Convergence threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative change")
    ax.set_title("(a) Deployment convergence")
    ax.legend(fontsize=8)

    # (b) Z*
    ax = axes[0, 1]
    ax.plot(iters, record.z_star, "s-", color="#d6604d", linewidth=1.8, markersize=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$Z^*$ (poacher activity)")
    ax.set_title("(b) Equilibrium poacher activity")

    # (c) Objective
    ax = axes[1, 0]
    ax.plot(iters, record.objective, "^-", color="#4dac26", linewidth=1.8, markersize=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted interception")
    ax.set_title("(c) Objective value")

    # (d) Lambda0
    ax = axes[1, 1]
    ax.plot(iters, record.lambda0_dry, "D-", color="#7b3294", linewidth=1.8, markersize=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda_0^{\mathrm{dry}}$ (NHPP baseline)")
    ax.set_title(r"(d) Updated threat intensity $\lambda_0$")

    save_path = output_dir / "feedback_convergence.png"
    save_academic_figure(fig, save_path)
    return save_path


def plot_deployment_comparison(
    deploy_open_loop: NDArray[np.float64],
    deploy_feedback: NDArray[np.float64],
    output_dir: Path,
) -> Path:
    """Side-by-side comparison of open-loop vs self-consistent deployment."""
    apply_academic_style()
    diff = deploy_feedback - deploy_open_loop

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    fig.suptitle(
        "Open-loop vs self-consistent deployment",
        fontsize=13, fontweight="semibold",
    )

    vmax = float(max(deploy_open_loop.max(), deploy_feedback.max()))
    for ax, data, title in zip(
        axes,
        [deploy_open_loop, deploy_feedback, diff],
        ["(a) Open-loop (Layer 2 only)", "(b) Self-consistent (feedback loop)",
         "(c) Difference (feedback − open-loop)"],
    ):
        cmap = "Reds" if "Difference" not in title else "coolwarm"
        vkw = {"vmin": 0, "vmax": vmax} if "Difference" not in title else {}
        im = ax.imshow(data, cmap=cmap, origin="lower", **vkw)
        ax.set_title(title)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(im, ax=ax, label="Staff / cell", shrink=0.85)

    save_path = output_dir / "feedback_deployment_comparison.png"
    save_academic_figure(fig, save_path)
    return save_path


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-objective Pareto frontier with knee point
# ──────────────────────────────────────────────────────────────────────────────

def _find_knee_point(capture: NDArray, fairness: NDArray) -> int:
    """Identifies the knee point via maximum perpendicular distance.

    The knee is the Pareto point furthest from the line connecting the
    minimum-fairness and maximum-capture endpoints, i.e. the point of
    maximum marginal return in the capture-equity tradeoff.
    """
    if len(capture) < 3:
        return 0
    # Normalise both axes to [0,1]
    c_n = (capture - capture.min()) / max(float(capture.max() - capture.min()), 1e-9)
    f_n = (fairness - fairness.min()) / max(float(fairness.max() - fairness.min()), 1e-9)

    # Direction vector of the line from first to last Pareto point
    dx = c_n[-1] - c_n[0]
    dy = f_n[-1] - f_n[0]
    length = max(np.sqrt(dx**2 + dy**2), 1e-9)

    # Perpendicular distance from each point to the line
    dist = np.abs(dy * c_n - dx * f_n + (c_n[-1] * f_n[0] - c_n[0] * f_n[-1])) / length
    return int(np.argmax(dist))


def plot_pareto_frontier_enhanced(
    pareto_points: Dict[str, NDArray],
    baseline_capture: float,
    baseline_fairness: float,
    output_dir: Path,
    feedback_capture: Optional[float] = None,
    feedback_fairness: Optional[float] = None,
) -> Path:
    """Enhanced Pareto frontier plot with knee point and management scenarios.

    Args:
        pareto_points: Dict with keys 'capture', 'fairness', 'staff'.
        baseline_capture: Capture score from single-objective SLSQP.
        baseline_fairness: Fairness (1 − Gini) from baseline deployment.
        output_dir: Where to save the figure.
        feedback_capture: Capture score from self-consistent loop.
        feedback_fairness: Fairness from self-consistent loop.
    """
    apply_academic_style()
    capture = pareto_points["capture"].astype(float)
    fairness = pareto_points["fairness"].astype(float)
    staff = pareto_points["staff"].astype(float)

    # Sort by capture for clean curve
    order = np.argsort(capture)
    capture, fairness, staff = capture[order], fairness[order], staff[order]

    knee_idx = _find_knee_point(capture, fairness)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(
        "Multi-objective Pareto frontier: capture efficiency vs. patrol equity",
        fontsize=13, fontweight="semibold",
    )

    # ── Left panel: Pareto curve ──
    ax = axes[0]
    sc = ax.scatter(capture, fairness, c=staff, cmap="viridis", s=60,
                    zorder=3, label="Pareto points")
    ax.plot(capture, fairness, "-", color="#4393c3", linewidth=1.5,
            alpha=0.6, zorder=2)
    plt.colorbar(sc, ax=ax, label="Effective staff deployed")

    # Knee point
    ax.scatter(capture[knee_idx], fairness[knee_idx], s=150, marker="*",
               color="#d6604d", zorder=5, label="Knee point (balanced)")

    # Baseline (single-objective SLSQP)
    ax.scatter(baseline_capture, baseline_fairness, s=120, marker="D",
               color="#4dac26", zorder=5, label="Baseline (SLSQP, max capture)")

    # Self-consistent feedback solution
    if feedback_capture is not None and feedback_fairness is not None:
        ax.scatter(feedback_capture, feedback_fairness, s=120, marker="P",
                   color="#7b3294", zorder=5, label="Self-consistent (feedback loop)")

    # Management scenario labels
    scenario_labels = [
        (capture[0], fairness[0], "Max equity"),
        (capture[-1], fairness[-1], "Max capture"),
    ]
    for cx, fy, label in scenario_labels:
        ax.annotate(
            label, xy=(cx, fy),
            xytext=(cx + 0.02 * (capture[-1] - capture[0]),
                    fy + 0.02 * (fairness.max() - fairness.min())),
            fontsize=8, color="dimgrey",
            arrowprops=dict(arrowstyle="->", color="dimgrey", lw=0.8),
        )

    ax.set_xlabel("Weighted capture score")
    ax.set_ylabel("Patrol equity (1 − Gini)")
    ax.set_title("(a) Pareto frontier")
    ax.legend(fontsize=8)

    # ── Right panel: tradeoff slope ──
    ax2 = axes[1]
    if len(capture) > 2:
        d_capture = np.diff(capture)
        d_fairness = np.diff(fairness)
        marginal_rate = np.where(
            np.abs(d_capture) > 1e-12,
            -d_fairness / d_capture,
            0.0,
        )
        mid_capture = 0.5 * (capture[:-1] + capture[1:])
        ax2.bar(mid_capture, marginal_rate,
                width=0.8 * float(np.median(np.diff(mid_capture))),
                color="#4393c3", alpha=0.7, label="Fairness cost per unit capture gain")
        ax2.axvline(capture[knee_idx], color="#d6604d", linestyle="--",
                    linewidth=1.5, label="Knee point")
        ax2.set_xlabel("Weighted capture score")
        ax2.set_ylabel("−Δ fairness / Δ capture")
        ax2.set_title("(b) Marginal equity–capture tradeoff")
        ax2.legend(fontsize=8)

    save_path = output_dir / "pareto_frontier_enhanced.png"
    save_academic_figure(fig, save_path)
    return save_path


# ──────────────────────────────────────────────────────────────────────────────
# 3. GBIF occurrence overlay
# ──────────────────────────────────────────────────────────────────────────────

def plot_gbif_wpp_overlay(
    wpp_field: NDArray[np.float64],
    occurrence_prior: NDArray[np.float64],
    wpp_with_prior: NDArray[np.float64],
    output_dir: Path,
    n_etosha_records: int = 0,
) -> Path:
    """Three-panel figure showing GBIF data anchoring.

    Panels:
      (a) GBIF occurrence density (kernel-smoothed).
      (b) Original WPP field (model-only).
      (c) WPP field after GBIF prior integration.
    """
    apply_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    fig.suptitle(
        f"Real-data anchoring: GBIF black rhino occurrences "
        f"({n_etosha_records} records within Etosha bounding box)",
        fontsize=12, fontweight="semibold",
    )

    datasets = [
        (occurrence_prior, "viridis", "(a) GBIF occurrence density\n(Diceros bicornis, GBIF 2026)",
         "Occurrence density"),
        (wpp_field, "YlOrRd", "(b) WPP field — model only", "WPP score"),
        (wpp_with_prior, "YlOrRd", "(c) WPP field — with GBIF prior", "WPP score"),
    ]

    for ax, (data, cmap, title, cbar_label) in zip(axes, datasets):
        im = ax.imshow(data, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)

    save_path = output_dir / "gbif_wpp_overlay.png"
    save_academic_figure(fig, save_path)
    return save_path
