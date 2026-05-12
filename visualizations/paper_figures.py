"""Publication-grade figures for the anti-poaching patrol framework."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy.typing import NDArray

from src.jacobian_stability import StabilityEngine
from visualizations.style_guide import apply_academic_style, save_academic_figure


def _apply_paper_style() -> None:
    apply_academic_style()


def _save_figure(fig: plt.Figure, save_path: Path) -> None:
    """Saves figures with safer margins to prevent overlaps."""
    save_academic_figure(fig, save_path)


def plot_phase_space_portrait(
    stability_engine: StabilityEngine,
    output_dir: Path,
    stable_staff: float,
    collapse_staff: float,
    predator_fixed: float = 18.0,
) -> Path:
    """Plots 2D phase portrait on prey-poacher subspace."""
    _apply_paper_style()
    prey_axis = np.linspace(120.0, 1150.0, 33)
    poacher_axis = np.linspace(1.0, 36.0, 29)
    prey_grid, poacher_grid = np.meshgrid(prey_axis, poacher_axis)

    def _field(staff_value: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        density = staff_value / stability_engine.area_km2
        p = stability_engine.params
        d_prey = (
            p.r_prey * prey_grid * (1.0 - prey_grid / p.k_prey)
            - p.alpha_predation * prey_grid * predator_fixed
            - p.alpha_poaching * prey_grid * poacher_grid
        )
        d_poachers = (
            p.rho_poacher_growth * prey_grid * poacher_grid
            - p.sigma_suppression * density * poacher_grid
            - p.m_poacher * poacher_grid
        )
        speed = np.sqrt(d_prey**2 + d_poachers**2) + 1e-9
        # Compress extreme magnitudes but keep direction, producing smoother streamlines.
        return d_prey / speed, d_poachers / speed

    def _rk4_step(state: NDArray[np.float64], dt: float, density: float) -> NDArray[np.float64]:
        k1 = stability_engine.rhs(state, patrol_density=density)
        k2 = stability_engine.rhs(state + 0.5 * dt * k1, patrol_density=density)
        k3 = stability_engine.rhs(state + 0.5 * dt * k2, patrol_density=density)
        k4 = stability_engine.rhs(state + dt * k3, patrol_density=density)
        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return np.maximum(nxt, np.array([1.0, 0.2, 0.1], dtype=float))

    def _trajectory(
        init_state: NDArray[np.float64],
        staff_value: float,
        dt: float = 0.02,
        steps: int = 1200,
    ) -> NDArray[np.float64]:
        density = staff_value / stability_engine.area_km2
        traj = np.zeros((steps, 2), dtype=float)
        state = np.array([init_state[0], init_state[1], init_state[2]], dtype=float)
        for t in range(steps):
            traj[t, 0] = state[0]
            traj[t, 1] = state[2]
            state = _rk4_step(state, dt=dt, density=density)
            # Stop tracing once trajectory leaves plotting window.
            if (
                state[0] < prey_axis[0]
                or state[0] > prey_axis[-1]
                or state[2] < poacher_axis[0]
                or state[2] > poacher_axis[-1]
            ):
                return traj[: t + 1]
        return traj

    initials = np.array(
        [
            [220.0, 16.0, 34.0],
            [320.0, 14.0, 22.0],
            [430.0, 20.0, 14.0],
            [620.0, 18.0, 10.0],
            [820.0, 15.0, 7.0],
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, staff_value, title in [
        (axes[0], stable_staff, "Adequate staffing: stable attractor"),
        (axes[1], collapse_staff, "Staff shortage: collapse topology"),
    ]:
        u, v = _field(staff_value)
        speed = np.sqrt(u**2 + v**2)
        ax.streamplot(
            prey_axis,
            poacher_axis,
            u,
            v,
            color=speed,
            cmap="Greys",
            density=1.15,
            linewidth=0.8,
            arrowsize=0.8,
        )
        for init_state in initials:
            traj = _trajectory(init_state, staff_value=staff_value)
            ax.plot(traj[:, 0], traj[:, 1], lw=2.0)
            ax.scatter(traj[0, 0], traj[0, 1], s=20, c="black", alpha=0.7)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=18, c="#1f3b73", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Prey population N")
        ax.set_ylabel("Poacher intensity Z")
        ax.set_xlim(prey_axis[0], prey_axis[-1])
        ax.set_ylim(poacher_axis[0], poacher_axis[-1])

    save_path = output_dir / "phase_space_portrait.png"
    _save_figure(fig, save_path)
    return save_path


def plot_3d_topology_wpp_surface(
    topology_mask: NDArray[np.float64],
    wpp_field: NDArray[np.float64],
    deployment_map: NDArray[np.float64],
    output_dir: Path,
) -> Path:
    """Plots 3D WPP surface with terrain proxy and deployment peaks."""
    _apply_paper_style()
    yy, xx = np.indices(wpp_field.shape, dtype=float)
    terrain = 0.45 * (topology_mask - 1.0)
    z_surface = terrain + (wpp_field / max(float(np.max(wpp_field)), 1e-9))

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        xx,
        yy,
        z_surface,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="Normalized WPP + terrain")

    hot_mask = deployment_map >= np.percentile(deployment_map, 90)
    ax.scatter(
        xx[hot_mask],
        yy[hot_mask],
        z_surface[hot_mask] + 0.15 + deployment_map[hot_mask] / (np.max(deployment_map) + 1e-9),
        c=deployment_map[hot_mask],
        cmap="Reds",
        s=35,
        depthshade=True,
        label="High deployment density",
    )
    ax.set_title("3D topology-WPP landscape with patrol density peaks", pad=12)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_zlabel("Landscape protection potential")
    ax.view_init(elev=32, azim=-127)
    ax.legend(loc="upper left", borderaxespad=0.8)

    save_path = output_dir / "topology_wpp_3d_surface.png"
    _save_figure(fig, save_path)
    return save_path


def plot_multi_metric_radar(
    scenario_metrics_abs: Dict[str, Dict[str, float]],
    scenario_metrics_rel: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> Path:
    """Plots dual-panel radar chart (absolute and baseline-relative)."""
    _apply_paper_style()
    labels = ["Interception", "WPP Coverage", "Stability", "Tech Resilience", "Fairness"]
    metric_keys = [
        "interception_rate",
        "wpp_coverage",
        "stability_margin_norm",
        "low_tech_resilience",
        "fairness",
    ]
    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8.2),
        subplot_kw={"polar": True},
        constrained_layout=True,
    )
    scenario_names = list(scenario_metrics_abs.keys())

    def _draw_panel(ax: plt.Axes, metrics: Dict[str, Dict[str, float]], y_max: float, title: str) -> None:
        for scenario_name in scenario_names:
            values = np.array([float(metrics[scenario_name][key]) for key in metric_keys], dtype=float)
            values = np.clip(values, 0.0, y_max)
            values = np.concatenate([values, values[:1]])
            ax.plot(angles, values, linewidth=2.2, marker="o", markersize=4, label=scenario_name)
            ax.fill(angles, values, alpha=0.14)
        ax.set_thetagrids(angles[:-1] * 180.0 / np.pi, labels, fontsize=10)
        ax.set_ylim(0.0, y_max)
        ax.set_rlabel_position(18)
        ax.set_title(title, pad=18)

    _draw_panel(axes[0], scenario_metrics_abs, y_max=1.0, title="Absolute Multi-Metric Scores")
    _draw_panel(axes[1], scenario_metrics_rel, y_max=1.2, title="Relative Performance (Baseline=1.0)")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True)
    fig.text(
        0.5,
        0.01,
        "Left: absolute scores in [0,1]. Right: value / baseline (capped at 1.2 for readability).",
        ha="center",
        va="center",
        fontsize=9,
        color="#4B5563",
    )

    save_path = output_dir / "scenario_radar.png"
    _save_figure(fig, save_path)
    return save_path


def plot_lorenz_curve(
    lorenz_x: NDArray[np.float64],
    lorenz_y: NDArray[np.float64],
    gini_value: float,
    output_dir: Path,
) -> Path:
    """Plots Lorenz curve of deployment concentration."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    ax.plot(lorenz_x, lorenz_y, lw=2.4, color="#304C89", label="Lorenz curve")
    ax.plot([0, 1], [0, 1], "--", color="#7D8597", lw=1.5, label="Equality line")
    ax.fill_between(lorenz_x, lorenz_y, lorenz_x, color="#6C8EBF", alpha=0.2)
    ax.set_xlabel("Cumulative share of grid cells")
    ax.set_ylabel("Cumulative share of patrol allocation")
    ax.set_title(f"Deployment inequality profile (Gini={gini_value:.3f})")
    ax.legend(loc="lower right")

    save_path = output_dir / "lorenz_curve.png"
    _save_figure(fig, save_path)
    return save_path


def plot_pareto_front(
    pareto_staff: NDArray[np.float64],
    pareto_capture: NDArray[np.float64],
    pareto_fairness: NDArray[np.float64],
    output_dir: Path,
) -> Path:
    """Plots Pareto frontier between capture and fairness."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6.8), constrained_layout=True)
    sc = ax.scatter(pareto_capture, pareto_fairness, c=pareto_staff, cmap="plasma", s=50)
    ax.plot(pareto_capture, pareto_fairness, color="#1B4965", lw=1.7, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Staff level on Pareto frontier")
    ax.set_xlabel("Capture performance")
    ax.set_ylabel("Fairness score (1 - Gini)")
    ax.set_title("Pareto frontier: protection gain vs deployment equity")

    save_path = output_dir / "pareto_front.png"
    _save_figure(fig, save_path)
    return save_path


def plot_marginal_utility_curve(
    staff_axis: NDArray[np.float64],
    marginal_gain_curve: NDArray[np.float64],
    output_dir: Path,
) -> Path:
    """Plots marginal gain from each additional ranger."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6.8), constrained_layout=True)
    ax.plot(staff_axis, marginal_gain_curve, color="#2A9D8F", lw=2.0)
    ax.set_xlabel("Additional ranger index")
    ax.set_ylabel("Marginal weighted-capture gain")
    ax.set_title("Diminishing returns under incremental staffing")
    ax.grid(alpha=0.3)

    save_path = output_dir / "marginal_utility_curve.png"
    _save_figure(fig, save_path)
    return save_path


def plot_voronoi_patrol_zones(
    zone_map: NDArray[np.float64],
    active_coords: NDArray[np.float64],
    output_dir: Path,
) -> Path:
    """Plots Voronoi-like patrol zones induced by active deployment posts."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6.8), constrained_layout=True)
    im = ax.imshow(zone_map, cmap="tab20", origin="lower")
    if active_coords.size > 0:
        ax.scatter(active_coords[:, 1], active_coords[:, 0], c="black", s=10, marker="x", label="Active posts")
        ax.legend(loc="upper right")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_title("Voronoi tessellation of patrol responsibility zones")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Zone id")

    save_path = output_dir / "voronoi_patrol_zones.png"
    _save_figure(fig, save_path)
    return save_path
