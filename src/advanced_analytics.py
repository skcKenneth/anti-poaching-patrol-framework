"""Advanced evaluation analytics for the anti-poaching patrol framework."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


def weighted_capture_score(
    deployment: NDArray[np.float64],
    wpp_field: NDArray[np.float64],
    friction_field: NDArray[np.float64],
    lambda_detection: float = 0.25,
) -> float:
    """Returns weighted capture objective value."""
    term = 1.0 - np.exp(-lambda_detection * deployment / np.maximum(friction_field, 1e-9))
    return float(np.sum(wpp_field * term))


def weighted_interception_rate(
    deployment: NDArray[np.float64],
    wpp_field: NDArray[np.float64],
    friction_field: NDArray[np.float64],
    lambda_detection: float = 0.25,
) -> float:
    """Returns WPP-weighted interception probability."""
    term = 1.0 - np.exp(-lambda_detection * deployment / np.maximum(friction_field, 1e-9))
    return float(np.average(term, weights=np.maximum(wpp_field, 1e-9)))


def run_monte_carlo_robustness(
    deployment: NDArray[np.float64],
    wpp_field: NDArray[np.float64],
    friction_field: NDArray[np.float64],
    n_runs: int = 1000,
    hotspot_sigma: float = 0.12,
    friction_sigma: float = 0.08,
    lambda_detection: float = 0.25,
    seed: int = 2026,
) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
    """Runs Monte Carlo perturbation tests around the fixed best strategy."""
    rng = np.random.default_rng(seed)

    # Perturb hotspots (WPP intensity) and terrain friction with Gaussian shocks.
    hotspot_noise = rng.normal(loc=0.0, scale=hotspot_sigma, size=(n_runs,) + wpp_field.shape)
    friction_noise = rng.normal(loc=0.0, scale=friction_sigma, size=(n_runs,) + friction_field.shape)

    wpp_perturbed = np.maximum(wpp_field[None, :, :] * (1.0 + hotspot_noise), 0.0)
    friction_perturbed = np.maximum(friction_field[None, :, :] * (1.0 + friction_noise), 1e-6)

    term = 1.0 - np.exp(-lambda_detection * deployment[None, :, :] / friction_perturbed)
    capture_samples = np.sum(wpp_perturbed * term, axis=(1, 2))
    interception_samples = np.sum(wpp_perturbed * term, axis=(1, 2)) / np.sum(
        np.maximum(wpp_perturbed, 1e-9), axis=(1, 2)
    )

    ci_capture = np.percentile(capture_samples, [2.5, 97.5])
    ci_intercept = np.percentile(interception_samples, [2.5, 97.5])

    summary = {
        "mc_runs": float(n_runs),
        "mc_capture_mean": float(np.mean(capture_samples)),
        "mc_capture_std": float(np.std(capture_samples)),
        "mc_capture_ci_low": float(ci_capture[0]),
        "mc_capture_ci_high": float(ci_capture[1]),
        "mc_interception_mean": float(np.mean(interception_samples)),
        "mc_interception_std": float(np.std(interception_samples)),
        "mc_interception_ci_low": float(ci_intercept[0]),
        "mc_interception_ci_high": float(ci_intercept[1]),
        "mc_capture_cv": float(np.std(capture_samples) / max(np.mean(capture_samples), 1e-9)),
        "mc_interception_cv": float(
            np.std(interception_samples) / max(np.mean(interception_samples), 1e-9)
        ),
    }

    samples = {
        "capture_samples": capture_samples.astype(float),
        "interception_samples": interception_samples.astype(float),
    }
    return summary, samples


def _gini(values: NDArray[np.float64]) -> float:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(np.maximum(values, 0.0))
    if np.allclose(sorted_vals.sum(), 0.0):
        return 0.0
    n = sorted_vals.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1.0) / n)


def compute_fairness_metrics(deployment: NDArray[np.float64]) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
    """Computes Gini coefficient and Lorenz curve coordinates."""
    flat = np.sort(np.maximum(deployment.ravel().astype(float), 0.0))
    n = flat.size
    cum_staff = np.cumsum(flat)
    total = max(float(cum_staff[-1]) if n > 0 else 0.0, 1e-9)

    lorenz_x = np.linspace(0.0, 1.0, n + 1)
    lorenz_y = np.concatenate(([0.0], cum_staff / total))
    gini = _gini(flat)

    return {"deployment_gini": float(gini)}, {"lorenz_x": lorenz_x, "lorenz_y": lorenz_y}


def _greedy_allocation_path(
    wpp_field: NDArray[np.float64],
    friction_field: NDArray[np.float64],
    max_staff: int,
    lambda_detection: float = 0.25,
    per_cell_cap: int = 50,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Builds integer allocation path and marginal gains via greedy increments."""
    wpp = np.maximum(wpp_field.ravel().astype(float), 0.0)
    mu = np.maximum(friction_field.ravel().astype(float), 1e-9)
    n_cells = wpp.size
    alloc = np.zeros(n_cells, dtype=float)

    capture_series = np.zeros(max_staff + 1, dtype=float)
    marginal_series = np.zeros(max_staff, dtype=float)

    for k in range(max_staff):
        gain = wpp * (
            np.exp(-lambda_detection * alloc / mu)
            - np.exp(-lambda_detection * (alloc + 1.0) / mu)
        )
        gain = np.where(alloc < float(per_cell_cap), gain, -1e18)
        idx = int(np.argmax(gain))
        marginal_gain = max(float(gain[idx]), 0.0)
        alloc[idx] += 1.0
        marginal_series[k] = marginal_gain
        capture_series[k + 1] = capture_series[k] + marginal_gain

    return alloc.reshape(wpp_field.shape), capture_series, marginal_series


def compute_marginal_utility_curve(
    wpp_field: NDArray[np.float64],
    friction_field: NDArray[np.float64],
    total_staff: int,
    lambda_detection: float = 0.25,
) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
    """Computes +1 ranger marginal utility and diminishing-return diagnostics."""
    _, capture_series, marginal_series = _greedy_allocation_path(
        wpp_field=wpp_field,
        friction_field=friction_field,
        max_staff=int(total_staff),
        lambda_detection=lambda_detection,
    )
    staff_axis = np.arange(1, int(total_staff) + 1, dtype=float)

    # Diminishing-return threshold: first point below 25% of initial marginal gain.
    initial = max(float(marginal_series[0]), 1e-9)
    threshold_idx = int(np.argmax(marginal_series <= 0.25 * initial))
    threshold_staff = float(staff_axis[threshold_idx]) if marginal_series[threshold_idx] <= 0.25 * initial else float(
        total_staff
    )

    summary = {
        "marginal_gain_first_ranger": float(marginal_series[0]),
        "marginal_gain_last_ranger": float(marginal_series[-1]),
        "marginal_gain_decay_ratio": float(marginal_series[-1] / initial),
        "diminishing_return_staff_threshold": threshold_staff,
        "greedy_capture_at_total_staff": float(capture_series[-1]),
    }
    curve = {
        "staff_axis": staff_axis,
        "capture_curve": capture_series[1:],
        "marginal_gain_curve": marginal_series,
    }
    return summary, curve


def compute_pareto_front(
    wpp_field: NDArray[np.float64],
    friction_field: NDArray[np.float64],
    total_staff: int,
    lambda_detection: float = 0.25,
) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
    """Computes Pareto points from capture-equity tradeoff candidates."""
    wpp = np.maximum(wpp_field.ravel().astype(float), 0.0)
    mu = np.maximum(friction_field.ravel().astype(float), 1e-9)
    total_staff_int = int(total_staff)
    gamma_grid = np.linspace(0.0, 0.18, 25, dtype=float)
    n_candidates = gamma_grid.size

    capture_grid = np.zeros(n_candidates, dtype=float)
    fairness_grid = np.zeros(n_candidates, dtype=float)
    effective_staff_grid = np.zeros(n_candidates, dtype=float)

    for gi, gamma in enumerate(gamma_grid):
        allocation = np.zeros_like(wpp)
        capture_acc = 0.0
        for _ in range(total_staff_int):
            # Capture gain per +1 staff minus concentration penalty.
            gain = wpp * (
                np.exp(-lambda_detection * allocation / mu)
                - np.exp(-lambda_detection * (allocation + 1.0) / mu)
            )
            score = gain - gamma * allocation
            score = np.where(allocation < 50.0, score, -1e18)
            idx = int(np.argmax(score))
            marginal_gain = max(float(gain[idx]), 0.0)
            allocation[idx] += 1.0
            capture_acc += marginal_gain

        capture_grid[gi] = capture_acc
        fairness_grid[gi] = 1.0 - _gini(allocation)
        effective_staff_grid[gi] = float(np.sum(allocation))

    # Remove duplicates that can appear at close gamma values.
    point_matrix = np.stack([capture_grid, fairness_grid], axis=1)
    _, unique_idx = np.unique(np.round(point_matrix, decimals=8), axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    capture_grid = capture_grid[unique_idx]
    fairness_grid = fairness_grid[unique_idx]
    effective_staff_grid = effective_staff_grid[unique_idx]

    # Nondominated points in (capture max, fairness max).
    is_nd = np.ones(capture_grid.shape[0], dtype=bool)
    for i in range(capture_grid.shape[0]):
        dominates_i = (capture_grid >= capture_grid[i]) & (fairness_grid >= fairness_grid[i]) & (
            (capture_grid > capture_grid[i]) | (fairness_grid > fairness_grid[i])
        )
        dominates_i[i] = False
        if np.any(dominates_i):
            is_nd[i] = False

    nd_staff = effective_staff_grid[is_nd].astype(float)
    nd_capture = capture_grid[is_nd].astype(float)
    nd_fairness = fairness_grid[is_nd].astype(float)
    order = np.argsort(nd_capture)

    points = {
        "staff": nd_staff[order],
        "capture": nd_capture[order],
        "fairness": nd_fairness[order],
    }
    summary = {
        "pareto_points_count": float(points["staff"].size),
        "pareto_capture_min": float(np.min(points["capture"])),
        "pareto_capture_max": float(np.max(points["capture"])),
        "pareto_fairness_min": float(np.min(points["fairness"])),
        "pareto_fairness_max": float(np.max(points["fairness"])),
    }
    return summary, points


def compute_global_sensitivity(
    deployment: NDArray[np.float64],
    wpp_field: NDArray[np.float64],
    friction_field: NDArray[np.float64],
    lambda_detection: float = 0.25,
    delta: float = 0.10,
) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
    """Computes local elasticities around baseline scenario."""
    base_capture = weighted_capture_score(deployment, wpp_field, friction_field, lambda_detection=lambda_detection)

    scenarios = {
        "lambda_detection": (
            weighted_capture_score(
                deployment, wpp_field, friction_field, lambda_detection=lambda_detection * (1.0 + delta)
            ),
            weighted_capture_score(
                deployment, wpp_field, friction_field, lambda_detection=lambda_detection * (1.0 - delta)
            ),
        ),
        "friction_scale": (
            weighted_capture_score(
                deployment, wpp_field, friction_field * (1.0 + delta), lambda_detection=lambda_detection
            ),
            weighted_capture_score(
                deployment, wpp_field, friction_field * (1.0 - delta), lambda_detection=lambda_detection
            ),
        ),
        "wpp_hotspot_scale": (
            weighted_capture_score(
                deployment, wpp_field * (1.0 + delta), friction_field, lambda_detection=lambda_detection
            ),
            weighted_capture_score(
                deployment, wpp_field * (1.0 - delta), friction_field, lambda_detection=lambda_detection
            ),
        ),
    }

    names = []
    elasticity = []
    up_values = []
    down_values = []
    for key, (up, down) in scenarios.items():
        central_diff = (up - down) / max(2.0 * delta * base_capture, 1e-9)
        names.append(key)
        elasticity.append(float(central_diff))
        up_values.append(float(up))
        down_values.append(float(down))

    summary = {
        "sensitivity_base_capture": float(base_capture),
        "sensitivity_max_abs_elasticity": float(np.max(np.abs(elasticity))),
    }
    details = {
        "parameter_names": np.asarray(names, dtype=object),
        "elasticity": np.asarray(elasticity, dtype=float),
        "capture_up": np.asarray(up_values, dtype=float),
        "capture_down": np.asarray(down_values, dtype=float),
    }
    return summary, details


def compute_voronoi_zone_metrics(
    deployment: NDArray[np.float64],
    wpp_field: NDArray[np.float64],
    active_threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
    """Constructs Voronoi-like patrol zones and returns heterogeneity metrics."""
    active_coords = np.argwhere(deployment > active_threshold)
    yy, xx = np.indices(deployment.shape)

    if active_coords.shape[0] == 0:
        zone_map = np.full(deployment.shape, -1, dtype=int)
        summary = {
            "voronoi_active_posts": 0.0,
            "voronoi_zone_wpp_cv": 0.0,
            "voronoi_zone_area_cv": 0.0,
        }
        details = {
            "zone_map": zone_map.astype(float),
            "zone_wpp": np.zeros(1, dtype=float),
            "zone_area": np.zeros(1, dtype=float),
        }
        return summary, details

    dy = yy[None, :, :] - active_coords[:, 0][:, None, None]
    dx = xx[None, :, :] - active_coords[:, 1][:, None, None]
    dist2 = dy**2 + dx**2
    nearest = np.argmin(dist2, axis=0)
    zone_map = nearest.astype(int)

    n_zones = active_coords.shape[0]
    zone_wpp = np.bincount(zone_map.ravel(), weights=wpp_field.ravel(), minlength=n_zones).astype(float)
    zone_area = np.bincount(zone_map.ravel(), minlength=n_zones).astype(float)

    zone_wpp_cv = float(np.std(zone_wpp) / max(np.mean(zone_wpp), 1e-9))
    zone_area_cv = float(np.std(zone_area) / max(np.mean(zone_area), 1e-9))
    summary = {
        "voronoi_active_posts": float(n_zones),
        "voronoi_zone_wpp_cv": zone_wpp_cv,
        "voronoi_zone_area_cv": zone_area_cv,
    }
    details = {
        "zone_map": zone_map.astype(float),
        "zone_wpp": zone_wpp,
        "zone_area": zone_area,
        "active_coords": active_coords.astype(float),
    }
    return summary, details
