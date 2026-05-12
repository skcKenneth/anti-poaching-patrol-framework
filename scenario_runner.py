"""Scenario studies and evaluation suite for the anti-poaching patrol framework."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.ssg_dro_optimizer import DROOptimizer
from src.wpp_maut_engine import WPPEngine
from src.advanced_analytics import (
    compute_fairness_metrics,
    compute_global_sensitivity,
    compute_marginal_utility_curve,
    compute_pareto_front,
    compute_voronoi_zone_metrics,
    run_monte_carlo_robustness,
    weighted_interception_rate,
)
from utils.spatial_friction import FrictionMatrix
from utils.topology_constructor import EtoshaTopology
from visualizations.paper_figures import (
    plot_lorenz_curve,
    plot_marginal_utility_curve,
    plot_pareto_front,
    plot_voronoi_patrol_zones,
)
from visualizations.style_guide import apply_academic_style, save_academic_figure


def run_season_comparison(
    topology: EtoshaTopology, wpp_engine: WPPEngine, output_dir: Path
) -> Dict[str, float]:
    """Runs dry-vs-wet deployment comparison and saves figure."""
    apply_academic_style()
    friction_dry = FrictionMatrix(topology).generate_friction_map(season="dry")
    wpp_dry = wpp_engine.generate_wpp_field(topology, friction_dry, season="dry")
    deploy_dry = DROOptimizer(topology, friction_dry).solve(wpp_dry)

    friction_wet = FrictionMatrix(topology).generate_friction_map(season="wet")
    wpp_wet = wpp_engine.generate_wpp_field(topology, friction_wet, season="wet")
    deploy_wet = DROOptimizer(topology, friction_wet).solve(wpp_wet)

    shift_norm = float(np.linalg.norm(deploy_dry - deploy_wet))
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2), constrained_layout=True)
    im0 = axes[0].imshow(deploy_dry, cmap="Reds", origin="lower")
    axes[0].set_title("Dry season deployment")
    axes[0].set_xlabel("Grid X")
    axes[0].set_ylabel("Grid Y")
    fig.colorbar(im0, ax=axes[0], label="Staff/cell")

    im1 = axes[1].imshow(deploy_wet, cmap="Reds", origin="lower")
    axes[1].set_title("Wet season deployment")
    axes[1].set_xlabel("Grid X")
    axes[1].set_ylabel("Grid Y")
    fig.colorbar(im1, ax=axes[1], label="Staff/cell")

    delta = deploy_wet - deploy_dry
    im2 = axes[2].imshow(delta, cmap="coolwarm", origin="lower")
    axes[2].set_title(f"Shift map (L2={shift_norm:.2f})")
    axes[2].set_xlabel("Grid X")
    axes[2].set_ylabel("Grid Y")
    fig.colorbar(im2, ax=axes[2], label="Wet - Dry")

    save_path = output_dir / "seasonal_comparison.png"
    save_academic_figure(fig, save_path)
    return {"seasonal_shift_l2": shift_norm}


def run_tech_degradation(
    topology: EtoshaTopology, wpp_engine: WPPEngine, output_dir: Path
) -> Dict[str, float]:
    """Evaluates interception loss from lambda degradation 0.25 -> 0.08."""
    apply_academic_style()
    params = topology.specs["model_parameters"]
    lambda_full = float(params["lambda_full_tech"])
    lambda_low = float(params["lambda_low_tech"])

    friction = FrictionMatrix(topology).generate_friction_map(season="dry")
    wpp = wpp_engine.generate_wpp_field(topology, friction, season="dry")
    base_optimizer = DROOptimizer(topology, friction, lambda_detection=lambda_full)
    allocation = base_optimizer.solve(wpp)

    coverage_sat = np.ones_like(wpp, dtype=float)
    coverage_uav = np.clip(1.0 - friction / np.max(friction), 0.2, 0.95)
    p_full = base_optimizer.interception_probability(
        allocation, friction, p_sat=0.33, p_uav=0.50, coverage_sat=coverage_sat, coverage_uav=coverage_uav
    )
    low_optimizer = DROOptimizer(topology, friction, lambda_detection=lambda_low)
    p_low = low_optimizer.interception_probability(
        allocation, friction, p_sat=0.30, p_uav=0.0, coverage_sat=coverage_sat, coverage_uav=np.zeros_like(coverage_uav)
    )

    weighted_full = float(np.average(p_full, weights=wpp))
    weighted_low = float(np.average(p_low, weights=wpp))
    loss_pct = float(100.0 * (weighted_full - weighted_low) / max(weighted_full, 1e-9))

    fig, ax = plt.subplots(figsize=(9, 6.4), constrained_layout=True)
    ax.bar(["Full tech", "Ground only"], [weighted_full, weighted_low], color=["#2E86AB", "#CC4E5C"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Weighted interception probability")
    ax.set_title(f"Technology degradation loss = {loss_pct:.2f}%")
    save_path = output_dir / "tech_degradation.png"
    save_academic_figure(fig, save_path)

    return {
        "intercept_full_tech": weighted_full,
        "intercept_low_tech": weighted_low,
        "tech_loss_pct": loss_pct,
    }


def run_model_evaluation_suite(
    topology: EtoshaTopology,
    wpp_engine: WPPEngine,
    red_line_staff: float,
    output_dir: Path,
) -> Dict[str, float]:
    """Runs baseline and stress-test metrics, then saves JSON and CSV."""
    friction = FrictionMatrix(topology).generate_friction_map(season="dry")
    wpp = wpp_engine.generate_wpp_field(topology, friction, season="dry")
    deploy, diag = DROOptimizer(topology, friction).solve(wpp, return_diagnostics=True)

    staff_total = float(np.sum(deploy))
    utilization = float(staff_total / float(topology.specs["resources"]["total_personnel"]))
    weighted_capture = float(np.sum(wpp * (1.0 - np.exp(-0.25 * deploy / np.maximum(friction, 1e-9)))))

    shortages = np.array([0.05, 0.10, 0.20, 0.30], dtype=float)
    scenario_scores: Dict[str, float] = {}
    for s in shortages:
        scaled = deploy * (1.0 - s)
        score = float(np.sum(wpp * (1.0 - np.exp(-0.25 * scaled / np.maximum(friction, 1e-9)))))
        scenario_scores[f"shortage_{int(100 * s)}pct"] = score

    eval_summary = {
        "baseline_weighted_capture": weighted_capture,
        "manpower_utilization": utilization,
        "solver_success": diag["success"],
        "solver_iterations": diag["iterations"],
        "tau_current_staff": float(topology.specs["resources"]["total_personnel"]),
        "tau_critical_staff": float(red_line_staff),
        "safety_margin_staff": float(topology.specs["resources"]["total_personnel"] - red_line_staff),
        **scenario_scores,
    }

    mc_summary, mc_samples = run_monte_carlo_robustness(
        deployment=deploy,
        wpp_field=wpp,
        friction_field=friction,
        n_runs=1000,
        hotspot_sigma=0.12,
        friction_sigma=0.08,
        lambda_detection=0.25,
        seed=2026,
    )
    fair_summary, fair_details = compute_fairness_metrics(deploy)
    marginal_summary, marginal_curve = compute_marginal_utility_curve(
        wpp_field=wpp,
        friction_field=friction,
        total_staff=int(topology.specs["resources"]["total_personnel"]),
        lambda_detection=0.25,
    )
    pareto_summary, pareto_points = compute_pareto_front(
        wpp_field=wpp,
        friction_field=friction,
        total_staff=int(topology.specs["resources"]["total_personnel"]),
        lambda_detection=0.25,
    )
    sensitivity_summary, sensitivity_details = compute_global_sensitivity(
        deployment=deploy,
        wpp_field=wpp,
        friction_field=friction,
        lambda_detection=0.25,
        delta=0.10,
    )
    voronoi_summary, voronoi_details = compute_voronoi_zone_metrics(
        deployment=deploy,
        wpp_field=wpp,
        active_threshold=0.5,
    )

    eval_summary.update(mc_summary)
    eval_summary.update(fair_summary)
    eval_summary.update(marginal_summary)
    eval_summary.update(pareto_summary)
    eval_summary.update(sensitivity_summary)
    eval_summary.update(voronoi_summary)

    baseline_interception = weighted_interception_rate(deploy, wpp, friction, lambda_detection=0.25)
    shortage_20_deploy = deploy * 0.8
    shortage_interception = weighted_interception_rate(shortage_20_deploy, wpp, friction, lambda_detection=0.25)
    lowtech_interception = weighted_interception_rate(deploy, wpp, friction, lambda_detection=0.08)
    stability_margin_norm = float(
        np.clip((float(topology.specs["resources"]["total_personnel"]) - red_line_staff) / 120.0, 0.0, 1.0)
    )
    eval_summary.update(
        {
            "radar_baseline_interception": baseline_interception,
            "radar_shortage20_interception": shortage_interception,
            "radar_lowtech_interception": lowtech_interception,
            "radar_stability_margin_norm": stability_margin_norm,
        }
    )

    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(eval_summary, file_obj, indent=2)

    with (output_dir / "evaluation_table.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["metric", "value"])
        for key, value in eval_summary.items():
            writer.writerow([key, value])

    with (output_dir / "monte_carlo_robustness.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "summary": mc_summary,
                "capture_samples": mc_samples["capture_samples"].tolist(),
                "interception_samples": mc_samples["interception_samples"].tolist(),
            },
            file_obj,
            indent=2,
        )

    with (output_dir / "pareto_front.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["staff", "capture", "fairness"])
        for s, c, f in zip(pareto_points["staff"], pareto_points["capture"], pareto_points["fairness"]):
            writer.writerow([float(s), float(c), float(f)])

    with (output_dir / "sensitivity_analysis.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "summary": sensitivity_summary,
                "parameter_names": sensitivity_details["parameter_names"].tolist(),
                "elasticity": sensitivity_details["elasticity"].tolist(),
                "capture_up": sensitivity_details["capture_up"].tolist(),
                "capture_down": sensitivity_details["capture_down"].tolist(),
            },
            file_obj,
            indent=2,
        )

    with (output_dir / "advanced_evaluation_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(eval_summary, file_obj, indent=2)

    plot_lorenz_curve(
        lorenz_x=fair_details["lorenz_x"],
        lorenz_y=fair_details["lorenz_y"],
        gini_value=float(fair_summary["deployment_gini"]),
        output_dir=output_dir,
    )
    plot_pareto_front(
        pareto_staff=pareto_points["staff"],
        pareto_capture=pareto_points["capture"],
        pareto_fairness=pareto_points["fairness"],
        output_dir=output_dir,
    )
    plot_marginal_utility_curve(
        staff_axis=marginal_curve["staff_axis"],
        marginal_gain_curve=marginal_curve["marginal_gain_curve"],
        output_dir=output_dir,
    )
    plot_voronoi_patrol_zones(
        zone_map=voronoi_details["zone_map"],
        active_coords=voronoi_details.get("active_coords", np.zeros((0, 2), dtype=float)),
        output_dir=output_dir,
    )

    return eval_summary