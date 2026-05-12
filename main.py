"""Three-layer anti-poaching patrol allocation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from global_adaption import run_global_adaptation
from run_enhancements import run_enhancements
from scenario_runner import run_model_evaluation_suite, run_season_comparison, run_tech_degradation
from src.advanced_analytics import compute_fairness_metrics, weighted_capture_score, weighted_interception_rate
from src.jacobian_stability import StabilityEngine
from src.rhc_temporal_shift import TemporalRHC
from src.ssg_dro_optimizer import DROOptimizer
from src.wpp_maut_engine import WPPEngine
from utils.spatial_friction import FrictionMatrix
from utils.topology_constructor import EtoshaTopology
from visualizations.bifurcation_plots import plot_stability_bifurcation
from visualizations.paper_figures import (
    plot_3d_topology_wpp_surface,
    plot_multi_metric_radar,
    plot_phase_space_portrait,
)
from visualizations.spatial_heatmaps import render_protection_strategy


def _build_forecast_multipliers(shape: tuple[int, int]) -> list[np.ndarray]:
    """Creates deterministic horizon multipliers for RHC."""
    yy, xx = np.indices(shape, dtype=float)
    center_y = 0.5 * shape[0]
    center_x = 0.5 * shape[1]
    radial = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    normalized = radial / max(float(np.max(radial)), 1e-9)
    return [
        1.00 - 0.05 * normalized,
        1.05 - 0.03 * normalized,
        1.10 - 0.02 * normalized,
        1.12 - 0.01 * normalized,
        1.08 - 0.01 * normalized,
        1.04 - 0.01 * normalized,
    ]


def run_pipeline(mode: str = "all") -> Dict[str, float]:
    """Runs selected pipeline mode and writes outputs to `outputs/`.

    Args:
        mode: One of `baseline`, `seasonal`, `tech`, `global`, `all`.
    """
    project_root = Path(__file__).parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = EtoshaTopology(config_path=str(project_root / "data" / "raw_pdf_specs.json"))
    wpp_engine = WPPEngine.from_specs(topology.specs)
    metrics: Dict[str, float] = {}

    if mode in {"baseline", "all", "seasonal", "tech"}:
        friction = FrictionMatrix(topology).generate_friction_map(season="dry")
        wpp = wpp_engine.generate_wpp_field(topology, friction, season="dry")

        rhc = TemporalRHC(horizon_steps=6, discount=0.95)
        forecast = _build_forecast_multipliers(wpp.shape)
        wpp_rhc = rhc.aggregate_forecast(wpp, forecast)

        optimizer = DROOptimizer(topology, friction)
        deployment = optimizer.solve(wpp_rhc)
        render_protection_strategy(wpp_rhc, deployment, topology.waterhole_coords)

        stability = StabilityEngine(area_km2=topology.total_area_km2)
        red_line = stability.find_red_line()
        staff_grid = np.linspace(100.0, 400.0, 150)
        eig_curve = np.array([stability.dominant_real_eigenvalue(s, np.array([500.0, 18.0, 12.0])) for s in staff_grid])
        plot_stability_bifurcation(
            staff_range=staff_grid,
            stability_scores=eig_curve,
            threshold=red_line,
            current_staff=int(topology.specs["resources"]["total_personnel"]),
        )
        metrics["critical_staff_threshold"] = float(red_line)

    if mode in {"seasonal", "all"}:
        metrics.update(run_season_comparison(topology, wpp_engine, output_dir))

    if mode in {"tech", "all"}:
        metrics.update(run_tech_degradation(topology, wpp_engine, output_dir))

    if mode in {"enhanced", "all"}:
        metrics.update(run_enhancements(topology, wpp_engine, output_dir))

    if mode in {"global", "all"}:
        metrics.update(run_global_adaptation(topology, output_dir))

    if mode in {"all"}:
        red_line = metrics.get("critical_staff_threshold", 0.0)
        eval_metrics = run_model_evaluation_suite(topology, wpp_engine, red_line, output_dir)
        metrics.update(eval_metrics)

        # Publication-grade baseline dynamics and topology visualizations.
        friction = FrictionMatrix(topology).generate_friction_map(season="dry")
        wpp = wpp_engine.generate_wpp_field(topology, friction, season="dry")
        deployment = DROOptimizer(topology, friction).solve(wpp)
        stability = StabilityEngine(area_km2=topology.total_area_km2)
        plot_phase_space_portrait(
            stability_engine=stability,
            output_dir=output_dir,
            stable_staff=float(topology.specs["resources"]["total_personnel"]),
            collapse_staff=max(40.0, 0.8 * float(red_line)),
        )
        plot_3d_topology_wpp_surface(
            topology_mask=topology.topology_mask,
            wpp_field=wpp,
            deployment_map=deployment,
            output_dir=output_dir,
        )

        lambda_full = float(topology.specs["model_parameters"]["lambda_full_tech"])
        lambda_low = float(topology.specs["model_parameters"]["lambda_low_tech"])
        staff_full = int(topology.specs["resources"]["total_personnel"])
        staff_m20 = int(np.floor(0.8 * staff_full))
        red_line = float(eval_metrics["tau_critical_staff"])

        baseline_opt = DROOptimizer(topology, friction, lambda_detection=lambda_full)
        deploy_baseline = baseline_opt.solve(wpp)

        manpower_opt = DROOptimizer(topology, friction, lambda_detection=lambda_full)
        manpower_opt.staff_limit = staff_m20
        deploy_m20 = manpower_opt.solve(wpp)

        techfail_opt = DROOptimizer(topology, friction, lambda_detection=lambda_low)
        deploy_techfail = techfail_opt.solve(wpp)

        # Core performance values.
        inter_base_full = weighted_interception_rate(deploy_baseline, wpp, friction, lambda_detection=lambda_full)
        inter_base_low = weighted_interception_rate(deploy_baseline, wpp, friction, lambda_detection=lambda_low)
        inter_m20_full = weighted_interception_rate(deploy_m20, wpp, friction, lambda_detection=lambda_full)
        inter_m20_low = weighted_interception_rate(deploy_m20, wpp, friction, lambda_detection=lambda_low)
        inter_tech_low = weighted_interception_rate(deploy_techfail, wpp, friction, lambda_detection=lambda_low)
        inter_tech_full = weighted_interception_rate(deploy_techfail, wpp, friction, lambda_detection=lambda_full)

        cap_base = weighted_capture_score(deploy_baseline, wpp, friction, lambda_detection=lambda_full)
        cap_m20 = weighted_capture_score(deploy_m20, wpp, friction, lambda_detection=lambda_full)
        cap_tech = weighted_capture_score(deploy_techfail, wpp, friction, lambda_detection=lambda_low)

        fairness_base = 1.0 - float(compute_fairness_metrics(deploy_baseline)[0]["deployment_gini"])
        fairness_m20 = 1.0 - float(compute_fairness_metrics(deploy_m20)[0]["deployment_gini"])
        fairness_tech = 1.0 - float(compute_fairness_metrics(deploy_techfail)[0]["deployment_gini"])

        def _stability_index(staff_count: float, interception_ratio: float) -> float:
            # Staff margin is the structural component; interception keeps stability tied to operational capability.
            denom = max(float(staff_full) - red_line, 1e-9)
            staff_margin = float(np.clip((staff_count - red_line) / denom, 0.0, 1.0))
            return float(np.sqrt(staff_margin * np.clip(interception_ratio, 0.0, 1.0)))

        stab_base = _stability_index(float(staff_full), 1.0)
        stab_m20 = _stability_index(float(staff_m20), inter_m20_full / max(inter_base_full, 1e-9))
        stab_tech = _stability_index(float(staff_full), inter_tech_low / max(inter_base_full, 1e-9))

        capture_upper_bound = max(float(np.sum(wpp)), 1e-9)

        # Absolute 0-1 scores (higher is better).
        scenario_metrics_abs = {
            "Baseline": {
                "interception_rate": float(np.clip(inter_base_full, 0.0, 1.0)),
                "wpp_coverage": float(np.clip(cap_base / capture_upper_bound, 0.0, 1.0)),
                "stability_margin_norm": float(np.clip(stab_base, 0.0, 1.0)),
                "low_tech_resilience": float(np.clip(inter_base_low / max(inter_base_full, 1e-9), 0.0, 1.0)),
                "fairness": float(np.clip(fairness_base, 0.0, 1.0)),
            },
            "Manpower -20%": {
                "interception_rate": float(np.clip(inter_m20_full, 0.0, 1.0)),
                "wpp_coverage": float(np.clip(cap_m20 / capture_upper_bound, 0.0, 1.0)),
                "stability_margin_norm": float(np.clip(stab_m20, 0.0, 1.0)),
                "low_tech_resilience": float(np.clip(inter_m20_low / max(inter_m20_full, 1e-9), 0.0, 1.0)),
                "fairness": float(np.clip(fairness_m20, 0.0, 1.0)),
            },
            "Tech failure": {
                "interception_rate": float(np.clip(inter_tech_low, 0.0, 1.0)),
                "wpp_coverage": float(np.clip(cap_tech / capture_upper_bound, 0.0, 1.0)),
                "stability_margin_norm": float(np.clip(stab_tech, 0.0, 1.0)),
                "low_tech_resilience": float(np.clip(inter_tech_low / max(inter_tech_full, 1e-9), 0.0, 1.0)),
                "fairness": float(np.clip(fairness_tech, 0.0, 1.0)),
            },
        }

        metric_keys = [
            "interception_rate",
            "wpp_coverage",
            "stability_margin_norm",
            "low_tech_resilience",
            "fairness",
        ]
        baseline_abs = scenario_metrics_abs["Baseline"]
        scenario_metrics_rel: Dict[str, Dict[str, float]] = {}
        for scenario_name, values in scenario_metrics_abs.items():
            scenario_metrics_rel[scenario_name] = {}
            for key in metric_keys:
                ratio = values[key] / max(baseline_abs[key], 1e-9)
                scenario_metrics_rel[scenario_name][key] = float(np.clip(ratio, 0.0, 1.2))

        plot_multi_metric_radar(
            scenario_metrics_abs=scenario_metrics_abs,
            scenario_metrics_rel=scenario_metrics_rel,
            output_dir=output_dir,
        )

    with (output_dir / "run_metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)
    return metrics


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anti-Poaching Patrol Framework — Etosha Case Study")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["baseline", "seasonal", "tech", "global", "enhanced", "all"],
        help="Pipeline mode selector.",
    )
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    results = run_pipeline(mode=args.mode)
    print("Pipeline completed.")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
