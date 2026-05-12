"""Runner for the three analytical enhancements.

Invoke directly:
    python run_enhancements.py

Or from main.py with --mode enhanced.

Three contributions:
  1. Self-consistent feedback loop (Layers 1–2–3 iterated to convergence).
  2. Multi-objective Pareto frontier with knee-point identification.
  3. GBIF-anchored WPP field using real black-rhino occurrence records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from data.fetch_real_data import fetch_gbif_occurrences, gbif_summary, get_occurrence_prior
from src.advanced_analytics import (
    compute_fairness_metrics,
    compute_pareto_front,
    weighted_capture_score,
    weighted_interception_rate,
)
from src.feedback_loop import feedback_loop_summary, run_feedback_loop
from src.ssg_dro_optimizer import DROOptimizer
from src.wpp_maut_engine import WPPEngine
from utils.spatial_friction import FrictionMatrix
from utils.topology_constructor import EtoshaTopology
from visualizations.enhanced_figures import (
    plot_deployment_comparison,
    plot_feedback_convergence,
    plot_gbif_wpp_overlay,
    plot_pareto_frontier_enhanced,
)


def run_enhancements(
    topology: EtoshaTopology,
    wpp_engine: WPPEngine,
    output_dir: Path,
    use_gbif_cache: bool = True,
) -> Dict[str, float]:
    """Runs all three enhancements and writes figures + metrics.

    Args:
        topology: Etosha spatial topology.
        wpp_engine: Base WPP engine (unmodified).
        output_dir: Output directory for figures and JSON.
        use_gbif_cache: Use cached GBIF data if available.

    Returns:
        Dictionary of scalar metrics from all three enhancements.
    """
    metrics: Dict[str, float] = {}
    friction = FrictionMatrix(topology).generate_friction_map(season="dry")

    # ── Enhancement 3: GBIF occurrence prior ──────────────────────────────
    print("[Enhancement 3] Fetching GBIF occurrence data...")
    try:
        records = fetch_gbif_occurrences(use_cache=use_gbif_cache)
        summary = gbif_summary(records)
        print(f"  GBIF: {summary['n_total']} total, {summary['n_namibia']} Namibia, "
              f"{summary['n_etosha_bbox']} in Etosha bbox")

        occurrence_prior, n_etosha = get_occurrence_prior(
            grid_side=topology.grid_side, use_cache=use_gbif_cache
        )
        metrics.update({
            "gbif_n_total": float(summary["n_total"]),
            "gbif_n_namibia": float(summary["n_namibia"]),
            "gbif_n_etosha_bbox": float(n_etosha),
        })
    except Exception as exc:
        print(f"  GBIF fetch failed ({exc}); using zero prior.")
        occurrence_prior = np.zeros((topology.grid_side, topology.grid_side), dtype=float)
        n_etosha = 0
        metrics.update({"gbif_n_total": 0.0, "gbif_n_namibia": 0.0, "gbif_n_etosha_bbox": 0.0})

    # Build WPP with and without prior for comparison figure
    wpp_baseline = wpp_engine.generate_wpp_field(topology, friction, season="dry")
    wpp_with_prior = wpp_baseline * (
        1.0 + occurrence_prior / max(float(occurrence_prior.max()), 1e-9)
    )
    wpp_with_prior = np.maximum(wpp_with_prior, 0.0)

    plot_gbif_wpp_overlay(
        wpp_field=wpp_baseline,
        occurrence_prior=occurrence_prior,
        wpp_with_prior=wpp_with_prior,
        output_dir=output_dir,
        n_etosha_records=n_etosha,
    )
    print("  Saved: gbif_wpp_overlay.png")

    # ── Enhancement 1: Self-consistent feedback loop ───────────────────────
    print("[Enhancement 1] Running self-consistent feedback loop...")
    deploy_feedback, wpp_feedback, conv_record = run_feedback_loop(
        topology=topology,
        wpp_engine=wpp_engine,
        season="dry",
        max_iter=20,
        tol=1e-3,
        occurrence_prior=occurrence_prior if occurrence_prior.max() > 0 else None,
    )
    fb_summary = feedback_loop_summary(conv_record)
    metrics.update(fb_summary)
    print(f"  Converged: {conv_record.converged} after {conv_record.n_iterations} iterations")
    print(f"  Final Z*: {fb_summary['feedback_final_z_star']:.3f}")

    plot_feedback_convergence(record=conv_record, output_dir=output_dir)
    print("  Saved: feedback_convergence.png")

    # Open-loop baseline for comparison
    deploy_open_loop = DROOptimizer(topology, friction).solve(wpp_baseline)
    plot_deployment_comparison(
        deploy_open_loop=deploy_open_loop,
        deploy_feedback=deploy_feedback,
        output_dir=output_dir,
    )
    print("  Saved: feedback_deployment_comparison.png")

    # ── Enhancement 2: Multi-objective Pareto frontier ────────────────────
    print("[Enhancement 2] Computing multi-objective Pareto frontier...")
    pareto_summary, pareto_points = compute_pareto_front(
        wpp_field=wpp_baseline,
        friction_field=friction,
        total_staff=int(topology.specs["resources"]["total_personnel"]),
        lambda_detection=float(topology.specs["model_parameters"]["lambda_full_tech"]),
    )
    metrics.update({f"pareto_{k}": float(v) for k, v in pareto_summary.items()})

    # Scalar metrics for baseline and feedback solutions
    baseline_capture = weighted_capture_score(
        deploy_open_loop, wpp_baseline, friction,
        lambda_detection=float(topology.specs["model_parameters"]["lambda_full_tech"]),
    )
    baseline_fairness = 1.0 - float(compute_fairness_metrics(deploy_open_loop)[0]["deployment_gini"])

    feedback_capture = weighted_capture_score(
        deploy_feedback, wpp_feedback, friction,
        lambda_detection=float(topology.specs["model_parameters"]["lambda_full_tech"]),
    )
    feedback_fairness = 1.0 - float(compute_fairness_metrics(deploy_feedback)[0]["deployment_gini"])

    metrics.update({
        "baseline_capture": float(baseline_capture),
        "baseline_fairness": float(baseline_fairness),
        "feedback_capture": float(feedback_capture),
        "feedback_fairness": float(feedback_fairness),
    })

    plot_pareto_frontier_enhanced(
        pareto_points=pareto_points,
        baseline_capture=baseline_capture,
        baseline_fairness=baseline_fairness,
        output_dir=output_dir,
        feedback_capture=feedback_capture,
        feedback_fairness=feedback_fairness,
    )
    print("  Saved: pareto_frontier_enhanced.png")

    # Save enhancement metrics
    with (output_dir / "enhancement_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = EtoshaTopology(config_path=str(project_root / "data" / "raw_pdf_specs.json"))
    wpp_engine = WPPEngine.from_specs(topology.specs)

    results = run_enhancements(topology, wpp_engine, output_dir)
    print("\n=== Enhancement metrics ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
