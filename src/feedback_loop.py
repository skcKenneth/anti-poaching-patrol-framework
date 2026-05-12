"""Self-consistent iterative feedback loop connecting Layers 1-2-3.

The three-layer pipeline is run iteratively until the patrol deployment
converges. At each iteration:

  1. Layer 1: generate WPP field from current threat intensity.
  2. Layer 2: solve SLSQP patrol allocation.
  3. Layer 3: integrate the 3D ODE to equilibrium; extract Z* (poacher
     activity at equilibrium given this deployment).
  4. Update the NHPP baseline intensity proportional to Z* / Z*_ref,
     reflecting that a high-equilibrium poacher activity implies a more
     threatening landscape.
  5. Repeat until the relative change in deployment falls below `tol`.

This produces a *self-consistent* solution: the spatial deployment is
optimal given the threat landscape, and the threat landscape is
consistent with the population dynamics induced by that deployment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import odeint

from src.jacobian_stability import StabilityEngine
from src.ssg_dro_optimizer import DROOptimizer
from src.wpp_maut_engine import WPPEngine
from utils.spatial_friction import FrictionMatrix
from utils.topology_constructor import EtoshaTopology


@dataclass
class ConvergenceRecord:
    """Stores per-iteration convergence diagnostics."""

    relative_change: List[float] = field(default_factory=list)
    z_star: List[float] = field(default_factory=list)
    objective: List[float] = field(default_factory=list)
    lambda0_dry: List[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return len(self.relative_change) > 0 and self.relative_change[-1] < 1e-3

    @property
    def n_iterations(self) -> int:
        return len(self.relative_change)


def _integrate_to_equilibrium(
    stability: StabilityEngine,
    patrol_density: float,
    state0: NDArray[np.float64],
    t_end: float = 400.0,
    n_steps: int = 2000,
) -> NDArray[np.float64]:
    """Integrates the 3D ODE system to approximate steady state.

    Args:
        stability: Configured StabilityEngine instance.
        patrol_density: Rangers per km² (scalar, spatially averaged).
        state0: Initial state vector [N, P, Z].
        t_end: Integration horizon (years).
        n_steps: Number of time steps.

    Returns:
        Final state vector [N*, P*, Z*].
    """
    t_span = np.linspace(0.0, t_end, n_steps)

    def _rhs(state: NDArray[np.float64], _t: float) -> NDArray[np.float64]:
        return stability.rhs(state, patrol_density=patrol_density)

    solution = odeint(_rhs, state0, t_span, rtol=1e-6, atol=1e-8)
    return solution[-1]


def run_feedback_loop(
    topology: EtoshaTopology,
    wpp_engine: WPPEngine,
    season: str = "dry",
    max_iter: int = 20,
    tol: float = 1e-3,
    z_ref: float = 12.0,
    state0: Tuple[float, float, float] = (500.0, 18.0, 12.0),
    occurrence_prior: NDArray[np.float64] | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], ConvergenceRecord]:
    """Runs the self-consistent feedback loop to convergence.

    Args:
        topology: Etosha spatial topology.
        wpp_engine: Initial WPP engine (unmodified).
        season: Season for friction and NHPP baseline.
        max_iter: Maximum number of outer iterations.
        tol: Convergence threshold on relative deployment change.
        z_ref: Reference poacher activity (initial equilibrium Z₀).
        state0: Initial ODE state [N₀, P₀, Z₀].
        occurrence_prior: Optional GBIF-derived occurrence density grid;
            if provided, used to weight the WPP field in each iteration.

    Returns:
        Tuple of (final_deployment, final_wpp_field, convergence_record).
    """
    friction = FrictionMatrix(topology).generate_friction_map(season=season)
    stability = StabilityEngine(area_km2=topology.total_area_km2)
    record = ConvergenceRecord()

    # --- Iteration 0: initial solve ---
    current_engine = wpp_engine
    wpp = current_engine.generate_wpp_field(
        topology, friction, season=season,
        endangered_population=float(state0[0]),
        abundant_population=2500.0,
    )
    if occurrence_prior is not None:
        wpp = wpp * (1.0 + occurrence_prior / max(float(occurrence_prior.max()), 1e-9))

    deployment = DROOptimizer(topology, friction).solve(wpp)
    current_specs = copy.deepcopy(topology.specs)
    lambda0_base_dry = float(current_specs["model_parameters"]["nhpp_lambda0_dry"])
    lambda0_base_wet = float(current_specs["model_parameters"]["nhpp_lambda0_wet"])

    for iteration in range(max_iter):
        prev_deployment = deployment.copy()

        # --- Layer 3: integrate ODE to equilibrium ---
        total_staff = float(np.sum(deployment))
        patrol_density = total_staff / float(topology.total_area_km2)
        eq_state = _integrate_to_equilibrium(
            stability, patrol_density, np.array(state0, dtype=float)
        )
        z_star = float(max(eq_state[2], 0.0))

        # --- Update threat intensity proportional to Z* ---
        # A higher equilibrium poacher activity → higher NHPP baseline.
        threat_scale = float(np.clip(z_star / max(z_ref, 1e-9), 0.1, 5.0))
        current_specs["model_parameters"]["nhpp_lambda0_dry"] = (
            lambda0_base_dry * threat_scale
        )
        current_specs["model_parameters"]["nhpp_lambda0_wet"] = (
            lambda0_base_wet * threat_scale
        )

        # --- Layer 1: regenerate WPP with updated threat ---
        current_engine = WPPEngine.from_specs(current_specs)
        wpp = current_engine.generate_wpp_field(
            topology, friction, season=season,
            endangered_population=float(eq_state[0]),
            abundant_population=2500.0,
        )
        if occurrence_prior is not None:
            wpp = wpp * (1.0 + occurrence_prior / max(float(occurrence_prior.max()), 1e-9))
        wpp = np.maximum(wpp, 0.0)

        # --- Layer 2: re-solve allocation ---
        deployment = DROOptimizer(topology, friction).solve(wpp)

        # --- Convergence diagnostics ---
        denom = float(np.linalg.norm(prev_deployment))
        delta = float(np.linalg.norm(deployment - prev_deployment)) / max(denom, 1e-9)
        objective = float(np.sum(
            (1.0 - np.exp(-0.25 * deployment / np.maximum(friction, 1e-9))) * wpp
        ))

        record.relative_change.append(delta)
        record.z_star.append(z_star)
        record.objective.append(objective)
        record.lambda0_dry.append(current_specs["model_parameters"]["nhpp_lambda0_dry"])

        if delta < tol:
            break

    return deployment, wpp, record


def feedback_loop_summary(record: ConvergenceRecord) -> Dict[str, float]:
    """Returns scalar summary metrics from a convergence record."""
    return {
        "feedback_n_iterations": float(record.n_iterations),
        "feedback_converged": float(record.converged),
        "feedback_final_delta": float(record.relative_change[-1]) if record.relative_change else 0.0,
        "feedback_final_z_star": float(record.z_star[-1]) if record.z_star else 0.0,
        "feedback_z_star_change": float(
            abs(record.z_star[-1] - record.z_star[0]) / max(record.z_star[0], 1e-9)
        ) if len(record.z_star) > 1 else 0.0,
        "feedback_final_objective": float(record.objective[-1]) if record.objective else 0.0,
    }
