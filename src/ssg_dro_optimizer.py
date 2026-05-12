"""SLSQP resource allocation under white-box interception model."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from utils.topology_constructor import EtoshaTopology


class DROOptimizer:
    """Optimizes ranger deployment with explicit nonlinear objective.

    Core objective (minimization form):

    $$
    \\min_x -\\sum_i \\left(1-\\exp\\left(-\\lambda\\frac{x_i}{\\mu_i}\\right)\\right)WPP_i
    $$

    subject to:
    $\\sum_i x_i = 295$ and $0 \\le x_i \\le 50$.
    """

    def __init__(
        self,
        topology: EtoshaTopology,
        friction_grid: NDArray[np.float64],
        lambda_detection: float = 0.25,
    ) -> None:
        self.topology = topology
        self.friction = friction_grid.astype(float)
        self.lambda_detection = float(lambda_detection)
        self.staff_limit = int(topology.specs["resources"]["total_personnel"])
        self.n_cells = topology.grid_side * topology.grid_side

    def _objective(
        self,
        allocation: NDArray[np.float64],
        wpp_flat: NDArray[np.float64],
        friction_flat: NDArray[np.float64],
    ) -> float:
        """Evaluates objective value for SLSQP."""
        term = 1.0 - np.exp(-self.lambda_detection * allocation / np.maximum(friction_flat, 1e-9))
        return float(-np.sum(term * wpp_flat))

    def interception_probability(
        self,
        allocation: NDArray[np.float64],
        friction_grid: NDArray[np.float64],
        p_sat: float,
        p_uav: float,
        coverage_sat: NDArray[np.float64],
        coverage_uav: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Computes resource-technology synergy interception probability.

        Uses complement product:
        $P=1-\\prod_k(1-P_{detect}^{(k)} I_{coverage}^{(k)})$ with
        $k \\in \\{sat,uav,ranger\\}$.
        """
        ranger_term = 1.0 - np.exp(-self.lambda_detection * allocation / np.maximum(friction_grid, 1e-9))
        sat_term = np.clip(p_sat * coverage_sat, 0.0, 1.0)
        uav_term = np.clip(p_uav * coverage_uav, 0.0, 1.0)
        joint = 1.0 - (1.0 - sat_term) * (1.0 - uav_term) * (1.0 - ranger_term)
        return np.clip(joint, 0.0, 1.0)

    def solve(
        self,
        wpp_field: NDArray[np.float64],
        return_diagnostics: bool = False,
    ) -> NDArray[np.float64] | Tuple[NDArray[np.float64], Dict[str, float]]:
        """Solves the constrained deployment optimization.

        Args:
            wpp_field: Input protection value field.
            return_diagnostics: If true, also returns solver metadata.
        """
        wpp_flat = wpp_field.ravel().astype(float)
        friction_flat = self.friction.ravel().astype(float)
        x0 = np.full(self.n_cells, self.staff_limit / self.n_cells, dtype=float)
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - self.staff_limit},)
        bounds = [(0.0, 50.0)] * self.n_cells

        result = minimize(
            self._objective,
            x0,
            args=(wpp_flat, friction_flat),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-7},
        )
        allocation_map = result.x.reshape(self.friction.shape)
        if not return_diagnostics:
            return allocation_map

        diagnostics = {
            "success": float(1.0 if result.success else 0.0),
            "objective": float(result.fun),
            "iterations": float(result.nit),
            "staff_sum": float(np.sum(result.x)),
        }
        return allocation_map, diagnostics