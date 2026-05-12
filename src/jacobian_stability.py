"""Jacobian-based stability analysis for ecological security dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StabilityParams:
    """Parameters of 3D prey-predator-poacher dynamics calibrated for Etosha."""

    r_prey: float = 0.18
    k_prey: float = 1800.0
    alpha_predation: float = 0.0055
    alpha_poaching: float = 0.0060
    eta_conversion: float = 0.00015 
    m_predator: float = 0.09
    rho_poacher_growth: float = 0.010
    sigma_suppression: float = 450.0 
    m_poacher: float = 0.06


class StabilityEngine:
    """Computes manpower red-line from Jacobian eigenvalue crossing.

    ODE system:

    $$
    \\dot N = rN(1-N/K)-\\alpha NP-\\alpha_p NZ
    $$
    $$
    \\dot P = \\eta NP-mP
    $$
    $$
    \\dot Z = \\rho NZ-\\sigma\\tau Z-m_z Z
    $$

    where $\\tau$ is patrol density (staff per km^2).
    """

    def __init__(self, area_km2: float, params: StabilityParams | None = None) -> None:
        self.area_km2 = float(area_km2)
        self.params = params if params is not None else StabilityParams()

    def rhs(self, state: NDArray[np.float64], patrol_density: float) -> NDArray[np.float64]:
        """Returns ODE derivatives for state = [prey, predator, poachers]."""
        prey, predator, poachers = state
        p = self.params

        d_prey = (
            p.r_prey * prey * (1.0 - prey / p.k_prey)
            - p.alpha_predation * prey * predator
            - p.alpha_poaching * prey * poachers
        )
        d_predator = p.eta_conversion * prey * predator - p.m_predator * predator
        d_poachers = (
            p.rho_poacher_growth * prey * poachers
            - p.sigma_suppression * patrol_density * poachers
            - p.m_poacher * poachers
        )
        return np.array([d_prey, d_predator, d_poachers], dtype=float)

    def jacobian(
        self, prey: float, predator: float, poachers: float, patrol_density: float
    ) -> NDArray[np.float64]:
        """Manually derived Jacobian matrix of the 3D ODE system."""
        p = self.params
        j = np.zeros((3, 3), dtype=float)

        # Partial derivatives for d_prey / d(*)
        j[0, 0] = p.r_prey * (1.0 - 2.0 * prey / p.k_prey) - p.alpha_predation * predator - p.alpha_poaching * poachers
        j[0, 1] = -p.alpha_predation * prey
        j[0, 2] = -p.alpha_poaching * prey

        # Partial derivatives for d_predator / d(*)
        j[1, 0] = p.eta_conversion * predator
        j[1, 1] = p.eta_conversion * prey - p.m_predator
        j[1, 2] = 0.0

        # Partial derivatives for d_poachers / d(*)
        j[2, 0] = p.rho_poacher_growth * poachers
        j[2, 1] = 0.0
        j[2, 2] = p.rho_poacher_growth * prey - p.sigma_suppression * patrol_density - p.m_poacher
        return j

    def dominant_real_eigenvalue(self, staff_count: float, state: NDArray[np.float64]) -> float:
        """Returns largest real part of eigenvalues under a staffing level."""
        patrol_density = staff_count / self.area_km2
        jac = self.jacobian(state[0], state[1], state[2], patrol_density)
        eigvals = np.linalg.eigvals(jac)
        return float(np.max(np.real(eigvals)))

    def find_red_line(
        self,
        base_state: Tuple[float, float, float] = (500.0, 18.0, 12.0),
        max_staff: float = 350.0,
        coarse_step: float = 1.0,
    ) -> float:
        """Finds critical staff count where stability changes sign."""
        state = np.asarray(base_state, dtype=float)

        staff = max_staff
        prev_staff = staff
        prev_val = self.dominant_real_eigenvalue(prev_staff, state)
        
        # If the system is fundamentally unstable even at max_staff, raise a warning
        if prev_val > 0.0:
            print("WARNING: System is fundamentally unstable even at max_staff (350). Check parameters!")
            return max_staff

        while staff >= 0.0:
            current_val = self.dominant_real_eigenvalue(staff, state)
            
            # When the eigenvalue crosses from negative (stable) to positive (collapse)
            if current_val > 0.0 and prev_val <= 0.0:
                left, right = staff, prev_staff
                # Binary search to find the most precise decimal threshold
                for _ in range(30):
                    mid = 0.5 * (left + right)
                    mid_val = self.dominant_real_eigenvalue(mid, state)
                    if mid_val > 0.0:
                        left = mid
                    else:
                        right = mid
                return float(right)

            prev_staff, prev_val = staff, current_val
            staff -= coarse_step

        return 0.0