"""Spatial terrain friction and movement latency utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from utils.topology_constructor import EtoshaTopology


class FrictionMatrix:
    """Constructs terrain friction maps by season.

    Friction coefficients implement physical mobility constraints:

    - Baseline savanna: $\\mu = 1.0$
    - Pan in dry season: $\\mu = 3.0$
    - Pan in wet season: $\\mu = 15.0$

    Args:
        topology: Prepared `EtoshaTopology` object.
    """

    def __init__(self, topology: EtoshaTopology) -> None:
        self.topology = topology
        self.friction_grid: NDArray[np.float64] = np.ones_like(topology.topology_mask, dtype=float)

    def generate_friction_map(self, season: str) -> NDArray[np.float64]:
        """Creates friction map for a target season.

        Args:
            season: Either `"dry"` or `"wet"`.

        Returns:
            2D friction array with same shape as topology mask.

        Raises:
            ValueError: If season is not recognized.
        """

        normalized = season.strip().lower()
        if normalized not in {"dry", "wet"}:
            raise ValueError("season must be either 'dry' or 'wet'.")

        self.friction_grid.fill(1.0)
        pan_mu = 3.0 if normalized == "dry" else 15.0
        self.friction_grid[self.topology.topology_mask == 2.0] = pan_mu
        return self.friction_grid.copy()

    def calculate_travel_time(
        self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]
    ) -> float:
        """Approximates travel latency using Manhattan length and mean friction.

        This simple deterministic proxy keeps evaluation fully white-box.

        Args:
            start_pos: `(y, x)` origin grid index.
            end_pos: `(y, x)` destination grid index.

        Returns:
            Estimated travel-time cost in friction-weighted cell units.
        """

        y1, x1 = start_pos
        y2, x2 = end_pos
        y_min, y_max = sorted((y1, y2))
        x_min, x_max = sorted((x1, x2))
        path_mu = float(np.mean(self.friction_grid[y_min : y_max + 1, x_min : x_max + 1]))
        manhattan = abs(y1 - y2) + abs(x1 - x2)
        return float(manhattan * path_mu)