"""Topology construction for Etosha computational domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


class EtoshaTopology:
    """Constructs a vectorized grid representation of Etosha.

    The spatial domain is discretized into a square grid with cell size
    `res_km`. The central Etosha Pan is approximated as a circle:

    $$
    (x - x_c)^2 + (y - y_c)^2 \\le r^2.
    $$

    Waterholes are sampled deterministically from non-pan cells using a fixed
    seed and no replacement to guarantee exactly 86 unique coordinates.

    Args:
        config_path: Path to `raw_pdf_specs.json`.
        resolution_km: Grid resolution in kilometers per cell.
        seed: Random seed for deterministic waterhole placement.
    """

    def __init__(self, config_path: str, resolution_km: int = 5, seed: int = 2026) -> None:
        config_file = Path(config_path)
        with config_file.open("r", encoding="utf-8") as file_obj:
            self.specs: dict = json.load(file_obj)

        self.total_area_km2: int = int(self.specs["park_identity"]["total_area_km2"])
        self.pan_area_km2: int = int(self.specs["infrastructure"]["etosha_pan"]["area_km2"])
        self.num_waterholes: int = int(self.specs["infrastructure"]["waterholes"]["total_count"])
        self.res_km: int = resolution_km
        self.seed: int = seed

        # Side length chosen from equivalent square area, rounded to nearest cell.
        side_km = float(np.sqrt(self.total_area_km2))
        self.grid_side: int = max(1, int(np.round(side_km / self.res_km)))
        self.topology_mask: NDArray[np.float64] = np.zeros((self.grid_side, self.grid_side), dtype=float)
        self.waterhole_coords: List[Tuple[int, int]] = []

        self._construct_grid(res_km=self.res_km)

    def _construct_grid(self, res_km: int = 5) -> None:
        """Builds pan mask and deterministic waterhole coordinates.

        Args:
            res_km: Resolution in kilometers. Kept explicit for API parity.
        """

        if res_km != self.res_km:
            self.res_km = res_km

        center_y: int = self.grid_side // 2
        center_x: int = self.grid_side // 2
        pan_radius_cells: float = np.sqrt(self.pan_area_km2 / np.pi) / float(self.res_km)

        yy, xx = np.ogrid[: self.grid_side, : self.grid_side]
        pan_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= pan_radius_cells**2
        self.topology_mask.fill(1.0)
        self.topology_mask[pan_mask] = 2.0

        candidate_indices = np.flatnonzero(~pan_mask)
        if candidate_indices.size < self.num_waterholes:
            raise ValueError("Grid too coarse: non-pan cells fewer than required waterholes.")

        rng = np.random.default_rng(self.seed)
        chosen_flat = rng.choice(candidate_indices, size=self.num_waterholes, replace=False)
        wy, wx = np.unravel_index(chosen_flat, pan_mask.shape)
        self.waterhole_coords = [(int(y_i), int(x_i)) for y_i, x_i in zip(wy, wx)]

    def get_waterhole_mask(self) -> NDArray[np.float64]:
        """Returns binary waterhole mask with shape `(grid_side, grid_side)`."""
        mask = np.zeros_like(self.topology_mask, dtype=float)
        if self.waterhole_coords:
            coords = np.asarray(self.waterhole_coords, dtype=int)
            mask[coords[:, 0], coords[:, 1]] = 1.0
        return mask