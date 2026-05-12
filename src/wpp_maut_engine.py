"""White-box NHPP + MAUT engine for WPP construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from utils.topology_constructor import EtoshaTopology


@dataclass(frozen=True)
class UtilityConfig:
    """Utility and threat model coefficients."""

    alpha: float
    kappa: float
    n_crit: float
    n_initial_endangered: float
    n_initial_abundant: float
    nhpp_beta: NDArray[np.float64]
    nhpp_lambda0_dry: float
    nhpp_lambda0_wet: float


class WPPEngine:
    """Computes Wildlife Protection Potential (WPP) in transparent form.

    The engine combines:

    1) Nonlinear MAUT:
       - Endangered species utility
         $U_R(N)=1-\\exp(-\\alpha\\max(N-N_{crit},0)/N_{init,R})$
       - Abundant species utility
         $U_E(N)=\\ln(1+\\kappa N)/\\ln(1+\\kappa N_{init,E})$

    2) NHPP threat intensity:
       $\\lambda(s,t)=\\lambda_0(t)\\exp(\\beta^\\top X(s,t))$

    3) Waterhole attraction:
       $\\sum_h 10\\exp(-0.5 d_h(s))$

    The final field is a weighted, nonnegative combination of ecological value
    and threat intensity.
    """

    def __init__(self, config: UtilityConfig) -> None:
        self.config = config

    @classmethod
    def from_specs(cls, specs: Dict) -> "WPPEngine":
        """Builds engine from JSON specs dictionary."""
        params = specs["model_parameters"]
        config = UtilityConfig(
            alpha=float(params["alpha"]),
            kappa=float(params["kappa"]),
            n_crit=float(params["n_crit"]),
            n_initial_endangered=float(params["n_initial_endangered"]),
            n_initial_abundant=float(params["n_initial_abundant"]),
            nhpp_beta=np.asarray(params["nhpp_beta"], dtype=float),
            nhpp_lambda0_dry=float(params["nhpp_lambda0_dry"]),
            nhpp_lambda0_wet=float(params["nhpp_lambda0_wet"]),
        )
        return cls(config=config)

    def u_endangered(self, population: float) -> float:
        """Returns endangered utility $U_R$."""
        surplus = max(population - self.config.n_crit, 0.0)
        value = 1.0 - np.exp(-self.config.alpha * surplus / self.config.n_initial_endangered)
        return float(value)

    def u_abundant(self, population: float) -> float:
        """Returns abundant utility $U_E$."""
        numerator = np.log(1.0 + self.config.kappa * population)
        denominator = np.log(1.0 + self.config.kappa * self.config.n_initial_abundant)
        return float(numerator / max(denominator, 1e-9))

    def _distance_to_feature(
        self, grid_shape: Tuple[int, int], feature_coords: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Returns per-cell nearest Euclidean distance to feature set."""
        yy, xx = np.indices(grid_shape, dtype=float)
        fy = feature_coords[:, 0][:, None, None]
        fx = feature_coords[:, 1][:, None, None]
        dist = np.sqrt((yy[None, :, :] - fy) ** 2 + (xx[None, :, :] - fx) ** 2)
        return np.min(dist, axis=0)

    def generate_nhpp_intensity(
        self,
        topology: EtoshaTopology,
        friction_grid: NDArray[np.float64],
        season: str,
    ) -> NDArray[np.float64]:
        """Builds NHPP intensity field from transparent covariates.

        Covariates $X(s,t)$ used here:
        - distance to nearest waterhole,
        - distance to border (edge proxy),
        - distance to road proxy (centerlines),
        - local friction.
        """

        grid_shape = friction_grid.shape
        coords = np.asarray(topology.waterhole_coords, dtype=np.int64)
        d_water = self._distance_to_feature(grid_shape, coords)

        yy, xx = np.indices(grid_shape, dtype=float)
        max_y, max_x = grid_shape[0] - 1, grid_shape[1] - 1
        d_edge = np.minimum.reduce([yy, xx, max_y - yy, max_x - xx])

        road_y = np.array([grid_shape[0] * 0.30, grid_shape[0] * 0.70], dtype=float)
        d_road = np.min(np.abs(yy[:, :, None] - road_y[None, None, :]), axis=2)

        x_stack = np.stack([d_water, d_edge, d_road, friction_grid], axis=0)
        beta = self.config.nhpp_beta[:, None, None]
        linear_part = np.sum(beta * x_stack, axis=0)

        lambda0 = self.config.nhpp_lambda0_dry if season.lower() == "dry" else self.config.nhpp_lambda0_wet
        intensity = lambda0 * np.exp(linear_part)
        return np.maximum(intensity, 1e-9)

    def generate_wpp_field(
        self,
        topology: EtoshaTopology,
        friction_grid: NDArray[np.float64],
        season: str,
        endangered_population: float = 300.0,
        abundant_population: float = 2500.0,
    ) -> NDArray[np.float64]:
        """Generates the final WPP field.

        Args:
            topology: Spatial topology object.
            friction_grid: Terrain friction map.
            season: `"dry"` or `"wet"` for NHPP baseline.
            endangered_population: Current endangered population state.
            abundant_population: Current abundant population state.

        Returns:
            Nonnegative WPP field with same grid shape as topology.
        """

        grid_shape = topology.topology_mask.shape
        yy, xx = np.indices(grid_shape, dtype=float)
        hole_coords = np.asarray(topology.waterhole_coords, dtype=np.float64)
        hy = hole_coords[:, 0][:, None, None]
        hx = hole_coords[:, 1][:, None, None]

        distances = np.sqrt((yy[None, :, :] - hy) ** 2 + (xx[None, :, :] - hx) ** 2)
        attraction = np.sum(10.0 * np.exp(-0.5 * distances), axis=0)

        endangered_weight = self.u_endangered(endangered_population)
        abundant_weight = self.u_abundant(abundant_population)
        utility_weight = 0.65 * endangered_weight + 0.35 * abundant_weight

        nhpp_intensity = self.generate_nhpp_intensity(topology, friction_grid, season)
        wpp_field = utility_weight * attraction * (1.0 + nhpp_intensity)
        return np.maximum(wpp_field, 0.0)