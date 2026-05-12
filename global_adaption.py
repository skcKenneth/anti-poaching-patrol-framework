"""Cross-region adaptation simulations for Requirement 6."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.ssg_dro_optimizer import DROOptimizer
from utils.topology_constructor import EtoshaTopology
from visualizations.style_guide import apply_academic_style, save_academic_figure


def simulate_congo(topology: EtoshaTopology) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulates Congo rainforest with uniformly high friction.

    Ecological interpretation:
    - Mobility is globally constrained ($\\mu=8.0$).
    - WPP is spatially diffuse and random-uniform.
    """
    grid_shape = topology.topology_mask.shape
    friction_congo = np.full(grid_shape, 8.0, dtype=float)
    rng = np.random.default_rng(101)
    wpp_congo = rng.uniform(1.0, 5.0, size=grid_shape)
    deploy_congo = DROOptimizer(topology, friction_congo).solve(wpp_congo)
    return deploy_congo, wpp_congo


def simulate_himalayas(topology: EtoshaTopology) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulates Himalayas with low-friction corridors and high barriers."""
    n = topology.grid_side
    yy, xx = np.indices((n, n))
    corridor_mask = np.abs(yy - xx) <= 3
    friction_himalayas = np.where(corridor_mask, 2.0, 50.0).astype(float)

    wpp_himalayas = np.where(corridor_mask, 15.0, 1.0).astype(float)
    deploy_himalayas = DROOptimizer(topology, friction_himalayas).solve(wpp_himalayas)
    return deploy_himalayas, corridor_mask.astype(float)


def run_global_adaptation(topology: EtoshaTopology, output_dir: Path) -> Dict[str, float]:
    """Runs cross-continental adaptation and saves corridor-defense figure."""
    apply_academic_style()
    deploy_congo, _ = simulate_congo(topology)
    deploy_himalayas, corridor_mask = simulate_himalayas(topology)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    im0 = axes[0].imshow(deploy_congo, cmap="Reds", origin="lower")
    axes[0].set_title("Congo deployment ($\\mu=8$)")
    axes[0].set_xlabel("Grid X")
    axes[0].set_ylabel("Grid Y")
    fig.colorbar(im0, ax=axes[0], label="Staff/cell")

    im1 = axes[1].imshow(corridor_mask, cmap="viridis", origin="lower")
    axes[1].set_title("Himalaya corridor geometry")
    axes[1].set_xlabel("Grid X")
    axes[1].set_ylabel("Grid Y")
    fig.colorbar(im1, ax=axes[1], label="Corridor indicator")

    im2 = axes[2].imshow(deploy_himalayas, cmap="Reds", origin="lower")
    axes[2].set_title("Himalaya corridor defense")
    axes[2].set_xlabel("Grid X")
    axes[2].set_ylabel("Grid Y")
    fig.colorbar(im2, ax=axes[2], label="Staff/cell")

    save_path = output_dir / "global_adaptation.png"
    save_academic_figure(fig, save_path)

    corridor_concentration = float(np.sum(deploy_himalayas * corridor_mask) / np.sum(deploy_himalayas))
    return {"himalaya_corridor_concentration": corridor_concentration}