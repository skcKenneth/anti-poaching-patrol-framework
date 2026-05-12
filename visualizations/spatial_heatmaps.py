"""Spatial heatmaps for WPP and deployment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from visualizations.style_guide import apply_academic_style, save_academic_figure


def render_protection_strategy(
    wpp_field: NDArray[np.float64],
    deployment_map: NDArray[np.float64],
    waterholes: List[Tuple[int, int]],
) -> Path:
    """Renders side-by-side WPP and deployment maps.

    Args:
        wpp_field: Wildlife protection potential field.
        deployment_map: Optimized ranger deployment map.
        waterholes: Waterhole coordinates `(y, x)`.

    Returns:
        Path to saved figure.
    """
    apply_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    im1 = axes[0].imshow(wpp_field, cmap="viridis", origin="lower")
    axes[0].set_title("Wildlife Protection Potential (WPP)")
    axes[0].set_xlabel("Grid X")
    axes[0].set_ylabel("Grid Y")
    fig.colorbar(im1, ax=axes[0], label="WPP intensity (a.u.)")

    im2 = axes[1].imshow(deployment_map, cmap="Reds", origin="lower")
    axes[1].set_title("Optimized Personnel Deployment")
    axes[1].set_xlabel("Grid X")
    axes[1].set_ylabel("Grid Y")
    fig.colorbar(im2, ax=axes[1], label="Ranger allocation (staff/cell)")

    if waterholes:
        coord = np.asarray(waterholes, dtype=float)
        axes[0].scatter(coord[:, 1], coord[:, 0], c="white", s=12, marker="x", alpha=0.85, label="Waterholes")
        axes[1].scatter(coord[:, 1], coord[:, 0], c="black", s=12, marker="x", alpha=0.8, label="Waterholes")
        axes[0].legend(loc="upper right")
        axes[1].legend(loc="upper right")

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "protection_strategy.png"
    save_academic_figure(fig, save_path)
    return save_path