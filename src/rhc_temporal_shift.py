"""Receding horizon control helper for temporal deployment shifts."""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray


class TemporalRHC:
    """Applies a white-box rolling-horizon weighting policy.

    At each step, the solver sees a horizon of predicted multipliers
    `{m_t, ..., m_{t+H-1}}` and executes only the first-step deployment.
    """

    def __init__(self, horizon_steps: int = 6, discount: float = 0.95) -> None:
        self.horizon_steps = horizon_steps
        self.discount = discount

    def aggregate_forecast(self, current_wpp: NDArray[np.float64], forecast_multipliers: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Builds horizon-weighted WPP forecast.

        Args:
            current_wpp: Current WPP field.
            forecast_multipliers: List of multiplicative maps over horizon.
        """
        horizon = min(self.horizon_steps, len(forecast_multipliers))
        aggregate = np.zeros_like(current_wpp, dtype=float)
        for idx in range(horizon):
            weight = self.discount**idx
            aggregate += weight * current_wpp * forecast_multipliers[idx]
        return np.maximum(aggregate, 0.0)