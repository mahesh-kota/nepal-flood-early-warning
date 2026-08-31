"""Synthetic telemetry generation for flood early-warning experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for synthetic telemetry generation."""

    minutes: int = 720
    surge_start_minute: int = 480
    surge_duration_minutes: int = 90
    random_seed: int = 7


def generate_synthetic_telemetry(config: TelemetryConfig | None = None) -> pd.DataFrame:
    """Generate minute-level river gauge telemetry with a flood surge anomaly.

    Returns:
        A dataframe with timestamp, level, velocity, and surge labels.
    """

    cfg = config or TelemetryConfig()
    rng = np.random.default_rng(cfg.random_seed)

    timeline = pd.date_range("2026-01-01", periods=cfg.minutes, freq="min")
    baseline = 2.2 + 0.05 * np.sin(np.linspace(0, 10 * np.pi, cfg.minutes))
    noise = rng.normal(0, 0.015, cfg.minutes)
    level = baseline + noise

    surge_end = min(cfg.surge_start_minute + cfg.surge_duration_minutes, cfg.minutes)
    surge_idx = np.arange(cfg.surge_start_minute, surge_end)

    if len(surge_idx) > 0:
        progress = np.linspace(0, 1, len(surge_idx))
        surge_ramp = 0.3 + 2.1 * (progress**2)
        level[surge_idx] += surge_ramp

    telemetry = pd.DataFrame({"timestamp": timeline, "level": level})
    telemetry["velocity"] = telemetry["level"].diff().fillna(0.0)
    telemetry["is_surge"] = False
    telemetry.loc[surge_idx, "is_surge"] = True

    return telemetry
