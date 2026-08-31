"""Tests for anomaly detection behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import TelemetryConfig, generate_synthetic_telemetry
from src.detector import calculate_dynamic_z_score, detect_anomalies


def test_dynamic_z_score_handles_zero_variance_window() -> None:
    """Z-scores should remain finite when rolling std is zero."""

    rate = pd.Series([0.0] * 20)
    z_score = calculate_dynamic_z_score(rate, window=5)

    assert np.isfinite(z_score).all()
    assert (z_score == 0.0).all()


def test_dynamic_z_score_detects_sharp_spike() -> None:
    """A sudden velocity spike should exceed the anomaly threshold."""

    levels = pd.Series([2.0] * 40 + [4.5] + [4.6] * 10)
    velocity = levels.diff().fillna(0.0)
    z_score = calculate_dynamic_z_score(velocity, window=10)

    assert z_score.iloc[40] > 3.0


def test_combined_detector_flags_surge_region() -> None:
    """Combined detector should flag anomalies once the flood surge starts."""

    telemetry = generate_synthetic_telemetry(
        TelemetryConfig(minutes=240, surge_start_minute=120, surge_duration_minutes=45, random_seed=13)
    )

    results = detect_anomalies(
        telemetry,
        z_score_cutoff=2.5,
        rolling_window=20,
        contamination=0.04,
        random_seed=13,
    )

    flagged = results.loc[results["alert"]]
    assert not flagged.empty
    assert flagged["is_surge"].any()
