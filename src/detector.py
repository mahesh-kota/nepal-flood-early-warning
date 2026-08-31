"""Anomaly detection pipeline for river level telemetry with SHAP explainability."""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

def calculate_dynamic_z_score(rate_of_change: pd.Series, window: int) -> pd.Series:
    """Compute dynamic rolling z-scores for instantaneous level changes."""
    if window < 2:
        raise ValueError("window must be >= 2")
    rolling_mean = rate_of_change.rolling(window=window, min_periods=window).mean().shift(1)
    rolling_std = rate_of_change.rolling(window=window, min_periods=window).std(ddof=0).shift(1)
    centered = rate_of_change - rolling_mean
    safe_std = rolling_std.replace(0, np.nan)
    z_score = centered / safe_std
    z_score = z_score.mask((rolling_std == 0) & (centered > 0), np.inf)
    z_score = z_score.mask((rolling_std == 0) & (centered < 0), -np.inf)
    return z_score.fillna(0.0)

def detect_anomalies(
    telemetry: pd.DataFrame,
    z_score_cutoff: float = 3.0,
    rolling_window: int = 30,
    contamination: float = 0.03,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Run anomaly detection and return results, SHAP values, and feature names."""
    output = telemetry.copy()
    output["velocity"] = output["level"].diff().fillna(0.0)
    output["acceleration"] = output["velocity"].diff().fillna(0.0)
    
    output["z_score"] = calculate_dynamic_z_score(output["velocity"], window=rolling_window)
    output["z_anomaly"] = output["z_score"].abs() >= z_score_cutoff
    
    feature_names = ["level", "velocity", "acceleration"]
    features = output[feature_names]
    
    model = IsolationForest(
        contamination=contamination,
        random_state=random_seed,
        n_estimators=100,
    )
    model.fit(features)
    output["iforest_anomaly"] = model.predict(features) == -1
    output["alert"] = output["z_anomaly"] | output["iforest_anomaly"]
    
    # SHAP TreeExplainer for feature attribution score
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    return output, shap_values, feature_names