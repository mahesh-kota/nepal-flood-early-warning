"""Streamlit app for flood early-warning anomaly visualization."""

from __future__ import annotations

from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_loader import TelemetryConfig, generate_synthetic_telemetry
from src.detector import detect_anomalies


st.set_page_config(page_title="Flood Early Warning", layout="wide")
st.title("Nepal Flood Early-Warning (Synthetic Stream Telemetry)")

with st.sidebar:
    st.header("Detection Settings")
    z_score_cutoff = st.slider("Z-Score cutoff", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
    rolling_window = st.slider("Rolling window size (minutes)", min_value=5, max_value=120, value=30, step=1)
    contamination = st.slider("Isolation Forest contamination", min_value=0.001, max_value=0.2, value=0.03, step=0.001)

config = TelemetryConfig()
telemetry = generate_synthetic_telemetry(config)
results = detect_anomalies(
    telemetry=telemetry,
    z_score_cutoff=z_score_cutoff,
    rolling_window=rolling_window,
    contamination=contamination,
)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(results["timestamp"], results["level"], color="steelblue", linewidth=1.6, label="River level")

alerts = results[results["alert"]]
ax.scatter(
    alerts["timestamp"],
    alerts["level"],
    color="crimson",
    s=36,
    label="Alert flag",
    zorder=3,
)

for ts in alerts["timestamp"].head(10):
    ax.axvline(ts, color="crimson", alpha=0.08, linewidth=1)

ax.set_xlabel("Timestamp")
ax.set_ylabel("Gauge height (m)")
ax.set_title("Simulated river level with anomaly alerts")
ax.legend(loc="upper left")
ax.grid(alpha=0.2)
st.pyplot(fig)

peak_idx = results["level"].idxmax()
peak_level = float(results.loc[peak_idx, "level"])
peak_time = pd.Timestamp(results.loc[peak_idx, "timestamp"])
start_time = pd.Timestamp(results["timestamp"].iloc[0])
time_to_peak = peak_time - start_time

surge_start = pd.Timestamp(results.loc[results["is_surge"], "timestamp"].iloc[0])
first_alert_series = results.loc[results["alert"], "timestamp"]
first_alert = pd.Timestamp(first_alert_series.iloc[0]) if not first_alert_series.empty else None
alert_response = first_alert - surge_start if first_alert is not None else None

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Peak flow height", f"{peak_level:.2f} m")
metric_b.metric("Time-to-peak", str(timedelta(minutes=int(time_to_peak.total_seconds() // 60))))
metric_c.metric(
    "Alert response time",
    str(timedelta(minutes=int(alert_response.total_seconds() // 60))) if alert_response is not None else "No alert",
)

st.subheader("Alert trigger timestamps")
st.dataframe(alerts[["timestamp", "level", "velocity", "z_score", "z_anomaly", "iforest_anomaly"]])
