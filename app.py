"""Production-grade Early-Warning Flood Monitoring Dashboard."""
from __future__ import annotations
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import TelemetryConfig, generate_synthetic_telemetry
from src.detector import detect_anomalies

st.set_page_config(page_title="Trishuli Basin Flood Early Warning", layout="wide")

# Custom CSS styling for professional enterprise look
# Custom CSS styling for professional enterprise look
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #00d4b1;
    }
    .status-alert {
        background-color: #3d0c11;
        color: #ff4b4b;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-ok {
        background-color: #0c3d20;
        color: #00d4b1;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Trishuli River Basin | Hydrological Early-Warning System")
st.caption("Real-Time Telemetry Processing & Anomaly Analytics Engine")

# Sidebar - Modular Engineering Controls
with st.sidebar:
    st.header("⚙️ Sensor & Model Parameters")
    z_score_cutoff = st.slider("Z-Score Sensitivity Threshold", 1.5, 6.0, 3.0, 0.1)
    rolling_window = st.slider("Rolling Velocity Window (min)", 5, 60, 20, 1)
    contamination = st.slider("Isolation Forest Contamination", 0.005, 0.08, 0.02, 0.005)
    
    st.markdown("---")
    st.subheader("Operational Thresholds")
    minor_flood_stage = st.number_input("Minor Flood Stage (m)", value=3.5)
    major_flood_stage = st.number_input("Major Flood Stage (m)", value=4.2)

# Run Data & Inference Pipeline
config = TelemetryConfig()
telemetry = generate_synthetic_telemetry(config)
results = detect_anomalies(
    telemetry=telemetry,
    z_score_cutoff=z_score_cutoff,
    rolling_window=rolling_window,
    contamination=contamination,
)

# Operational Metrics & Warning Logic
alerts = results[results["alert"]]
surge_start = results.loc[results["is_surge"], "timestamp"].iloc[0]
first_alert = alerts["timestamp"].iloc[0] if not alerts.empty else None

lead_time_minutes = int((surge_start - first_alert).total_seconds() / 60) if (first_alert and first_alert < surge_start) else 0
peak_level = results["level"].max()
peak_time = results.loc[results["level"].idxmax(), "timestamp"]

# System Status Header
col_status, col_lead, col_peak, col_velocity = st.columns(4)

with col_status:
    if not alerts.empty:
        st.markdown('<div class="status-alert">🚨 CRITICAL SURGE ALERT ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-ok">NORMAL FLOW CONDITIONS</div>', unsafe_allow_html=True)

with col_lead:
    st.metric("Early Warning Lead Time", f"{lead_time_minutes} min", delta="Pre-Surge Alert" if lead_time_minutes > 0 else "Instant")

with col_peak:
    st.metric("Peak Gauge Height", f"{peak_level:.2f} m", delta=f"{peak_level - minor_flood_stage:.2f} m above stage")

with col_velocity:
    max_vel = results["velocity"].max()
    st.metric("Max Flow Velocity (dy/dt)", f"{max_vel:.3f} m/min")

st.markdown("---")

# Subplot: Dual-Axis River Level + Rate of Change (dy/dt)
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08, 
    subplot_titles=("River Gauge Level & Flood Warning Stages", "Instantaneous Rate of Change Velocity (dy/dt)")
)

# 1. Main Hydrograph
fig.add_trace(
    go.Scatter(x=results["timestamp"], y=results["level"], name="Gauge Height (m)", line=dict(color="#00d4b1", width=2)),
    row=1, col=1
)

# Threshold Stage Lines
fig.add_hline(y=minor_flood_stage, line_dash="dash", line_color="orange", annotation_text="Minor Flood Stage", row=1, col=1)
fig.add_hline(y=major_flood_stage, line_dash="dash", line_color="red", annotation_text="Major Flood Stage", row=1, col=1)

# Anomaly Markers
if not alerts.empty:
    fig.add_trace(
        go.Scatter(
            x=alerts["timestamp"], y=alerts["level"], 
            mode="markers", name="Anomaly Flagged", 
            marker=dict(color="red", size=7, symbol="triangle-up")
        ),
        row=1, col=1
    )

# 2. Velocity Subplot (Rate of Change)
fig.add_trace(
    go.Scatter(x=results["timestamp"], y=results["velocity"], name="Velocity (m/min)", line=dict(color="#a45ee5", width=1.5)),
    row=2, col=1
)

fig.update_layout(
    height=600,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Incident Event Log (Aggregated grouped alerts instead of single-minute spam)
st.subheader("📋 Operational Incident Event Log")
if not alerts.empty:
    # Group contiguous alert minutes into discrete Incident Events
    alerts["event_id"] = (alerts.index != alerts.index.to_series().shift() + 1).cumsum()
    event_log = alerts.groupby("event_id").agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        max_height=("level", "max"),
        max_velocity=("velocity", "max"),
        total_flags=("alert", "count")
    ).reset_index(drop=True)
    
    st.dataframe(event_log, use_container_width=True)
else:
    st.info("No operational alert events recorded in current window.")