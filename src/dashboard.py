import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time
import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# CONFIGURATION & SETUP
# --------------------------------------------------------------------------------
st.set_page_config(page_title="MetroPT Live Monitor", layout="wide")
st.title("🚇 MetroPT Predictive Maintenance: Real-Time Inference")

# Constants
DATA_PATH = 'data/engineered_data.parquet'
MODEL_PATH = 'lstm_model.keras'
SEQUENCE_LENGTH = 30  # LSTM window size

# --------------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def get_compressor_state(current_amp):
    """Determines compressor state based on Motor Current (A) rules"""
    # Rules derived from MetroPT-3 Dataset Description
    if current_amp < 1.0:
        return "OFF", "grey"
    elif current_amp < 5.5:
        return "IDLE (Offloaded)", "blue"
    elif current_amp < 8.5:
        return "WORKING (Under Load)", "green"
    else:
        return "STARTING (Surge)", "orange"

@st.cache_resource
def load_model():
    if not tf.io.gfile.exists(MODEL_PATH):
        st.error(f"Model file {MODEL_PATH} not found. Please train the LSTM model first.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_and_prep_data():
    df = pd.read_parquet(DATA_PATH)
    
    # Define features (Same as training)
    target_col = 'RUL'
    # We exclude metadata but keep sensors we need for the dashboard visuals (like Motor_current)
    exclude_cols = [target_col, 'failure', 'timestamp', 'failure_column']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Split 75% (History) / 25% (Live)
    split_point = int(len(df) * 0.75)
    
    # Fit Scaler on History
    history_df = df.iloc[:split_point]
    scaler = StandardScaler()
    scaler.fit(history_df[feature_cols])
    
    # Prepare Live Data (Test Set)
    live_df = df.iloc[split_point - SEQUENCE_LENGTH:].reset_index(drop=True)
    
    return live_df, feature_cols, target_col, scaler

# Load everything
model = load_model()
live_data, feature_cols, target_col, scaler = load_and_prep_data()

# --------------------------------------------------------------------------------
# 2. DASHBOARD LAYOUT
# --------------------------------------------------------------------------------
# Sidebar Controls
st.sidebar.header("⚙️ Simulation Controls")
speed_multiplier = st.sidebar.slider("Simulation Speed", 1, 100, 1, help="1 = Real Time (10s), 100 = Fast Forward")
start_btn = st.sidebar.button("▶️ Start Simulation")

# Top Metrics Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st_timestamp = st.empty() # Timestamp Display
with kpi2:
    st_rul = st.empty()
with kpi3:
    st_status = st.empty()
with kpi4:
    st_progress = st.empty()

# New: Operational State Row
st.markdown("---")
col_state, col_warn1, col_warn2 = st.columns(3)
with col_state:
    st_motor_state = st.empty()
with col_warn1:
    st_warn_pressure = st.empty()
with col_warn2:
    st_metric_amp = st.empty()

# Charts
st.markdown("---")
chart_col, data_col = st.columns([2, 1])
with chart_col:
    chart_placeholder = st.empty()
with data_col:
    st.subheader("📡 Live Sensor Feed")
    table_placeholder = st.empty()

# --------------------------------------------------------------------------------
# 3. SIMULATION LOOP
# --------------------------------------------------------------------------------
if start_btn and model:
    history_actual = []
    history_pred = []
    indices = []
    
    # Initialize Buffer
    input_buffer = live_data[feature_cols].iloc[:SEQUENCE_LENGTH].copy()
    input_buffer_scaled = scaler.transform(input_buffer)
    running_sequence = list(input_buffer_scaled)

    # LOOP
    for i in range(SEQUENCE_LENGTH, len(live_data)):
        
        # --- 1. DATA PREP ---
        row = live_data.iloc[i]  # <--- 'row' is defined here!
        current_time = row['timestamp'] 
        actual_rul = row[target_col]
        
        # Scale Input for Model
        new_features = live_data[feature_cols].iloc[i:i+1]
        new_features_scaled = scaler.transform(new_features)
        
        # Update LSTM Sequence
        running_sequence.pop(0)
        running_sequence.append(new_features_scaled[0])
        
        # --- 2. INFERENCE ---
        model_input = np.array([running_sequence])
        predicted_rul = model.predict(model_input, verbose=0)[0][0]
        
        # --- 3. DIGITAL TWIN LOGIC (New Features) ---
        # Get Motor Current (Ensure column exists, default to 0 if not)
        curr_amp = row.get('Motor_current', 0)
        state_text, state_color = get_compressor_state(curr_amp)
        
        # Check LPS (Low Pressure Switch) Warning
        # LPS: 1 = Low Pressure (<7 bar), 0 = Normal
        is_low_pressure = row.get('LPS', 0) > 0.5 
        
        # --- 4. VISUALIZATION ---
        
        # A. Update Top KPIs
        st_timestamp.metric(label="Current Time", value=str(current_time)[0:19])
        st_rul.metric(label="Predicted RUL", value=f"{predicted_rul:.1f} h", delta=f"{actual_rul - predicted_rul:.1f}")
        
        # RUL Status Color
        if predicted_rul < 24:
            st_status.error("🚨 CRITICAL FAILURE IMMINENT")
        elif predicted_rul < 72:
            st_status.warning("⚠️ DEGRADING")
        else:
            st_status.success("✅ SYSTEM HEALTHY")

        # B. Update Operational State (The New Stuff)
        st_motor_state.markdown(f"### Motor State: :{state_color}[{state_text}]")
        st_metric_amp.metric("Motor Current", f"{curr_amp:.2f} A")
        
        if is_low_pressure:
            st_warn_pressure.error("🔻 PRESSURE DROP (<7 bar)")
        else:
            st_warn_pressure.success("✅ Pressure Normal")

        # C. Update Chart
        history_actual.append(actual_rul)
        history_pred.append(predicted_rul)
        indices.append(i)
        
        # Keep chart window clean (last 200 points)
        if len(indices) > 200: 
            history_actual.pop(0)
            history_pred.pop(0)
            indices.pop(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history_actual, mode='lines', name='Actual RUL', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(y=history_pred, mode='lines', name='Predicted RUL', line=dict(color='red', width=3)))
        fig.update_layout(
            title="Remaining Useful Life (RUL) Prediction",
            xaxis_title="Time Steps",
            yaxis_title="Hours",
            height=350, 
            margin=dict(l=0, r=0, t=30, b=0), 
            yaxis_range=[0, 150]
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # D. Update Table
        table_placeholder.dataframe(new_features.T, use_container_width=True)
        
        # --- 5. SPEED CONTROL ---
        wait_time = 10.0 / speed_multiplier
        if speed_multiplier == 1:
            for s in range(10, 0, -1):
                st_progress.text(f"⏳ Next reading in {s}s...")
                time.sleep(1)
        else:
            time.sleep(wait_time)