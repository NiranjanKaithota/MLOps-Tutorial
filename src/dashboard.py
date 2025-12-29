import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
import plotly.graph_objects as go
import os

# --------------------------------------------------------------------------------
# CONFIGURATION & SETUP
# --------------------------------------------------------------------------------
st.set_page_config(page_title="MetroPT Multi-Model Monitor", layout="wide")
st.title("🚇 MetroPT Predictive Maintenance: Model Benchmark")

# Constants
DATA_PATH = 'data/engineered_data.parquet'
SEQUENCE_LENGTH = 180  # V2 Context Window
SCALER_PATH = 'scaler.pkl'

# Model Paths
MODELS = {
    'GRU': 'gru_model.keras',   # The Champion
    'CNN': 'cnn_model.keras',   # The Runner-up
    'LSTM': 'lstm_model.keras'  # The Baseline
}

# --------------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def get_compressor_state(current_amp):
    """Determines compressor state based on Motor Current (A) rules"""
    if current_amp < 1.0: return "OFF", "grey"
    elif current_amp < 5.5: return "IDLE", "blue"
    elif current_amp < 8.5: return "WORKING", "green"
    else: return "STARTING", "orange"

@st.cache_resource
def load_all_models():
    loaded_models = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            loaded_models[name] = tf.keras.models.load_model(path)
        else:
            st.warning(f"⚠️ Model {name} not found at {path}. Skipping.")
    return loaded_models

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error("Scaler not found. Please run training first.")
        return None
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_and_prep_data():
    df = pd.read_parquet(DATA_PATH)
    
    # Define features
    target_col = 'RUL'
    exclude_cols = [target_col, 'failure', 'timestamp']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # ISOLATE SIMULATION DATA (The "Hidden" 25%)
    # This matches our strict V2.2 split logic
    sim_cutoff = int(len(df) * 0.75)
    live_df = df.iloc[sim_cutoff:].reset_index(drop=True)
    
    return live_df, feature_cols, target_col

# Load Resources
models = load_all_models()
scaler = load_scaler()
live_data, feature_cols, target_col = load_and_prep_data()

# --------------------------------------------------------------------------------
# 2. DASHBOARD LAYOUT
# --------------------------------------------------------------------------------
# Sidebar Controls
st.sidebar.header("⚙️ Simulation Controls")
speed_multiplier = st.sidebar.slider("Simulation Speed", 1, 100, 10)
start_btn = st.sidebar.button("▶️ Start Simulation")

# Top Metrics Row (Global State)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1: st_timestamp = st.empty()
with kpi2: st_motor_state = st.empty()
with kpi3: st_warn_pressure = st.empty()
with kpi4: st_metric_amp = st.empty()

st.markdown("---")

# MAIN LAYOUT: 3 Columns for Main Graph vs 1 Column for Side Graphs
col_main, col_side = st.columns([3, 1])

# Placeholders for Graphs
with col_main:
    st.subheader("🏆 Champion Model (GRU)")
    gru_chart_placeholder = st.empty()
    st_gru_metric = st.empty()

with col_side:
    st.subheader("Benchmarks")
    st.markdown("**CNN Model**")
    cnn_chart_placeholder = st.empty()
    st_cnn_metric = st.empty()
    
    st.divider()
    
    st.markdown("**LSTM Model**")
    lstm_chart_placeholder = st.empty()
    st_lstm_metric = st.empty()

# --------------------------------------------------------------------------------
# 3. SIMULATION LOOP
# --------------------------------------------------------------------------------
if start_btn and models and scaler:
    # History buffers for plotting
    history_actual = []
    history_preds = {'GRU': [], 'CNN': [], 'LSTM': []}
    indices = []
    
    # Initialize Input Buffer (First 180 steps)
    input_buffer = live_data[feature_cols].iloc[:SEQUENCE_LENGTH].copy()
    input_buffer_scaled = scaler.transform(input_buffer)
    running_sequence = list(input_buffer_scaled)

    # Loop through Simulation Data
    # Start loop AFTER the buffer
    for i in range(SEQUENCE_LENGTH, len(live_data)):
        
        # --- 1. DATA PREP ---
        row = live_data.iloc[i]
        actual_rul = row[target_col]
        
        # Update Sequence
        new_features = live_data[feature_cols].iloc[i:i+1]
        new_features_scaled = scaler.transform(new_features)
        
        running_sequence.pop(0)
        running_sequence.append(new_features_scaled[0])
        
        # Prepare 3D Input: (1, 180, 14)
        model_input = np.array([running_sequence])
        
        # --- 2. INFERENCE (ALL MODELS) ---
        preds = {}
        for name, model in models.items():
            # Predict and clip negative values
            p = model.predict(model_input, verbose=0)[0][0]
            preds[name] = max(0, p) 

        # --- 3. DIGITAL TWIN STATS ---
        curr_amp = row.get('Motor_current', 0)
        state_text, state_color = get_compressor_state(curr_amp)
        is_low_pressure = row.get('LPS', 0) > 0.5 
        
        # --- 4. VISUALIZATION UPDATES ---
        
        # A. Top KPIs
        st_timestamp.metric(label="Simulated Time", value=str(row['timestamp'])[5:19])
        st_motor_state.markdown(f"**State:** :{state_color}[{state_text}]")
        if is_low_pressure:
            st_warn_pressure.error("🔻 PRESSURE DROP")
        else:
            st_warn_pressure.success("✅ Pressure OK")
        st_metric_amp.metric("Current", f"{curr_amp:.2f} A")

        # B. Update Histories
        history_actual.append(actual_rul)
        indices.append(i)
        for name in preds:
            history_preds[name].append(preds[name])
        
        # Keep window size manageable (Last 100 points)
        if len(indices) > 100:
            history_actual.pop(0)
            indices.pop(0)
            for name in preds:
                history_preds[name].pop(0)

        # C. PLOT 1: GRU (Big Main Chart)
        fig_gru = go.Figure()
        fig_gru.add_trace(go.Scatter(y=history_actual, mode='lines', name='Actual RUL', line=dict(color='gray', dash='dash')))
        fig_gru.add_trace(go.Scatter(y=history_preds['GRU'], mode='lines', name='GRU Prediction', line=dict(color='green', width=3)))
        fig_gru.update_layout(
            height=400, 
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_range=[0, 130],
            xaxis_title="Time Steps", yaxis_title="RUL (Hours)"
        )
        gru_chart_placeholder.plotly_chart(fig_gru, use_container_width=True)
        
        # GRU Metric with Delta
        gru_err = abs(actual_rul - preds['GRU'])
        st_gru_metric.metric("GRU Prediction", f"{preds['GRU']:.1f} h", delta=f"Err: {gru_err:.1f} h", delta_color="inverse")

        # D. PLOT 2: CNN (Side Chart)
        if 'CNN' in models:
            fig_cnn = go.Figure()
            fig_cnn.add_trace(go.Scatter(y=history_actual, mode='lines', showlegend=False, line=dict(color='gray', dash='dash')))
            fig_cnn.add_trace(go.Scatter(y=history_preds['CNN'], mode='lines', showlegend=False, line=dict(color='red', width=2)))
            fig_cnn.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), yaxis_range=[0, 130])
            cnn_chart_placeholder.plotly_chart(fig_cnn, use_container_width=True)
            st_cnn_metric.write(f"**Pred:** {preds['CNN']:.1f} h (Err: {abs(actual_rul - preds['CNN']):.1f})")

        # E. PLOT 3: LSTM (Side Chart)
        if 'LSTM' in models:
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(y=history_actual, mode='lines', showlegend=False, line=dict(color='gray', dash='dash')))
            fig_lstm.add_trace(go.Scatter(y=history_preds['LSTM'], mode='lines', showlegend=False, line=dict(color='orange', width=2)))
            fig_lstm.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), yaxis_range=[0, 130])
            lstm_chart_placeholder.plotly_chart(fig_lstm, use_container_width=True)
            st_lstm_metric.write(f"**Pred:** {preds['LSTM']:.1f} h (Err: {abs(actual_rul - preds['LSTM']):.1f})")

        # --- 5. SPEED CONTROL ---
        time.sleep(1.0 / speed_multiplier)