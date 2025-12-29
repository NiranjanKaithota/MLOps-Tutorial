from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import uvicorn

# --------------------------------------------------------------------------------
# EVIDENTLY COMPATIBILITY BLOCK (Handles v0.2 vs v0.4+ APIs)
# --------------------------------------------------------------------------------
try:
    # Try new API (Evidently 0.7+)
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    try:
        # Try old API (Evidently < 0.7)
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError as e:
        print(f"❌ Could not import Evidently modules: {e}")
        # We don't exit here to allow the API to run even if drift fails
        Report = None
        DataDriftPreset = None

# --------------------------------------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------------------------------------
MODELS = {}
SCALER = None
FULL_DF = None      # Full dataset (for reference)
LIVE_DF = None      # Simulation dataset
BUFFER = []         # For Model Inference (Scaled)
HISTORY_BUFFER = [] # For Drift Detection (Raw Data)
SIMULATION_IDX = 0
SEQUENCE_LENGTH = 180
DRIFT_INTERVAL = 10 # Run drift check every 10 samples

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if we are running in Docker (files mapped to /app)
if os.path.exists('/app/gru_model.keras'):
    PROJECT_ROOT = '/app'
else:
    # Local Development fallback
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

print(f"📂 Project Root resolved to: {PROJECT_ROOT}") # Debug print

DATA_PATH = os.path.join(PROJECT_ROOT, 'data/engineered_data.parquet') 
SCALER_PATH = os.path.join(PROJECT_ROOT, 'scaler.pkl')
REPORT_DIR = os.path.join(BASE_DIR, 'static')

# Ensure static folder exists for reports
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# HELPER: DRIFT DETECTION TASK
# --------------------------------------------------------------------------------
def run_drift_report(current_data_window):
    """
    Generates an Evidently Data Drift Report.
    Reference: First 500 rows of training data (Stable).
    Current: The last 180 rows from the live stream.
    """
    if Report is None:
        print("⚠️ Evidently library not loaded correctly. Skipping report generation.")
        return

    print(f"🔎 [DRIFT] Analyzing last {len(current_data_window)} samples...")
    
    try:
        # 1. Define Datasets
        # Reference: We take a slice of the original dataset that we know is "healthy"
        reference_data = FULL_DF.iloc[:500] 
        
        # Current: The raw data from our live history buffer
        current_data = pd.DataFrame(current_data_window, columns=FULL_DF.columns)
        
        # 2. Build Report
        drift_report = Report(metrics=[DataDriftPreset()])
        
        # 3. Run Report
        # Note: In newer Evidently versions, run() returns the object itself or a result
        results = drift_report.run(reference_data=reference_data, current_data=current_data)
        
        # 4. Save Report
        report_path = os.path.join(REPORT_DIR, 'data_drift_report.html')
        
        # Compatibility check: if run() returned something with save_html, use it.
        # Otherwise use the report object directly.
        if results is not None and hasattr(results, 'save_html'):
            results.save_html(report_path)
        else:
            drift_report.save_html(report_path)
            
        print("✅ [DRIFT] Report updated: static/data_drift_report.html")
        
    except Exception as e:
        print(f"❌ [DRIFT] Failed to generate report: {e}")

# --------------------------------------------------------------------------------
# LIFESPAN MANAGER
# --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS, SCALER, FULL_DF, LIVE_DF, BUFFER, SIMULATION_IDX, HISTORY_BUFFER
    
    print("⚡ [STARTUP] Loading Resources...")
    
    # Load Models & Scaler
    try:
        # Load all models
        model_files = {
            'GRU': 'gru_model.keras',
            'CNN': 'cnn_model.keras',
            'LSTM': 'lstm_model.keras'
        }
        
        for name, filename in model_files.items():
            path = os.path.join(PROJECT_ROOT, filename)
            if os.path.exists(path):
                MODELS[name] = tf.keras.models.load_model(path)
                print(f"   ✅ {name} Model Loaded")
            else:
                print(f"   ⚠️ {name} Model not found at {path}")

        with open(SCALER_PATH, 'rb') as f:
            SCALER = pickle.load(f)
        
        # Load Data
        FULL_DF = pd.read_parquet(DATA_PATH)
        
        # Split: Simulation gets last 25%
        sim_cutoff = int(len(FULL_DF) * 0.75)
        LIVE_DF = FULL_DF.iloc[sim_cutoff:].reset_index(drop=True)
        
        # Init Buffer (Scaled for Inference)
        init_data = LIVE_DF.iloc[:SEQUENCE_LENGTH]
        exclude_cols = ['timestamp', 'failure', 'RUL']
        feature_cols = [c for c in LIVE_DF.columns if c not in exclude_cols]
        
        BUFFER = SCALER.transform(init_data[feature_cols]).tolist()
        
        # Init History Buffer (Raw for Drift)
        HISTORY_BUFFER = init_data.to_dict('records')
        
        SIMULATION_IDX = SEQUENCE_LENGTH
        print(f"   ✅ Data Loaded. Simulation rows: {len(LIVE_DF)}")
        
        # Generate initial report so the file exists immediately
        run_drift_report(init_data.to_dict('records'))
        
    except Exception as e:
        print(f"   ❌ Error loading resources: {e}")
        
    yield
    MODELS.clear()

# --------------------------------------------------------------------------------
# APP SETUP
# --------------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# Mount 'static' folder to serve the HTML report
app.mount("/static", StaticFiles(directory=REPORT_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------------------
@app.get("/api/next")
def get_next_step(background_tasks: BackgroundTasks):
    global SIMULATION_IDX, BUFFER, HISTORY_BUFFER
    
    if LIVE_DF is None or SIMULATION_IDX >= len(LIVE_DF):
        return {"status": "finished"}

    # 1. Get current raw row
    row = LIVE_DF.iloc[SIMULATION_IDX]
    
    # 2. Add to Drift History (Keep last 180 raw rows)
    HISTORY_BUFFER.append(row.to_dict())
    if len(HISTORY_BUFFER) > SEQUENCE_LENGTH:
        HISTORY_BUFFER.pop(0)
    
    # 3. Trigger Drift Check every 10 samples
    # We use background_tasks so we don't slow down the live UI
    if SIMULATION_IDX % DRIFT_INTERVAL == 0:
        current_window = list(HISTORY_BUFFER) # Copy data to avoid mutation issues
        background_tasks.add_task(run_drift_report, current_window)

    # 4. Prepare for Inference (Scaling)
    exclude_cols = ['timestamp', 'failure', 'RUL']
    feature_cols = [c for c in LIVE_DF.columns if c not in exclude_cols]
    
    current_features = row[feature_cols].values.reshape(1, -1)
    scaled_features = SCALER.transform(current_features)
    
    # 5. Update Inference Buffer
    BUFFER.pop(0)
    BUFFER.append(scaled_features[0])
    
    # 6. Inference (All Models)
    model_input = np.array([BUFFER])
    predictions = {}
    
    for name, model in MODELS.items():
        try:
            pred = model.predict(model_input, verbose=0)[0][0]
            predictions[name] = float(max(0, pred))
        except Exception as e:
            # print(f"Inference error {name}: {e}") # Optional logging
            predictions[name] = 0.0

    # 7. Status Logic (Based on GRU Champion)
    rul = predictions.get('GRU', 0)
    status = "NORMAL"
    if rul < 12.0: status = "CRITICAL"
    elif rul < 24.0: status = "WARNING"

    # 8. Response
    response = {
        "timestamp": str(row['timestamp']),
        "index": int(SIMULATION_IDX),
        "actual_rul": float(row['RUL']),
        "predictions": predictions,
        "status": status,
        "telemetry": {
            "motor_current": float(row.get('Motor_current', 0)),
            "oil_temp": float(row.get('Oil_temperature', 0)),
            "pressure": float(row.get('TP2', 0)),
            "vibration": float(row.get('Vibration', 0))
        }
    }
    
    SIMULATION_IDX += 1
    return response

@app.post("/api/reset")
def reset_simulation():
    global SIMULATION_IDX
    SIMULATION_IDX = SEQUENCE_LENGTH
    return {"message": "Simulation reset"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)