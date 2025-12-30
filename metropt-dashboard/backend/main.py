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
SCALER_X = None     # Scaler for Input Features
SCALER_Y = None     # Scaler for Target (RUL)
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
    # Local Development: go up two levels from backend/ to project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

print(f"📂 Project Root resolved to: {PROJECT_ROOT}") # Debug print

DATA_PATH = os.path.join(PROJECT_ROOT, 'data/engineered_data.parquet') 

# UPDATED PATHS: Look for X and Y scalers
SCALER_X_PATH = os.path.join(PROJECT_ROOT, 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join(PROJECT_ROOT, 'scaler_y.pkl')
REPORT_DIR = os.path.join(BASE_DIR, 'static')

# Ensure static folder exists for reports
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# HELPER: DRIFT DETECTION TASK
# --------------------------------------------------------------------------------
def run_drift_report(current_data_window):
    """
    Generates an Evidently Data Drift Report.
    """
    if Report is None:
        print("⚠️ Evidently library not loaded correctly. Skipping report generation.")
        return

    print(f"🔎 [DRIFT] Analyzing last {len(current_data_window)} samples...")
    
    try:
        # 1. Define Datasets
        reference_data = FULL_DF.iloc[:500] 
        current_data = pd.DataFrame(current_data_window, columns=FULL_DF.columns)
        
        # 2. Build Report
        drift_report = Report(metrics=[DataDriftPreset()])
        
        # 3. Run Report
        results = drift_report.run(reference_data=reference_data, current_data=current_data)
        
        # 4. Save Report
        report_path = os.path.join(REPORT_DIR, 'data_drift_report.html')
        
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
    global MODELS, SCALER_X, SCALER_Y, FULL_DF, LIVE_DF, BUFFER, SIMULATION_IDX, HISTORY_BUFFER
    
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

        # LOAD SCALERS (CRITICAL UPDATE)
        if os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
            with open(SCALER_X_PATH, 'rb') as f: SCALER_X = pickle.load(f)
            with open(SCALER_Y_PATH, 'rb') as f: SCALER_Y = pickle.load(f)
            print("   ✅ Dual Scalers Loaded (X and Y)")
        else:
            # Fallback for old single scaler setup
            old_scaler_path = os.path.join(PROJECT_ROOT, 'scaler.pkl')
            if os.path.exists(old_scaler_path):
                with open(old_scaler_path, 'rb') as f: SCALER_X = pickle.load(f)
                SCALER_Y = None
                print("   ⚠️ Loaded Legacy Scaler (Prediction output might be scaled incorrectly)")
            else:
                raise FileNotFoundError("No scalers found! Run train.py first.")

        # Load Data
        FULL_DF = pd.read_parquet(DATA_PATH)
        
        # Split: Simulation gets last 25%
        sim_cutoff = int(len(FULL_DF) * 0.75)
        LIVE_DF = FULL_DF.iloc[sim_cutoff:].reset_index(drop=True)
        
        # Init Buffer (Scaled for Inference)
        init_data = LIVE_DF.iloc[:SEQUENCE_LENGTH]
        exclude_cols = ['timestamp', 'failure', 'RUL']
        feature_cols = [c for c in LIVE_DF.columns if c not in exclude_cols]
        
        # Transform initial buffer using SCALER_X
        BUFFER = SCALER_X.transform(init_data[feature_cols]).tolist()
        
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
    if SIMULATION_IDX % DRIFT_INTERVAL == 0:
        current_window = list(HISTORY_BUFFER) 
        background_tasks.add_task(run_drift_report, current_window)

    # 4. Prepare for Inference (USE SCALER_X)
    exclude_cols = ['timestamp', 'failure', 'RUL']
    feature_cols = [c for c in LIVE_DF.columns if c not in exclude_cols]
    
    current_features = row[feature_cols].values.reshape(1, -1)
    scaled_features = SCALER_X.transform(current_features)
    
    # 5. Update Inference Buffer
    BUFFER.pop(0)
    BUFFER.append(scaled_features[0])
    
    # 6. Inference (All Models)
    model_input = np.array([BUFFER])
    predictions = {}
    
    for name, model in MODELS.items():
        try:
            # Prediction is Scaled (e.g., 0.5) - handle variable output shapes robustly
            pred_scaled = model.predict(model_input, verbose=0)
            pred_arr = np.asarray(pred_scaled).ravel()
            pred_val = float(pred_arr[0]) if pred_arr.size > 0 else 0.0

            # INVERSE TRANSFORM (Convert scaled -> hours) if scaler available
            if SCALER_Y is not None:
                try:
                    pred_hours = float(SCALER_Y.inverse_transform(np.array([[pred_val]]))[0][0])
                except Exception:
                    # If inverse_transform fails for any reason, fall back to raw value
                    pred_hours = pred_val
            else:
                # Legacy scaler absent: assume model outputs are scaled in [0,1]
                # Multiply by observed RUL max (fallback 120h)
                try:
                    y_max = float(FULL_DF['RUL'].max()) if FULL_DF is not None else 120.0
                    if not np.isfinite(y_max) or y_max <= 0:
                        y_max = 120.0
                except Exception:
                    y_max = 120.0
                pred_hours = pred_val * y_max

            predictions[name] = float(max(0.0, pred_hours))
        except Exception as e:
            # print(f"Inference error {name}: {e}") # Optional logging
            predictions[name] = 0.0

    # 7. Status Logic (Prioritize LSTM if available, else GRU)
    champion_rul = predictions.get('LSTM', predictions.get('GRU', 0))
    
    status = "NORMAL"
    if champion_rul < 12.0: status = "CRITICAL"
    elif champion_rul < 24.0: status = "WARNING"

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

@app.get("/api/rul-projection")
def get_rul_projection():
    """
    Returns RUL projection data from test data for the Overview chart.
    """
    global LIVE_DF, MODELS, SCALER_X, SCALER_Y, SEQUENCE_LENGTH
    
    if LIVE_DF is None or not MODELS or SCALER_X is None:
        return {"error": "Data or models not loaded"}, 500
    
    try:
        STRIDE = 100
        exclude_cols = ['timestamp', 'failure', 'RUL']
        feature_cols = [c for c in LIVE_DF.columns if c not in exclude_cols]
        
        chart_data = []
        
        # Use LSTM as champion if available
        model_name = 'LSTM' if 'LSTM' in MODELS else 'GRU'
        model = MODELS.get(model_name)
        
        if not model:
             return {"data": []}

        for i in range(0, len(LIVE_DF) - SEQUENCE_LENGTH, STRIDE):
            seq_data = LIVE_DF.iloc[i:i + SEQUENCE_LENGTH]
            
            # Scale features
            seq_features = SCALER_X.transform(seq_data[feature_cols])
            model_input = np.array([seq_features])
            
            actual_rul = float(LIVE_DF.iloc[i + SEQUENCE_LENGTH]['RUL'])
            
            # Predict & robustly extract value
            pred_scaled = model.predict(model_input, verbose=0)
            pred_arr = np.asarray(pred_scaled).ravel()
            pred_val = float(pred_arr[0]) if pred_arr.size > 0 else 0.0
            if SCALER_Y is not None:
                try:
                    predicted_rul = float(max(0.0, SCALER_Y.inverse_transform(np.array([[pred_val]]))[0][0]))
                except Exception:
                    predicted_rul = float(max(0.0, pred_val))
            else:
                try:
                    y_max = float(FULL_DF['RUL'].max()) if FULL_DF is not None else 120.0
                    if not np.isfinite(y_max) or y_max <= 0:
                        y_max = 120.0
                except Exception:
                    y_max = 120.0
                predicted_rul = float(max(0.0, pred_val * y_max))
            
            time_label = f"T{len(chart_data)}"
            
            chart_data.append({
                "time": time_label,
                "value": round(predicted_rul, 1),
                "baseline": round(actual_rul, 1)
            })
        
        if len(chart_data) > 25:
            chart_data = chart_data[-25:]
        
        return {"data": chart_data}
        
    except Exception as e:
        print(f"Error generating RUL projection: {e}")
        return {"error": str(e)}, 500


@app.get("/api/rul-projection-trained")
def get_rul_projection_trained():
    """
    Returns RUL projection data based on the trained portion of the dataset.
    """
    global FULL_DF, MODELS, SCALER_X, SCALER_Y, SEQUENCE_LENGTH

    if FULL_DF is None or SCALER_X is None:
        return {"error": "Data or scaler not loaded"}, 500

    try:
        cutoff = int(len(FULL_DF) * 0.75)
        trained_df = FULL_DF.iloc[:cutoff].reset_index(drop=True)

        STRIDE = max(1, int(len(trained_df) / 100))
        exclude_cols = ['timestamp', 'failure', 'RUL']
        feature_cols = [c for c in trained_df.columns if c not in exclude_cols]

        chart_data = []
        model_name = 'LSTM' if 'LSTM' in MODELS else 'GRU'
        model = MODELS.get(model_name)

        if not model: return {"data": []}

        for i in range(0, max(1, len(trained_df) - SEQUENCE_LENGTH), STRIDE):
            seq_data = trained_df.iloc[i:i + SEQUENCE_LENGTH]
            if len(seq_data) < SEQUENCE_LENGTH: break

            seq_features = SCALER_X.transform(seq_data[feature_cols])
            model_input = np.array([seq_features])

            actual_rul = float(trained_df.iloc[i + SEQUENCE_LENGTH]['RUL'])

            pred_scaled = model.predict(model_input, verbose=0)
            pred_arr = np.asarray(pred_scaled).ravel()
            pred_val = float(pred_arr[0]) if pred_arr.size > 0 else 0.0
            if SCALER_Y is not None:
                try:
                    predicted_rul = float(max(0.0, SCALER_Y.inverse_transform(np.array([[pred_val]]))[0][0]))
                except Exception:
                    predicted_rul = float(max(0.0, pred_val))
            else:
                try:
                    y_max = float(FULL_DF['RUL'].max()) if FULL_DF is not None else 120.0
                    if not np.isfinite(y_max) or y_max <= 0:
                        y_max = 120.0
                except Exception:
                    y_max = 120.0
                predicted_rul = float(max(0.0, pred_val * y_max))

            time_label = f"T{len(chart_data)}"

            chart_data.append({
                "time": time_label,
                "value": round(predicted_rul, 1),
                "baseline": round(actual_rul, 1)
            })

        if len(chart_data) > 25:
            chart_data = chart_data[-25:]

        return {"data": chart_data}

    except Exception as e:
        print(f"Error generating trained RUL projection: {e}")
        return {"error": str(e)}, 500


@app.get("/api/model-performance")
def get_model_performance():
    """
    Compute simple performance metrics (MAE, RMSE) for each loaded model.
    """
    global FULL_DF, MODELS, SCALER_X, SCALER_Y, SEQUENCE_LENGTH

    if FULL_DF is None or SCALER_X is None:
        return {"error": "Data or scaler not loaded"}, 500

    try:
        cutoff = int(len(FULL_DF) * 0.75)
        eval_df = FULL_DF.iloc[:cutoff].reset_index(drop=True)

        exclude_cols = ['timestamp', 'failure', 'RUL']
        feature_cols = [c for c in eval_df.columns if c not in exclude_cols]

        results = {}
        STRIDE = max(1, int(len(eval_df) / 200))

        for name, model in MODELS.items():
            preds = []
            trues = []

            for i in range(0, len(eval_df) - SEQUENCE_LENGTH, STRIDE):
                seq_data = eval_df.iloc[i:i + SEQUENCE_LENGTH]
                if len(seq_data) < SEQUENCE_LENGTH: break
                
                seq_features = SCALER_X.transform(seq_data[feature_cols])
                model_input = np.array([seq_features])

                try:
                    pred_scaled = model.predict(model_input, verbose=0)
                    pred_arr = np.asarray(pred_scaled).ravel()
                    pred_val = float(pred_arr[0]) if pred_arr.size > 0 else 0.0
                    if SCALER_Y is not None:
                        try:
                            p = float(max(0.0, SCALER_Y.inverse_transform(np.array([[pred_val]]))[0][0]))
                        except Exception:
                            p = float(max(0.0, pred_val))
                    else:
                        try:
                            y_max = float(FULL_DF['RUL'].max()) if FULL_DF is not None else 120.0
                            if not np.isfinite(y_max) or y_max <= 0:
                                y_max = 120.0
                        except Exception:
                            y_max = 120.0
                        p = float(max(0.0, pred_val * y_max))
                except Exception:
                    p = 0.0

                a = float(eval_df.iloc[i + SEQUENCE_LENGTH]['RUL'])

                preds.append(p)
                trues.append(a)

            if len(preds) == 0:
                mae, rmse = None, None
            else:
                preds_arr = np.array(preds)
                trues_arr = np.array(trues)
                mae = float(np.mean(np.abs(preds_arr - trues_arr)))
                rmse = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2)))

            # Small sample series for plotting (limit 25)
            full_series = list(zip(preds, trues))
            sample_series = full_series[-25:]
            series = [{"predicted": round(p, 1), "actual": round(a, 1)} for p, a in sample_series]

            drops = [] # (Logic for drops kept simple/empty for now)

            results[name] = {"mae": mae, "rmse": rmse, "series": series, "drops": drops}

        return {"results": results}

    except Exception as e:
        print(f"Error computing model performance: {e}")
        return {"error": str(e)}, 500


@app.get("/api/rul-distribution")
def get_rul_distribution(bins: int = 20):
    global FULL_DF
    if FULL_DF is None: return {"error": "Data not loaded"}, 500
    try:
        rul_values = FULL_DF['RUL'].dropna().values
        if len(rul_values) == 0: return {"bins": [], "counts": []}
        counts, bin_edges = np.histogram(rul_values, bins=bins)
        return {"bins": bin_edges.tolist(), "counts": counts.tolist()}
    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/api/rul-series")
def get_rul_series(stride: int = 100, limit: int = 500):
    global FULL_DF
    if FULL_DF is None: return {"error": "Data not loaded"}, 500
    try:
        data = []
        for i in range(0, len(FULL_DF), max(1, stride)):
            if len(data) >= limit: break
            row = FULL_DF.iloc[i]
            data.append({"index": int(i), "timestamp": str(row.get('timestamp', i)), "rul": float(row.get('RUL', 0))})
        return {"data": data}
    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/api/debug_status")
def debug_status():
    """Return loaded models and scalers and a sample prediction for quick debugging."""
    info = {
        'models': {name: bool(model is not None) for name, model in MODELS.items()},
        'scaler_x': SCALER_X is not None,
        'scaler_y': SCALER_Y is not None,
        'sample': {}
    }

    try:
        if LIVE_DF is None:
            return {'info': info, 'note': 'LIVE_DF not loaded'}

        # Prepare single sequence
        seq = LIVE_DF.iloc[:SEQUENCE_LENGTH]
        exclude_cols = ['timestamp', 'failure', 'RUL']
        feature_cols = [c for c in LIVE_DF.columns if c not in exclude_cols]
        seq_features = seq[feature_cols].values
        scaled = None
        if SCALER_X is not None:
            try:
                scaled = SCALER_X.transform(seq[feature_cols]).reshape(1, SEQUENCE_LENGTH, -1)
            except Exception as e:
                info['scaler_x_error'] = str(e)

        for name, model in MODELS.items():
            if model is None:
                info['sample'][name] = {'loaded': False}
                continue
            try:
                inp = scaled if scaled is not None else seq_features.reshape(1, SEQUENCE_LENGTH, -1)
                pred = model.predict(inp, verbose=0)
                arr = np.asarray(pred).ravel()
                raw = float(arr[0]) if arr.size > 0 else None
                inv = None
                if raw is not None and SCALER_Y is not None:
                    try:
                        inv = float(SCALER_Y.inverse_transform(np.array([[raw]]))[0][0])
                    except Exception as e:
                        inv = f"inv_error: {e}"
                info['sample'][name] = {'loaded': True, 'raw': raw, 'inverse': inv}
            except Exception as e:
                info['sample'][name] = {'error': str(e)}

        return {'info': info}
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)