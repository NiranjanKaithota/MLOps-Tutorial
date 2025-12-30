import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

# CONFIG
SEQ_LENGTH = 180
DATA_PATH = '../data/engineered_data.parquet' # Adjust path if needed
MODEL_PATH = 'D:\Projects\EL Models\MLOps\MLOps-Tutorial\gru_model.keras'                # Using your Champion Model
SCALER_PATH = 'D:\Projects\EL Models\MLOps\MLOps-Tutorial\scaler.pkl'
OUTPUT_FILE = '../metropt-dashboard/public/model_performance.json' # Save directly to Frontend public folder

print("⏳ Loading Data & Model...")
df = pd.read_parquet(DATA_PATH)
model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# PREPARE TEST SET (Last 20% of data)
split_idx = int(len(df) * 0.80) 
test_df = df.iloc[split_idx:].reset_index(drop=True)

# Select features (exclude targets)
feature_cols = [c for c in df.columns if c not in ['timestamp', 'failure', 'RUL']]
X_test_raw = scaler.transform(test_df[feature_cols])
y_test_raw = test_df['RUL'].values

# Create Sequences (Optimization: Stride of 100 to reduce file size for web)
# We don't need every single second for the graph, every 10-15 mins is fine for visualization
STRIDE = 100 
X_seq, y_seq, timestamps = [], [], []

print("🔄 Generating Sequences...")
for i in range(0, len(X_test_raw) - SEQ_LENGTH, STRIDE):
    X_seq.append(X_test_raw[i : i + SEQ_LENGTH])
    y_seq.append(y_test_raw[i + SEQ_LENGTH])
    timestamps.append(str(test_df['timestamp'].iloc[i + SEQ_LENGTH]))

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"🧠 Running Inference on {len(X_seq)} samples...")
predictions = model.predict(X_seq, verbose=1).flatten()

# Calculate Accuracy Metric
mae = mean_absolute_error(y_seq, predictions)
print(f"✅ MAE on this set: {mae:.2f}")

# FORMAT DATA FOR FRONTEND
# We want a list of objects: { time: "...", actual: 100, predicted: 98 }
chart_data = []
for i in range(len(y_seq)):
    chart_data.append({
        "time": timestamps[i],
        "Actual RUL": round(float(y_seq[i]), 1),
        "Predicted RUL": round(float(predictions[i]), 1)
    })

# SAVE TO JSON
with open(OUTPUT_FILE, 'w') as f:
    json.dump(chart_data, f)

print(f"🎉 Success! Performance data saved to {OUTPUT_FILE}")