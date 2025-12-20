import argparse
import pandas as pd
import numpy as np
import time
import os
import pickle

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# MLOps Import
from clearml import Task
# Evidently for monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

# --------------------------------------------------------------------------------
# 1. ARGUMENT PARSING & CLEARML SETUP
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="lasso", 
                    help="Model to train: lasso, random_forest, or lstm")
args = parser.parse_args()

# Initialize ClearML Task
# reuse_last_task_id=False ensures every run creates a NEW entry in the dashboard
task = Task.init(
    project_name="MetroPT Maintenance", 
    task_name=f"Training - {args.model}", 
    reuse_last_task_id=False
)

# Log the model type explicitly
task.connect({"model_type": args.model})


# --------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def create_lstm_sequences(data, features, target, sequence_length):
    """
    Converts tabular data into 3D sequences for LSTM [samples, time_steps, features]
    """
    X, y = [], []
    # Convert DataFrame to numpy array for speed
    data_array = data[features].values
    target_array = data[target].values
    
    for i in range(len(data) - sequence_length):
        X.append(data_array[i : i + sequence_length])
        y.append(target_array[i + sequence_length - 1])
        
    return np.array(X), np.array(y)

def save_model_artifact(model, model_name):
    """
    Saves the model locally and uploads it to ClearML
    """
    filename = f"{model_name}_model.pkl"
    
    if model_name == 'lstm':
        # Keras format
        filename = f"{model_name}_model.keras"
        model.save(filename)
    else:
        # Pickle format for Sklearn
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            
    print(f"Model saved locally as {filename}")


# --------------------------------------------------------------------------------
# 3. MAIN TRAINING LOGIC
# --------------------------------------------------------------------------------
def train(model_name, sequence_length=30):
    print(f"🚀 Starting training pipeline for: {model_name}")
    
    # --- A. Load Data ---
    data_path = 'data\\engineered_data.parquet'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}. Did you copy your 12MB parquet file into the 'data' folder?")
        
    df = pd.read_parquet(data_path)
    print(f"Data Loaded. Shape: {df.shape}")

    target_column = 'RUL'
    # Dynamic feature selection: Exclude target, timestamps, and failure flags
    feature_columns = [col for col in df.columns if col not in [target_column, 'failure', 'timestamp', 'failure_column']]

    # --- B. Preprocessing ---
    # 1. Split Data (Time-based split is better for RUL, but random is standard for baseline comparisons)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=(0.15/0.85), random_state=42)

    # 2. Scale Data
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    val_df[feature_columns] = scaler.transform(val_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    print("Data Split and Scaled.")
    
    start_time = time.time()
    model = None
    preds = None

    # --- C. Model Training ---
    if model_name == 'lasso':
        print("Training Lasso Regression...")
        model = Lasso(alpha=0.1, random_state=42)
        model.fit(train_df[feature_columns], train_df[target_column])
        preds = model.predict(test_df[feature_columns])
        
    elif model_name == 'random_forest':
        print("Training Random Forest (this may take a moment)...")
        # n_jobs=-1 uses all CPU cores for faster training
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[feature_columns], train_df[target_column])
        preds = model.predict(test_df[feature_columns])

    elif model_name == 'lstm':
        print("Training LSTM Neural Network...")
        # Prepare sequences
        X_train, y_train = create_lstm_sequences(train_df, feature_columns, target_column, sequence_length)
        X_val, y_val = create_lstm_sequences(val_df, feature_columns, target_column, sequence_length)
        X_test, y_test = create_lstm_sequences(test_df, feature_columns, target_column, sequence_length)

        n_features = X_train.shape[2]
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # ClearML automatically tracks Keras training (Loss vs Epochs)
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=20, # You can reduce this to 5 for a quick test
                  batch_size=64,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
        
        preds = model.predict(X_test)
        # Align test dataframe with sequences (sequences shorten data by sequence_length)
        test_df = test_df.iloc[sequence_length:]

    # --- D. Evaluation ---
    training_duration = time.time() - start_time
    
    rmse = np.sqrt(mean_squared_error(test_df[target_column], preds))
    mae = mean_absolute_error(test_df[target_column], preds)
    r2 = r2_score(test_df[target_column], preds)

    print("-" * 30)
    print(f"MODEL: {model_name.upper()}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Time: {training_duration:.2f} sec")
    print("-" * 30)

    # --- E. Logging to ClearML ---
    logger = task.get_logger()
    logger.report_scalar(title="Performance", series="RMSE", value=rmse, iteration=1)
    logger.report_scalar(title="Performance", series="MAE", value=mae, iteration=1)
    logger.report_scalar(title="Performance", series="R2 Score", value=r2, iteration=1)
    logger.report_scalar(title="System", series="Training Time", value=training_duration, iteration=1)
    
    # --- F. Evidently Reports (Data Drift + Regression Performance) ---
    try:
        # Ensure preds is a 1d array
        preds_arr = np.array(preds).ravel()

        test_predictions_df = test_df.copy()
        test_predictions_df['prediction'] = preds_arr

        # Create a combined report with data drift and regression performance
        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])

        # Run report: reference=train, current=test (with predictions)
        report.run(reference_data=train_df.reset_index(drop=True),
                   current_data=test_predictions_df.reset_index(drop=True))

        report_name = f"evidently_report_{model_name}.html"
        report.save_html(report_name)
        print(f"Evidently report saved as {report_name}")

        # Upload the report as an artifact to ClearML
        try:
            task.upload_artifact(name=f"evidently_report_{model_name}", artifact_object=report_name)
        except Exception:
            # Not critical if upload fails locally
            print("Warning: failed to upload Evidently report to ClearML task.")
    except Exception as e:
        print(f"Evidently report generation failed: {e}")
    
    # Save and upload model artifact
    save_model_artifact(model, model_name)
    print("Training Complete.")

if __name__ == "__main__":
    train(args.model)