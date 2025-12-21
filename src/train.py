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

# --------------------------------------------------------------------------------
# 1. ARGUMENT PARSING & CLEARML SETUP
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="lasso", 
                    help="Model to train: lasso, random_forest, or lstm")
args = parser.parse_args()

# Initialize ClearML Task
task = Task.init(
    project_name="MetroPT Maintenance", 
    task_name=f"Training - {args.model}", 
    reuse_last_task_id=False
)
task.connect({"model_type": args.model})

# --------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def create_lstm_sequences(data, features, target, sequence_length):
    X, y = [], []
    data_array = data[features].values
    target_array = data[target].values
    
    for i in range(len(data) - sequence_length):
        X.append(data_array[i : i + sequence_length])
        y.append(target_array[i + sequence_length - 1])
        
    return np.array(X), np.array(y)

def save_model_artifact(model, model_name):
    filename = f"{model_name}_model.pkl"
    if model_name == 'lstm':
        filename = f"{model_name}_model.keras"
        model.save(filename)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    print(f"Model saved locally as {filename}")

# --------------------------------------------------------------------------------
# 3. MAIN TRAINING LOGIC
# --------------------------------------------------------------------------------
def train(model_name, sequence_length=30):
    print(f"🚀 Starting training pipeline for: {model_name}")
    
    data_path = 'data/engineered_data.parquet'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}")
        
    df = pd.read_parquet(data_path)
    
    target_column = 'RUL'
    # Exclude non-feature columns
    feature_columns = [col for col in df.columns if col not in [target_column, 'failure', 'timestamp', 'failure_column']]

    # Split Data
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=(0.15/0.85), random_state=42)

    # Scale Data
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    val_df[feature_columns] = scaler.transform(val_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    start_time = time.time()
    model = None
    preds = None

    if model_name == 'lasso':
        model = Lasso(alpha=0.1, random_state=42)
        model.fit(train_df[feature_columns], train_df[target_column])
        preds = model.predict(test_df[feature_columns])
        
    elif model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[feature_columns], train_df[target_column])
        preds = model.predict(test_df[feature_columns])

    elif model_name == 'lstm':
        X_train, y_train = create_lstm_sequences(train_df, feature_columns, target_column, sequence_length)
        X_val, y_val = create_lstm_sequences(val_df, feature_columns, target_column, sequence_length)
        X_test, y_test = create_lstm_sequences(test_df, feature_columns, target_column, sequence_length)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
        
        preds = model.predict(X_test)
        test_df = test_df.iloc[sequence_length:]

    # Evaluation
    training_duration = time.time() - start_time
    rmse = np.sqrt(mean_squared_error(test_df[target_column], preds))
    mae = mean_absolute_error(test_df[target_column], preds)
    r2 = r2_score(test_df[target_column], preds)

    print(f"MODEL: {model_name.upper()} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    # Logging
    logger = task.get_logger()
    logger.report_scalar("Performance", "RMSE", rmse, iteration=1)
    logger.report_scalar("Performance", "MAE", mae, iteration=1)
    logger.report_scalar("Performance", "R2 Score", r2, iteration=1)
    
    save_model_artifact(model, model_name)

if __name__ == "__main__":
    train(args.model)