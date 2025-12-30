import argparse
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from clearml import Task 
import matplotlib

# Set non-interactive backend for server-side plotting
matplotlib.use('Agg')

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
SEQUENCE_LENGTH = 180
BATCH_SIZE = 256
EPOCHS = 11          # Increased slightly to allow convergence with new scaling
THRESHOLD_HOURS = 24 

# --------------------------------------------------------------------------------
# CUSTOM CALLBACK FOR LIVE LOGGING
# --------------------------------------------------------------------------------
class ClearMLLivePlotting(Callback):
    def __init__(self, task, model_name):
        super().__init__()
        self.task = task
        self.logger = task.get_logger()
        self.model_name = model_name.upper()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Log Train vs Validation Loss
        if 'loss' in logs:
            self.logger.report_scalar(
                title=f"Loss ({self.model_name})", 
                series="Train", 
                value=logs['loss'], 
                iteration=epoch + 1
            )
        if 'val_loss' in logs:
            self.logger.report_scalar(
                title=f"Loss ({self.model_name})", 
                series="Validation", 
                value=logs['val_loss'], 
                iteration=epoch + 1
            )
            
        # Log Train vs Validation MAE
        if 'mae' in logs:
            self.logger.report_scalar(
                title=f"MAE ({self.model_name})", 
                series="Train", 
                value=logs['mae'], 
                iteration=epoch + 1
            )
        if 'val_mae' in logs:
            self.logger.report_scalar(
                title=f"MAE ({self.model_name})", 
                series="Validation", 
                value=logs['val_mae'], 
                iteration=epoch + 1
            )

# --------------------------------------------------------------------------------
# PLOTTING UTILS
# --------------------------------------------------------------------------------
def plot_training_history(history, model_name):
    """Generates and saves a plot of Loss and MAE over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss (MSE)
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{model_name} - Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE (Scaled)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: MAE
    ax2.plot(history.history['mae'], label='Train MAE')
    ax2.plot(history.history['val_mae'], label='Val MAE')
    ax2.set_title(f'{model_name} - Mean Absolute Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (Scaled)')
    ax2.legend()
    ax2.grid(True)

    filename = f"{model_name}_history.png"
    plt.savefig(filename)
    plt.close()
    return filename

# --------------------------------------------------------------------------------
# MODEL ARCHITECTURES
# --------------------------------------------------------------------------------
def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # Removed recurrent_dropout for GPU compatibility (speed)
        LSTM(64, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1) # Linear activation for regression
    ])
    return model

def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    return model

# --------------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", choices=['lstm', 'gru', 'cnn'], help="Choose model type")
    args = parser.parse_args()

    print(f"🚀 Starting Fixed Training for: {args.model.upper()}...")
    
    task = Task.init(project_name="MetroPT Maintenance V2", task_name=f"{args.model.upper()} Training")
    task.connect({"sequence_length": SEQUENCE_LENGTH, "batch_size": BATCH_SIZE, "model_type": args.model})

    # 1. Load Data
    data_path = 'data/engineered_data.parquet'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
    df = pd.read_parquet(data_path)

    # 2. ISOLATE SIMULATION DATA
    sim_cutoff = int(len(df) * 0.75)
    dev_df = df.iloc[:sim_cutoff].copy()
    
    # 3. SPLIT DEVELOPMENT DATA
    target_col = 'RUL'
    exclude_cols = ['timestamp', 'failure', 'RUL'] 
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    train_split_idx = int(len(dev_df) * 0.80)
    train_df = dev_df.iloc[:train_split_idx].copy()
    test_df = dev_df.iloc[train_split_idx:].copy()
    
    # 4. SCALING (CRITICAL FIX: Scale Y independently)
    print("   Scaling Data (X and Y separately)...")
    
    # Scale Features (X)
    scaler_X = StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_df[feature_cols])
    test_X_scaled = scaler_X.transform(test_df[feature_cols])
    
    # Scale Target (Y) - Essential for LSTM/GRU convergence
    scaler_y = StandardScaler()
    train_y_scaled = scaler_y.fit_transform(train_df[[target_col]])
    test_y_scaled = scaler_y.transform(test_df[[target_col]])
    
    # Save Scalers
    with open('scaler_X.pkl', 'wb') as f: pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)

    # 5. Generators
    print(f"   Creating Generators (Length={SEQUENCE_LENGTH})...")
    
    # CRITICAL FIX: shuffle=True for training to prevent overfitting to time
    train_gen = TimeseriesGenerator(
        train_X_scaled, 
        train_y_scaled, 
        length=SEQUENCE_LENGTH, 
        batch_size=BATCH_SIZE,
        shuffle=True 
    )
    
    # shuffle=False for testing to keep sequential order for plotting
    test_gen = TimeseriesGenerator(
        test_X_scaled, 
        test_y_scaled, 
        length=SEQUENCE_LENGTH, 
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 6. Build Model
    input_shape = (SEQUENCE_LENGTH, len(feature_cols))
    if args.model == 'lstm': model = build_lstm(input_shape)
    elif args.model == 'gru':  model = build_gru(input_shape)
    elif args.model == 'cnn':  model = build_cnn(input_shape)
    
    # Compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # 7. Train
    print("   Training...")
    model_filename = f"{args.model}_model.keras"
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), # Increased patience
        ModelCheckpoint(model_filename, save_best_only=True, monitor='val_loss'),
        ClearMLLivePlotting(task, args.model)
    ]
    
    history = model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # Save History
    with open(f"history_{args.model}.pkl", 'wb') as f:
        pickle.dump(history.history, f)

    # 8. Log Graphs
    plot_file = plot_training_history(history, args.model.upper())
    logger = task.get_logger()
    logger.report_image("Training Curves", f"{args.model.upper()} Loss/MAE", local_path=plot_file)

    # 9. Evaluate Metrics (CRITICAL FIX: Inverse Transform)
    print(f"\n📊 Evaluating {args.model.upper()}...")
    
    # Predict (Output is Scaled)
    preds_scaled = model.predict(test_gen)
    
    # Inverse Transform Predictions to 'Hours'
    preds = scaler_y.inverse_transform(preds_scaled)
    
    # Get Actuals (Extract from DF to ensure we have original unscaled values)
    # The generator consumes the first 'SEQUENCE_LENGTH' points, so we slice them off
    y_test_aligned = test_df[target_col].values[SEQUENCE_LENGTH:]
    
    # Ensure shapes match (Generator might drop last incomplete batch if configured, though usually doesn't)
    min_len = min(len(preds), len(y_test_aligned))
    preds = preds[:min_len]
    y_test_aligned = y_test_aligned[:min_len]

    # Debugging Metric Issues
    print(f"   Test Samples: {len(preds)}")
    if len(preds) < 100:
        print("   WARNING: Very few test samples. RMSE and MAE might look identical due to small sample size.")

    # Calculate Metrics on Real Hours
    rmse = np.sqrt(mean_squared_error(y_test_aligned, preds))
    mae = mean_absolute_error(y_test_aligned, preds)
    
    errors = np.abs(y_test_aligned - preds.flatten())
    acc_within_threshold = np.mean(errors <= THRESHOLD_HOURS) * 100

    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   Accuracy @ {THRESHOLD_HOURS}h: {acc_within_threshold:.2f}%")

    # Log Scalar Metrics
    logger.report_scalar("RMSE Comparison", args.model.upper(), rmse, iteration=1)
    logger.report_scalar("MAE Comparison", args.model.upper(), mae, iteration=1)
    logger.report_scalar("Accuracy @ 24h", args.model.upper(), acc_within_threshold, iteration=1)

if __name__ == "__main__":
    train()