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
matplotlib.use('Agg')

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
SEQUENCE_LENGTH = 180
BATCH_SIZE = 256
EPOCHS = 11
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
# PLOTTING UTILS (Static Backup)
# --------------------------------------------------------------------------------
def plot_training_history(history, model_name):
    """Generates and saves a plot of Loss and MAE over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss (MSE)
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{model_name} - Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: MAE
    ax2.plot(history.history['mae'], label='Train MAE')
    ax2.plot(history.history['val_mae'], label='Val MAE')
    ax2.set_title(f'{model_name} - Mean Absolute Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Hours Error')
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
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout = 0.0, kernel_regularizer = l2(0.001)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
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

    print(f"🚀 Starting V2.3 Training (Live Logging) for: {args.model.upper()}...")
    
    task = Task.init(project_name="MetroPT Maintenance V2", task_name=f"{args.model.upper()} Training")
    task.connect({"sequence_length": SEQUENCE_LENGTH, "batch_size": BATCH_SIZE, "model_type": args.model})

    # 1. Load Data
    data_path = 'data/engineered_data.parquet'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
    df = pd.read_parquet(data_path)

    # 2. ISOLATE SIMULATION DATA (Strict Split)
    sim_cutoff = int(len(df) * 0.75)
    dev_df = df.iloc[:sim_cutoff].copy()
    
    # 3. SPLIT DEVELOPMENT DATA (Train vs Test)
    target_col = 'RUL'
    exclude_cols = ['timestamp', 'failure', 'RUL'] 
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    train_split_idx = int(len(dev_df) * 0.80)
    train_df = dev_df.iloc[:train_split_idx].copy()
    test_df = dev_df.iloc[train_split_idx:].copy()
    
    # 4. Scaling
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 5. Generators
    print(f"   Creating Generators (Length={SEQUENCE_LENGTH})...")
    train_gen = TimeseriesGenerator(train_scaled, train_df[target_col].values, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
    test_gen = TimeseriesGenerator(test_scaled, test_df[target_col].values, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)

    # 6. Build Model
    input_shape = (SEQUENCE_LENGTH, len(feature_cols))
    if args.model == 'lstm': model = build_lstm(input_shape)
    elif args.model == 'gru':  model = build_gru(input_shape)
    elif args.model == 'cnn':  model = build_cnn(input_shape)
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # 7. Train (With LIVE Callback)
    print("   Training...")
    model_filename = f"{args.model}_model.keras"
    
    # Define Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(model_filename, save_best_only=True, monitor='val_loss'),
        ClearMLLivePlotting(task, args.model)  # <--- NEW LIVE LOGGER
    ]
    
    history = model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # Save History for Comparison
    history_filename = f"history_{args.model}.pkl"
    with open(history_filename, 'wb') as f:
        pickle.dump(history.history, f)

    # 8. Generate & Log Final Graphs (Backup)
    plot_file = plot_training_history(history, args.model.upper())
    logger = task.get_logger()
    logger.report_image("Training Curves", f"{args.model.upper()} Loss/MAE", local_path=plot_file)

    # 9. Evaluate Metrics
    print(f"\n📊 Evaluating {args.model.upper()}...")
    preds = model.predict(test_gen)
    y_test_aligned = test_df[target_col].values[SEQUENCE_LENGTH:]
    preds = preds[:len(y_test_aligned)]
    
    rmse = np.sqrt(mean_squared_error(y_test_aligned, preds))
    mae = mean_absolute_error(y_test_aligned, preds)
    
    errors = np.abs(y_test_aligned - preds.flatten())
    acc_within_threshold = np.mean(errors <= THRESHOLD_HOURS) * 100

    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   Accuracy @ {THRESHOLD_HOURS}h: {acc_within_threshold:.2f}%")

    # logger.report_scalar("Comparison", "RMSE", rmse, iteration=1, series=args.model)
    # logger.report_scalar("Comparison", "MAE", mae, iteration=1, series=args.model)
    # logger.report_scalar("Comparison", "Acc_24h", acc_within_threshold, iteration=1, series=args.model)

    # Log Scalar Metrics (CORRECTED SYNTAX)
    # Syntax: report_scalar(title, series, value, iteration)
    
    # 1. RMSE Comparison
    logger.report_scalar("RMSE Comparison", args.model.upper(), rmse, iteration=1)
    
    # 2. MAE Comparison
    logger.report_scalar("MAE Comparison", args.model.upper(), mae, iteration=1)
    
    # 3. Accuracy Comparison
    logger.report_scalar("Accuracy @ 24h", args.model.upper(), acc_within_threshold, iteration=1)

if __name__ == "__main__":
    train()