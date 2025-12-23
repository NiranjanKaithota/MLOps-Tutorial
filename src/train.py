import argparse
import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clearml import Task

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
SEQUENCE_LENGTH = 180  # 30 Minutes of context
BATCH_SIZE = 256       # Batches per step
EPOCHS = 12

# --------------------------------------------------------------------------------
# MODEL ARCHITECTURES
# --------------------------------------------------------------------------------
def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
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

    print(f"🚀 Starting V2 Training Pipeline (Generator Mode) for: {args.model.upper()}...")
    
    # Initialize ClearML
    task = Task.init(project_name="MetroPT Maintenance V2", task_name=f"{args.model.upper()} Generator Training")
    task.connect({"sequence_length": SEQUENCE_LENGTH, "batch_size": BATCH_SIZE, "model_type": args.model})

    # 1. Load Data
    data_path = 'data/engineered_data.parquet'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
    
    df = pd.read_parquet(data_path)

    # 2. Split Features/Target
    target_col = 'RUL'
    exclude_cols = ['timestamp', 'failure', 'RUL'] 
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 3. Chronological Split (70/30)
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # 4. Scaling
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 5. Create Generators (THE FIX: Zero Memory Overhead)
    print(f"   Creating Data Generators (Length={SEQUENCE_LENGTH})...")
    
    # Train Generator
    train_gen = TimeseriesGenerator(
        train_scaled, 
        train_df[target_col].values,
        length=SEQUENCE_LENGTH, 
        batch_size=BATCH_SIZE
    )

    # Test Generator
    test_gen = TimeseriesGenerator(
        test_scaled, 
        test_df[target_col].values,
        length=SEQUENCE_LENGTH, 
        batch_size=BATCH_SIZE
    )
    
    print(f"   Train Batches: {len(train_gen)}")
    print(f"   Test Batches:  {len(test_gen)}")

    # 6. Build Selected Model
    # Input shape for generator is (Sequence_Length, Num_Features)
    input_shape = (SEQUENCE_LENGTH, len(feature_cols))
    
    if args.model == 'lstm':
        model = build_lstm(input_shape)
    elif args.model == 'gru':
        model = build_gru(input_shape)
    elif args.model == 'cnn':
        model = build_cnn(input_shape)
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # 7. Train
    model_filename = f"{args.model}_model.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(model_filename, save_best_only=True, monitor='val_loss')
    ]
    
    # Note: fit() works with generators automatically
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Evaluate
    print(f"\n📊 Evaluating {args.model.upper()}...")
    
    # We must predict on the generator to avoid memory crash
    preds = model.predict(test_gen)
    
    # Generators shuffle data by default? No, TimeseriesGenerator does NOT shuffle order 
    # BUT it cuts off the first 'sequence_length' rows. 
    # We need to align y_test with the generator's output.
    
    # Get the actual targets corresponding to the generator predictions
    # The generator starts outputting at index `sequence_length`
    y_test_aligned = test_df[target_col].values[SEQUENCE_LENGTH:]
    
    # Trim predictions if there's a slight batch mismatch (rare but safe to check)
    preds = preds[:len(y_test_aligned)]
    
    rmse = np.sqrt(mean_squared_error(y_test_aligned, preds))
    mae = mean_absolute_error(y_test_aligned, preds)
    r2 = r2_score(y_test_aligned, preds)

    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R2:   {r2:.4f}")

    # Log to ClearML
    # logger = task.get_logger()
    # logger.report_scalar("Comparison", "RMSE", rmse, iteration=1, series=args.model)
    # logger.report_scalar("Comparison", "R2", r2, iteration=1, series=args.model)

    # Log to ClearML (FIXED)
    logger = task.get_logger()
    
    # Title="RMSE Comparison", Series=ModelName (e.g., 'LSTM'), Value=rmse
    logger.report_scalar("RMSE Comparison", args.model.upper(), rmse, iteration=1)
    
    # Title="R2 Comparison", Series=ModelName, Value=r2
    logger.report_scalar("R2 Comparison", args.model.upper(), r2, iteration=1)

if __name__ == "__main__":
    train()