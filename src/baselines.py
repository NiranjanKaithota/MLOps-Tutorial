import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clearml import Task

# Initialize ClearML
task = Task.init(project_name="MetroPT Maintenance V2", task_name="Baseline Models Training")

def train_baselines():
    print("🚀 Starting Baseline Training (Lasso & Random Forest)...")
    
    # 1. Load Data
    try:
        df = pd.read_parquet('data/engineered_data.parquet')
    except FileNotFoundError:
        print("❌ Error: data/engineered_data.parquet not found.")
        return

    # 2. Split Features/Target
    target_col = 'RUL'
    exclude_cols = ['timestamp', 'failure', 'RUL']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 3. Chronological Split (70/30)
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"   Train Rows: {len(train_df)}")
    print(f"   Test Rows:  {len(test_df)}")

    # 4. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    y_train = train_df[target_col]
    X_test = scaler.transform(test_df[feature_cols])
    y_test = test_df[target_col]

    # --- MODEL 1: LASSO (Linear Baseline) ---
    print("\n🔹 Training LASSO...")
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    preds_lasso = lasso.predict(X_test)
    
    rmse_l = np.sqrt(mean_squared_error(y_test, preds_lasso))
    r2_l = r2_score(y_test, preds_lasso)
    print(f"   [LASSO] RMSE: {rmse_l:.4f} | R2: {r2_l:.4f}")
    
    # --- MODEL 2: RANDOM FOREST (Non-Linear Baseline) ---
    print("\n🔹 Training RANDOM FOREST (This may take a minute)...")
    # Limited depth to prevent overfitting and speed up training
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    
    rmse_rf = np.sqrt(mean_squared_error(y_test, preds_rf))
    r2_rf = r2_score(y_test, preds_rf)
    print(f"   [RANDOM FOREST] RMSE: {rmse_rf:.4f} | R2: {r2_rf:.4f}")

    # Log to ClearML (CORRECTED SYNTAX)
    logger = task.get_logger()
    # title, series, value, iteration
    logger.report_scalar("RMSE Comparison", "Lasso", rmse_l, iteration=1)
    logger.report_scalar("RMSE Comparison", "RandomForest", rmse_rf, iteration=1)
    
    # Save the winner (RF)
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("\n✅ Baselines Complete. Random Forest saved.")

if __name__ == "__main__":
    train_baselines()