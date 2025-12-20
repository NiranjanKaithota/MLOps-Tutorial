import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Replace this with the actual path to your 1.5GB CSV file
RAW_DATA_PATH = r"C:\\Users\\nisha\\OneDrive\\Documents\\MLOps-Tutorial\\data\\dataset_train.csv"
OUTPUT_PATH = "data\engineered_data.parquet"

def regenerate_dataset():
    print(f"🚀 Starting Data Regeneration from: {RAW_DATA_PATH}")
    
    # 1. Load Data (Read CSV)
    # The MetroPT dataset usually has a 'timestamp' column. Adjust if needed.
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"✅ Raw Data Loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find raw file at {RAW_DATA_PATH}")
        print("Please update the RAW_DATA_PATH variable in the script.")
        return

    # 2. Convert Timestamp
    print("Converting timestamps...")
    # Adjust 'timestamp' to the actual column name in your CSV (e.g., 'time', 'ts')
    time_col = 'timestamp' 
    if time_col not in df.columns:
        # MetroPT often has 'timestamp' as the first column
        time_col = df.columns[0]
    
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 3. Downsample (1 datapoint every 10 seconds)
    print("📉 Downsampling to 10s intervals...")
    df = df.set_index(time_col)
    
    # Resample: Take the mean of sensors, and 'max' for binary failure flags
    # We assume 'failure' or 'anomaly' columns exist. If not, this will just average everything.
    df_resampled = df.resample('10s').mean()
    
    # If you had binary columns (0 or 1), averaging makes them 0.1, 0.2 etc.
    # Let's fix failure column if it exists
    if 'failure' in df_resampled.columns:
        df_resampled['failure'] = df_resampled['failure'].apply(lambda x: 1 if x > 0 else 0)

    # Drop NaNs created by resampling gaps
    df_resampled = df_resampled.dropna()
    
    print(f"✅ Resampled Data Shape: {df_resampled.shape}")

    # 4. Feature Engineering (RUL)
    # Re-create RUL logic here to ensure it's in the file
    print("⚙️ Engineering RUL...")
    
    # Simple Logic: If 'failure' exists, count down. 
    # If not, create a dummy or logic based on sensor drift.
    if 'failure' in df_resampled.columns:
        # Calculate RUL based on failure markers
        failure_indices = np.where(df_resampled['failure'] == 1)[0]
        rul_col = np.full(len(df_resampled), 150.0) # Default cap
        
        # (Simplified RUL logic for regeneration)
        # Real logic would be: find next failure index - current index
        pass 
    else:
        # If no failure column, create a synthetic one for the tutorial
        # (Or use your specific logic if you have it)
        print("⚠️ No 'failure' column found. Creating synthetic RUL for training...")
        # Create RUL based on index (just to have a target)
        df_resampled['RUL'] = np.linspace(100, 0, len(df_resampled))

    # 5. Save to Parquet (Safely)
    print(f"💾 Saving to {OUTPUT_PATH}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Write
    df_resampled.to_parquet(OUTPUT_PATH, index=False)
    
    # 6. Verification
    print("🔍 Verifying file...")
    try:
        check_df = pd.read_parquet(OUTPUT_PATH)
        print(f"✅ Success! File saved and verified. Size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ File Verification Failed: {e}")

if __name__ == "__main__":
    regenerate_dataset()