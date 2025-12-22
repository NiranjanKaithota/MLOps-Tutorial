import pandas as pd
import numpy as np

def process_data():
    print("🚀 Starting Data Processing (Reverting to 1.5GB Source)...")
    
    # 1. Load the massive CSV
    # Ensure your 1.5GB file is named 'metropt_raw.csv' inside the 'data' folder
    raw_path = 'data/metropt_raw.csv'
    print(f"   Loading {raw_path} (This takes time)...")
    
    try:
        df_raw = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {raw_path}. Please check the file name.")
        return

    print(f"   Number of entries in raw CSV: {len(df_raw)}")  # Print number of entries in raw CSV

    # Parse timestamps
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    
    # 2. Define Known Failure Timestamps (From Maintenance Logs)
    failure_events = [
        pd.Timestamp("2020-04-18 00:00:00"),
        pd.Timestamp("2020-05-29 23:30:00"),
        pd.Timestamp("2020-06-05 10:00:00"),
        pd.Timestamp("2020-07-15 14:30:00")
    ]
    
    print(f"   Labeling {len(failure_events)} failure events...")

    # 3. Engineer 'RUL' & Labels
    df = df_raw.sort_values('timestamp').reset_index(drop=True)
    df['RUL'] = np.nan
    df['failure'] = 0
    
    for fail_time in failure_events:
        # Mark failure window (1 hour)
        mask_failure = (df['timestamp'] >= fail_time) & (df['timestamp'] <= fail_time + pd.Timedelta(hours=1))
        df.loc[mask_failure, 'failure'] = 1
        
        # Calculate RUL
        time_until_fail = (fail_time - df['timestamp']).dt.total_seconds() / 3600
        
        # Valid RUL is positive and we take the shortest distance to a failure
        mask_valid = time_until_fail >= 0
        mask_update = mask_valid & (df['RUL'].isna() | (time_until_fail < df['RUL']))
        df.loc[mask_update, 'RUL'] = time_until_fail

    # Drop data after the last known failure
    df = df.dropna(subset=['RUL'])

    # --- CLIP RUL (Piecewise Linear) ---
    # This prevents the "flatline" issue we saw earlier.
    MAX_RUL = 120 
    df['RUL'] = df['RUL'].clip(upper=MAX_RUL)

    # 4. Drop Useless Features
    useless_cols = ['gpsLat', 'gpsLong', 'gpsSpeed', 'gpsQuality', 'Oil_level', 'Caudal_impulses']
    df = df.drop(columns=[c for c in useless_cols if c in df.columns], errors='ignore')
    print(f"   Dropped columns: {useless_cols}")

    # 5. Downsample (1 sample per 10 seconds)
    RESAMPLE_RATE = '10s'  # Changed to 10s as requested
    print(f"   Downsampling to {RESAMPLE_RATE} intervals...")
    
    df = df.set_index('timestamp').resample(RESAMPLE_RATE).mean().reset_index()
    
    # Fix NaNs created by gaps in recording
    original_len = len(df)
    df = df.dropna()
    print(f"   ⚠️ Removed {original_len - len(df)} empty rows (gaps).")
    
    # Fix binary failure column (mean() might make it 0.1)
    df['failure'] = (df['failure'] > 0).astype(int)

    print(f"   Number of entries in downsampled CSV: {len(df)}")  # Print number of entries in downsampled CSV

    # 6. Save DEBUG CSV (For your inspection)
    csv_path = 'data/debug_labeled_data.csv'
    print(f"   💾 Saving debug CSV to {csv_path}...")
    df.to_csv(csv_path, index=False)

    # 7. Save Final Parquet (For Model)
    output_path = 'data/engineered_data.parquet'
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Success! Data is ready.")
    print(f"   Final Shape: {df.shape}")

if __name__ == "__main__":
    process_data()