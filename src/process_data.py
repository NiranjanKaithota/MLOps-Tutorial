import pandas as pd
import numpy as np

def process_data():
    print("🚀 Starting V2 Data Pipeline (Threshold Variance Mode)...")
    
    raw_path = 'data/metropt_raw.csv'
    print(f"   📂 Loading {raw_path}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print("   ❌ Error: File not found. Please ensure 'data/metropt_raw.csv' exists.")
        return

    # 2. DateTime Parsing
    print("   🕒 Parsing timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 3. Drop Useless ID Column
    if 'id' in df.columns:
        print("   🧹 Dropping 'id' column...")
        df = df.drop(columns=['id'])

    # 4. Label Ground Truth Failures (Official 2020 Logs)
    print("   🏷️ Labeling Failures...")
    df['failure'] = 0
    failure_events = [
        pd.Timestamp("2020-04-18 00:00:00"),
        pd.Timestamp("2020-05-29 23:30:00"),
        pd.Timestamp("2020-06-05 10:00:00"),
        pd.Timestamp("2020-07-15 14:30:00")
    ]
    
    for fail_time in failure_events:
        mask_failure = (df['timestamp'] >= fail_time) & (df['timestamp'] <= fail_time + pd.Timedelta(hours=1))
        df.loc[mask_failure, 'failure'] = 1

    # 5. Calculate RUL (Piecewise Linear)
    print("   ⏳ Calculating RUL...")
    df['next_failure'] = np.nan
    df.loc[df['failure'] == 1, 'next_failure'] = df.loc[df['failure'] == 1, 'timestamp']
    df['next_failure'] = df['next_failure'].bfill()
    df['RUL'] = (df['next_failure'] - df['timestamp']).dt.total_seconds() / 3600.0
    df['RUL'] = df['RUL'].fillna(0)
    df.drop(columns=['next_failure'], inplace=True)

    # CAP RUL at 120h
    df['RUL'] = df['RUL'].clip(upper=120)

    # 6. AUTOMATIC LOW VARIANCE FILTER (Updated)
    print("   🔍 Checking for low-variance features...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features_to_check = [c for c in numeric_cols if c not in ['RUL', 'failure']]
    
    # --- CHANGE START: Threshold Logic ---
    VARIANCE_THRESHOLD = 0.001  # If std dev is less than this, drop it
    low_variance_cols = []
    
    for col in features_to_check:
        std_dev = df[col].std()
        print(col, std_dev)
        if std_dev < VARIANCE_THRESHOLD:
            low_variance_cols.append(col)
            print(f"      Running check on {col}: std_dev = {std_dev:.6f} -> {'❌ DROP' if std_dev < VARIANCE_THRESHOLD else '✅ KEEP'}")
            
    if low_variance_cols:
        print(f"      ⚠️ REMOVING LOW VARIANCE COLUMNS: {low_variance_cols}")
        df = df.drop(columns=low_variance_cols)
    else:
        print("      ✅ All features have sufficient variance.")
    # --- CHANGE END ---

    # 7. Drop Metadata
    useless_metadata = ['gpsLat', 'gpsLong', 'gpsSpeed', 'gpsQuality', 'Caudal_impulses']
    df = df.drop(columns=[c for c in useless_metadata if c in df.columns], errors='ignore')

    # 8. Downsample (10s)
    print("   📉 Downsampling to 10s intervals...")
    df = df.set_index('timestamp').resample('10s').mean().reset_index()
    df = df.dropna()
    df['failure'] = (df['failure'] > 0).astype(int)

    # 9. Save
    output_path = 'data/engineered_data.parquet'
    print(f"   💾 Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Pipeline Complete.")
    print(f"   Final Shape: {df.shape}")
    print(f"   Active Features: {list(df.columns)}")

if __name__ == "__main__":
    process_data()