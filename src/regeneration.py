import pandas as pd
import numpy as np

def regenerate_data():
    print("🚀 Starting Data Regeneration Pipeline...")

    # 1. Load Raw Data
    raw_path = "data/metropt_raw.csv"
    print(f"📂 Loading raw data from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
        # print(f"   📊 Original Row Count: {len(df)}")
    except FileNotFoundError:
        print("❌ Error: File not found.")
        return

    # 2. Fix Timestamp
    print("🕒 Parsing timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- SMART YEAR DETECTION ---
    data_year = df['timestamp'].iloc[0].year
    print(f"   ℹ️ Detected Data Year: {data_year}")
    
    # Base failures are from 2020. We calculate the offset.
    # If data is 2022, we shift failures by +2 years.
    year_offset = data_year - 2020
    
    # 3. Label Failures (With Year Shift)
    print(f"🏷️ Labeling failure events (Shifted by {year_offset} years)...")
    df['failure'] = 0
    
    # Original 2020 Failure Windows
    base_failures = [
        ("2020-04-18 00:00:00", "2020-04-18 23:59:00"), # Failure 1
        ("2020-05-29 23:30:00", "2020-05-30 06:00:00"), # Failure 2
        ("2020-06-05 10:00:00", "2020-06-07 14:30:00"), # Failure 3
        ("2020-07-15 14:30:00", "2020-07-15 19:00:00")  # Failure 4
    ]
    
    # Apply Offset
    shifted_failures = []
    for start, end in base_failures:
        s = pd.Timestamp(start) + pd.DateOffset(years=year_offset)
        e = pd.Timestamp(end) + pd.DateOffset(years=year_offset)
        shifted_failures.append((s, e))
        
        # Mark failure in dataframe
        mask = (df['timestamp'] >= s) & (df['timestamp'] <= e)
        df.loc[mask, 'failure'] = 1
        
    print(f"   First Failure Target shifted to: {shifted_failures[0][0]}")

    # 4. Calculate RUL
    print("⏳ Calculating RUL (Backfill Method)...")
    
    # Create a "Next Failure Time" column initialized with NaNs
    df['next_failure'] = np.nan
    
    # At every failure row, set 'next_failure' to the current time
    df.loc[df['failure'] == 1, 'next_failure'] = df.loc[df['failure'] == 1, 'timestamp']
    
    # Backfill: For every healthy row, 'next_failure' becomes the nearest FUTURE failure time
    df['next_failure'] = df['next_failure'].bfill()
    
    # Calculate difference in hours
    df['RUL'] = (df['next_failure'] - df['timestamp']).dt.total_seconds() / 3600.0
    
    # Fill remaining NaNs (rows after the last failure) with 0
    df['RUL'] = df['RUL'].fillna(0)
    
    # Drop the helper column
    df.drop(columns=['next_failure'], inplace=True)
    
    # Clip RUL to 120h (Piecewise Linear) to help the model
    df['RUL'] = df['RUL'].clip(upper=120)

    # 5. Drop Useless Features
    print("🧹 Dropping GPS and Metadata...")
    useless_cols = ['gpsLat', 'gpsLong', 'gpsQuality', 'gpsSpeed', 'Oil_level', 'Caudal_impulses']
    df.drop(columns=[c for c in useless_cols if c in df.columns], inplace=True)

    # 6. Downsample (CRITICAL for RAM)
    print("📉 Downsampling to 10s intervals...")
    # Resample to 10s and take the mean
    df = df.set_index('timestamp').resample('10s').mean().reset_index()
    # Remove empty gaps
    df = df.dropna()
    # Fix failure binary (mean might make it 0.1, force to 0 or 1)
    df['failure'] = (df['failure'] > 0).astype(int)

    # 7. Save
    output_path = "data/engineered_data.parquet"
    print(f"💾 Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Success! Processed {len(df)} rows.")

if __name__ == "__main__":
    regenerate_data()