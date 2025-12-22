import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_parquet('data/engineered_data.parquet')

# Prep Data
target_col = 'RUL'
drop_cols = [target_col, 'failure', 'timestamp', 'failure_column']
feature_cols = [c for c in df.columns if c not in drop_cols]

# Train a quick RF model
print("⏳ Training diagnostic model...")
X = df[feature_cols]
y = df[target_col]
print(X.head())
print("--------------------------------")
print(y.info())
# Use a small sample to be fast
X_sample = X.sample(n=50000, random_state=42)
y_sample = y.sample(n=50000, random_state=42)

model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
model.fit(X_sample, y_sample)

# Get Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n📊 FEATURE IMPORTANCE RANKING:")
print(f"{'Rank':<5} | {'Feature Name':<20} | {'Importance':<10}")
print("-" * 45)

useless_features = []

for f in range(X.shape[1]):
    idx = indices[f]
    score = importances[idx]
    feature_name = feature_cols[idx]
    print(f"{f+1:<5} | {feature_name:<20} | {score:.6f}")
    
    # Flag features with < 0.1% importance
    if score < 0.001: 
        useless_features.append(feature_name)

print("\n🗑️ RECOMMENDED TO DROP:", useless_features)
