import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')
with open('feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

ticker = 'FPT'
# Query last 10 trading days of historical normalized features
df_norm_hist = pd.read_sql(text("""
    SELECT * FROM daily_normalized_data 
    WHERE ticker = :ticker 
    ORDER BY date DESC LIMIT 10
"""), engine, params={'ticker': ticker})

df_raw = pd.read_sql(text("""
    SELECT date, close FROM daily_raw_data 
    WHERE ticker = :ticker 
    ORDER BY date DESC LIMIT 10
"""), engine, params={'ticker': ticker})

# Let's inspect
print("--- RAW DATA ---")
print(df_raw[['date', 'close']])
print("--- NORM DATA ---")
print(df_norm_hist[['date', 'close_LogReturn']])

# Run predictions
X_df = df_norm_hist.copy()
for f in feature_cols:
    if f not in X_df.columns:
        X_df[f] = 0.0
    else:
        X_df[f] = X_df[f].fillna(0.0)

X_matrix = X_df[feature_cols]
probs = model.predict_proba(X_matrix)[:, 1]

for idx, row in df_norm_hist.iterrows():
    date_str = str(row['date'])
    prob = probs[idx]
    close_row = df_raw[df_raw['date'] == row['date']]
    close_val = float(close_row['close'].values[0]) if not close_row.empty else 0.0
    pred_val = close_val * np.exp((prob - 0.5) * 0.08)
    print(f"Date: {date_str} | Close: {close_val} | Prob: {prob:.4f} | Pred: {pred_val:.2f}")
