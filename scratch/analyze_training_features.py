import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# Fetch training data features joined with raw daily prices
df = pd.read_sql("""
    SELECT t.ticker, t.date, t.atr_14, t.high_low, r.open, r.high, r.low, r.close, r.volume
    FROM model_training_data t
    JOIN daily_raw_data r ON t.ticker = r.ticker AND t.date = r.date
    WHERE t.ticker = 'HPG' AND t.date BETWEEN '2023-01-01' AND '2023-02-01'
    ORDER BY t.date
""", conn)

print("--- HPG Sample Joined Data ---")
print(df.to_string())

# Let's try to calculate different versions of ATR:
# 1. Standard ATR: moving average of True Range
# 2. Normalized ATR: ATR / close, ATR / open, etc.
# Let's see if one of these matches the values in model_training_data
df_all_hpg = pd.read_sql("""
    SELECT date, open, high, low, close, volume
    FROM daily_raw_data
    WHERE ticker = 'HPG'
    ORDER BY date
""", conn)

close = df_all_hpg['close']
high = df_all_hpg['high']
low = df_all_hpg['low']

tr = pd.concat([
    high - low,
    (high - close.shift()).abs(),
    (low - close.shift()).abs()
], axis=1).max(axis=1)

# Normalised true range: TR / close
tr_norm = tr / close
# Normalised true range: TR / close.shift()
tr_norm_prev = tr / close.shift()

df_all_hpg['atr_14_raw'] = tr.rolling(14).mean()
df_all_hpg['atr_14_norm'] = tr_norm.rolling(14).mean()
df_all_hpg['atr_14_norm_prev'] = tr_norm_prev.rolling(14).mean()

# Merge back with model_training_data to check
df_compare = pd.read_sql("""
    SELECT date, atr_14
    FROM model_training_data
    WHERE ticker = 'HPG'
""", conn)
df_compare['date'] = pd.to_datetime(df_compare['date'])
df_all_hpg['date'] = pd.to_datetime(df_all_hpg['date'])

df_merged = pd.merge(df_compare, df_all_hpg, on='date', how='inner')
print("\n--- Compare Calculated ATR vs Database ATR ---")
print(df_merged[['date', 'atr_14', 'atr_14_raw', 'atr_14_norm', 'atr_14_norm_prev', 'close']].head(15).to_string())

conn.close()
