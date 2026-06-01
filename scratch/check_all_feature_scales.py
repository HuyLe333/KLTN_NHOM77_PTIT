import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

feature_cols = [
    'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
    'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'sma_50_LogReturn', 'volume_LogReturn',
    'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
    'atr_14', 'high_low', 'market_return', 'foreign_net'
]

# Get stats from model_training_data
df_train = pd.read_sql(f"SELECT {', '.join(feature_cols)} FROM model_training_data", conn)
train_desc = df_train.describe().loc[['min', 'max', 'mean', 'std']]

# Get stats from daily_normalized_data on 2026-06-01
df_norm = pd.read_sql(f"SELECT {', '.join(feature_cols)} FROM daily_normalized_data WHERE date='2026-06-01'", conn)
norm_desc = df_norm.describe().loc[['min', 'max', 'mean', 'std']]

print("=== FEATURE SCALE COMPARISON ===")
for col in feature_cols:
    print(f"\nFeature: {col}")
    print(f"  Training   - min: {train_desc.loc['min', col]:.4f}, max: {train_desc.loc['max', col]:.4f}, mean: {train_desc.loc['mean', col]:.4f}, std: {train_desc.loc['std', col]:.4f}")
    print(f"  2026-06-01 - min: {norm_desc.loc['min', col]:.4f}, max: {norm_desc.loc['max', col]:.4f}, mean: {norm_desc.loc['mean', col]:.4f}, std: {norm_desc.loc['std', col]:.4f}")

conn.close()
