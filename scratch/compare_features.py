import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# Select features for a few tickers on 2026-05-29 and 2026-06-01
df_feats = pd.read_sql("""
    SELECT ticker, date, price_vs_sma50, volatility_20, volume_ratio_20, return_3d, return_5d, return_10d, return_20d, 
           sma_50_LogReturn, volume_LogReturn, PCA_Trend, PCA_Oscillators, PCA_MACD, PCA_ShortReturns, 
           atr_14, high_low, market_return, foreign_net
    FROM daily_normalized_data 
    WHERE ticker IN ('HPG', 'FPT', 'ACB') AND date IN ('2026-05-29', '2026-06-01')
    ORDER BY ticker, date
""", conn)

print("--- Feature comparison 2026-05-29 vs 2026-06-01 ---")
print(df_feats.to_string())

conn.close()
