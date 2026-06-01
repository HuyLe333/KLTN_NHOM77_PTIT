import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# Check model_training_data atr_14 stats
print("--- model_training_data atr_14 stats ---")
df_train = pd.read_sql("SELECT ticker, COUNT(*), MIN(atr_14), MAX(atr_14), AVG(atr_14), std(atr_14) FROM model_training_data WHERE ticker IN ('HPG', 'FPT', 'ACB') GROUP BY ticker", conn)
print(df_train)

# Check daily_normalized_data atr_14 stats on 2026-06-01
print("\n--- daily_normalized_data atr_14 stats on 2026-06-01 ---")
df_norm = pd.read_sql("SELECT ticker, date, atr_14 FROM daily_normalized_data WHERE date='2026-06-01' AND ticker IN ('HPG', 'FPT', 'ACB')", conn)
print(df_norm)

conn.close()
