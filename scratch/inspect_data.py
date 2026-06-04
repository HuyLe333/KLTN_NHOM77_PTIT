import mysql.connector
import pandas as pd

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME
)

cursor = conn.cursor()

print("--- Tickers in model_training_data ---")
df_mt = pd.read_sql("SELECT DISTINCT ticker FROM model_training_data", conn)
print(df_mt['ticker'].tolist()[:20])
print(f"Total tickers: {len(df_mt)}")

print("\n--- Tickers in daily_raw_data ---")
df_dr = pd.read_sql("SELECT DISTINCT ticker FROM daily_raw_data", conn)
print(df_dr['ticker'].tolist()[:20])
print(f"Total tickers: {len(df_dr)}")

conn.close()
