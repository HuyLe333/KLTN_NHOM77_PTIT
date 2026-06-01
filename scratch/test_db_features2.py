import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

df = pd.read_sql("""
    SELECT date, prediction, COUNT(*), AVG(probability_up), AVG(probability_down) 
    FROM model_predictions 
    GROUP BY date, prediction
""", conn)
print("--- predictions by date and label ---")
print(df.to_string())

# Also let's check what date we have in model_predictions
df_dates = pd.read_sql("SELECT DISTINCT date FROM model_predictions ORDER BY date DESC", conn)
print("\n--- unique prediction dates ---")
print(df_dates.to_string())

# Check daily_normalized_data's latest record date
df_norm_dates = pd.read_sql("SELECT MAX(date) FROM daily_normalized_data", conn)
print("\n--- latest normalized data date ---")
print(df_norm_dates.to_string())

# Let's inspect raw pricing of some ticker on the latest dates
df_raw_check = pd.read_sql("SELECT ticker, date, open, high, low, close, volume FROM daily_raw_data WHERE ticker='HPG' ORDER BY date DESC LIMIT 5", conn)
print("\n--- raw pricing of HPG latest 5 sessions ---")
print(df_raw_check.to_string())

conn.close()
