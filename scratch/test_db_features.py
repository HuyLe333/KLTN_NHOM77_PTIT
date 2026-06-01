import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# Check some records from model_training_data
df = pd.read_sql("SELECT ticker, date, high_low, market_return, close_LogReturn FROM model_training_data LIMIT 10", conn)
print("--- model_training_data sample ---")
print(df.to_string())

# Check some records from model_predictions
df_pred = pd.read_sql("SELECT ticker, date, prediction, probability_up, probability_down, confidence FROM model_predictions LIMIT 15", conn)
print("\n--- model_predictions sample ---")
print(df_pred.to_string())

# Check how many positive vs negative predictions there are in model_predictions
df_summary = pd.read_sql("SELECT prediction, COUNT(*), AVG(probability_up), AVG(probability_down) FROM model_predictions GROUP BY prediction", conn)
print("\n--- model_predictions summary ---")
print(df_summary.to_string())

conn.close()
