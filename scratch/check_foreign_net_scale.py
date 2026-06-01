import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# Check model_training_data foreign_net stats
print("--- model_training_data foreign_net stats ---")
df_train = pd.read_sql("SELECT COUNT(*), MIN(foreign_net), MAX(foreign_net), AVG(foreign_net), std(foreign_net) FROM model_training_data WHERE foreign_net != 0", conn)
print(df_train)

# Check daily_normalized_data foreign_net stats
print("\n--- daily_normalized_data foreign_net stats ---")
df_norm = pd.read_sql("SELECT date, COUNT(*), MIN(foreign_net), MAX(foreign_net), AVG(foreign_net), std(foreign_net) FROM daily_normalized_data GROUP BY date", conn)
print(df_norm)

# Check some non-zero foreign_net values in model_training_data
print("\n--- model_training_data foreign_net samples ---")
df_train_samples = pd.read_sql("SELECT ticker, date, foreign_net FROM model_training_data WHERE foreign_net != 0 LIMIT 15", conn)
print(df_train_samples)

conn.close()
