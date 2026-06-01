import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

df = pd.read_sql("""
    SELECT t.ticker, t.date, t.atr_14, r.high, r.low, r.close, 
           (r.high - r.low)/r.close as high_low_ratio,
           (r.high - r.low)/r.open as high_low_ratio_open,
           r.open
    FROM model_training_data t
    JOIN daily_raw_data r ON t.ticker = r.ticker AND t.date = r.date
    WHERE t.ticker = 'HPG'
    ORDER BY t.date
    LIMIT 20
""", conn)

print(df.to_string())
conn.close()
