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

# Fetch date ranges for VNINDEX and some tickers
query = """
SELECT ticker, MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count_rows 
FROM daily_raw_data 
WHERE ticker IN ('VNINDEX', 'ACB', 'BID', 'SSI')
GROUP BY ticker
"""
df = pd.read_sql(query, conn)
print(df)

conn.close()
