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

df_dr = pd.read_sql("SELECT DISTINCT ticker FROM daily_raw_data", conn)
tickers = df_dr['ticker'].tolist()
print("VNINDEX in raw data:", 'VNINDEX' in tickers)
print("VN30 in raw data:", 'VN30' in tickers)
print("VN30INDEX in raw data:", 'VN30INDEX' in tickers)
print("INDEX in raw data:", [t for t in tickers if 'INDEX' in t or 'VN' in t])

# Check if we can find VNINDEX in daily_raw_data or if it has another ticker name
# Or if it's stored in daily_raw_data but under a different name
# Or if we need to fetch it from the database_export file
conn.close()
