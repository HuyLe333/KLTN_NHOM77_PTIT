import pandas as pd
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# Check daily_raw_data min/max date
cursor = conn.cursor()
cursor.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM daily_raw_data")
print("daily_raw_data stats:", cursor.fetchone())

# Check daily_raw_data number of tickers
cursor.execute("SELECT COUNT(DISTINCT ticker) FROM daily_raw_data")
print("daily_raw_data unique tickers:", cursor.fetchone()[0])

# Check model_training_data unique tickers
cursor.execute("SELECT COUNT(DISTINCT ticker) FROM model_training_data")
print("model_training_data unique tickers:", cursor.fetchone()[0])

# Let's load data2.xlsx dates
df_excel = pd.read_excel("data2.xlsx", usecols=['ticker', 'date'])
print("data2.xlsx date range:", df_excel['date'].min(), "to", df_excel['date'].max())
print("data2.xlsx total rows:", len(df_excel))

# Let's check how many rows match between data2.xlsx and daily_raw_data
df_excel['date'] = pd.to_datetime(df_excel['date']).dt.date
df_raw_dates = pd.read_sql("SELECT ticker, date FROM daily_raw_data", conn)
df_raw_dates['date'] = pd.to_datetime(df_raw_dates['date']).dt.date

df_merged = pd.merge(df_excel, df_raw_dates, on=['ticker', 'date'], how='inner')
print("Number of matching rows in join:", len(df_merged))

conn.close()
