import mysql.connector
import sys
import os

# Check dates directly

def check_dates():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="170804",
        database="kltn_stock_db"
    )
    cursor = conn.cursor()
    
    tables = ['daily_raw_data', 'model_predictions', 'daily_normalized_data', 'model_training_data']
    for t in tables:
        try:
            cursor.execute(f"SELECT MAX(date) FROM {t}")
            max_date = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            count = cursor.fetchone()[0]
            print(f"Table: {t:25} | Max Date: {max_date} | Total Rows: {count:,}")
        except Exception as e:
            print(f"Table: {t:25} | Error: {e}")
            
    cursor.close()
    conn.close()

if __name__ == '__main__':
    check_dates()
