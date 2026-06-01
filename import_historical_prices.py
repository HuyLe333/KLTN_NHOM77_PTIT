import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine, text

EXCEL_PATH = r"D:\Khóa luận tốt nghiệp\phân tích data\data v1\database_export_20260518_1159.xlsx"
DB_USER = "root"
DB_PASS = "170804"
DB_HOST = "localhost"
DB_NAME = "kltn_stock_db"

def main():
    print("=" * 60)
    print("  IMPORTING ACCURATE HISTORICAL PRICES FROM EXCEL")
    print("=" * 60)
    
    t0 = time.time()
    print("Loading sheet 'daily_prices' from the Excel export file...")
    # Load required columns to save memory
    df = pd.read_excel(
        EXCEL_PATH, 
        sheet_name='daily_prices',
        usecols=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    )
    print(f"Loaded {len(df)} rows in {time.time() - t0:.2f} seconds.")
    
    # Cleaning data
    print("Cleaning and processing data...")
    df = df.dropna(subset=['ticker', 'date'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Convert prices to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with null close prices
    df = df.dropna(subset=['close'])
    print(f"Cleaned rows to insert: {len(df)}")
    
    # SQLAlchemy connection
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    # Step 1: Upload to temporary table
    temp_table_name = "temp_import_prices"
    print(f"Uploading to temporary table '{temp_table_name}'...")
    t_upload = time.time()
    df.to_sql(
        name=temp_table_name,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=10000
    )
    print(f"Uploaded to temp table in {time.time() - t_upload:.2f} seconds.")
    
    # Step 2: Merge into daily_raw_data using ON DUPLICATE KEY UPDATE
    print("Merging data into 'daily_raw_data' table...")
    t_merge = time.time()
    
    merge_query = """
    INSERT INTO daily_raw_data (ticker, date, open, high, low, close, volume)
    SELECT ticker, date, open, high, low, close, volume FROM temp_import_prices
    ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        volume = VALUES(volume);
    """
    
    with engine.connect() as conn:
        # Enable transactions/auto-commit
        conn.execute(text(merge_query))
        conn.commit()
        
        # Clean up temp table
        print(f"Dropping temporary table '{temp_table_name}'...")
        conn.execute(text(f"DROP TABLE {temp_table_name}"))
        conn.commit()
        
    print(f"Merged successfully in {time.time() - t_merge:.2f} seconds.")
    print("=" * 60)
    print("  IMPORT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
