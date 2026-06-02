import mysql.connector
import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine, text

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    print("=" * 60)
    print("  UPDATING DATABASE SCHEMA & HISTORICAL RRG DATA")
    print("=" * 60)

    # 1. Connect to MySQL and Add Columns if they don't exist
    print("\n[1/4] Connecting to MySQL and altering tables...")
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        # Alter model_training_data
        try:
            cursor.execute("ALTER TABLE model_training_data ADD COLUMN rs DOUBLE AFTER fb")
            print("    OK Added column 'rs' to model_training_data.")
        except mysql.connector.Error as err:
            if err.errno == 1060: # Column already exists
                print("    Column 'rs' already exists in model_training_data.")
            else:
                raise err
                
        try:
            cursor.execute("ALTER TABLE model_training_data ADD COLUMN rm DOUBLE AFTER rs")
            print("    OK Added column 'rm' to model_training_data.")
        except mysql.connector.Error as err:
            if err.errno == 1060:
                print("    Column 'rm' already exists in model_training_data.")
            else:
                raise err

        # Alter daily_normalized_data
        try:
            cursor.execute("ALTER TABLE daily_normalized_data ADD COLUMN rs DOUBLE AFTER fb")
            print("    OK Added column 'rs' to daily_normalized_data.")
        except mysql.connector.Error as err:
            if err.errno == 1060:
                print("    Column 'rs' already exists in daily_normalized_data.")
            else:
                raise err
                
        try:
            cursor.execute("ALTER TABLE daily_normalized_data ADD COLUMN rm DOUBLE AFTER rs")
            print("    OK Added column 'rm' to daily_normalized_data.")
        except mysql.connector.Error as err:
            if err.errno == 1060:
                print("    Column 'rm' already exists in daily_normalized_data.")
            else:
                raise err
                
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"    ❌ Altering tables failed: {e}")
        return

    # 2. Fetch daily_raw_data and compute RRG
    print("\n[2/4] Fetching raw prices and calculating daily RRG...")
    t0 = time.time()
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    df_raw = pd.read_sql("SELECT ticker, date, close FROM daily_raw_data", engine)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.drop_duplicates(subset=['ticker', 'date'])
    
    df_bench = df_raw[df_raw['ticker'] == 'VNINDEX'].sort_values('date').reset_index(drop=True)
    df_bench = df_bench.rename(columns={'close': 'close_bench'})
    
    rrg_dfs = []
    unique_tickers = [t for t in df_raw['ticker'].unique() if t != 'VNINDEX']
    
    for ticker in unique_tickers:
        df_ticker = df_raw[df_raw['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df_ticker) < 35:
            continue
            
        merged = pd.merge(df_ticker[['date', 'close']], df_bench[['date', 'close_bench']], on='date', how='inner')
        merged = merged.sort_values('date').reset_index(drop=True)
        
        if len(merged) < 35:
            continue
            
        close_ratio = merged['close'] / merged['close_bench']
        rs = 100 * close_ratio.rolling(window=12).mean() / close_ratio.rolling(window=26).mean()
        rm = 100 * rs / rs.rolling(window=9).mean()
        
        rrg_df = pd.DataFrame({
            'ticker': ticker,
            'date': merged['date'],
            'rs': rs,
            'rm': rm
        })
        rrg_dfs.append(rrg_df)
        
    df_rrg = pd.concat(rrg_dfs, ignore_index=True).dropna(subset=['rs', 'rm'])
    print(f"    OK Calculated historical RRG in {time.time() - t0:.2f} seconds. Rows: {len(df_rrg):,}")

    # 3. Upload to temp table
    print("\n[3/4] Uploading calculations to temp table...")
    t1 = time.time()
    df_rrg.to_sql('temp_rrg_updates', con=engine, if_exists='replace', index=False)
    print(f"    OK Temp table uploaded in {time.time() - t1:.2f} seconds.")

    # 4. Perform fast JOIN UPDATE in MySQL
    print("\n[4/4] Updating main tables via SQL JOIN...")
    t2 = time.time()
    with engine.connect() as conn:
        # Update model_training_data
        print("    Updating model_training_data...")
        conn.execute(text("""
            UPDATE model_training_data m
            JOIN temp_rrg_updates t ON m.ticker = t.ticker AND m.date = t.date
            SET m.rs = t.rs, m.rm = t.rm
        """))
        conn.commit()
        
        # Update daily_normalized_data
        print("    Updating daily_normalized_data...")
        conn.execute(text("""
            UPDATE daily_normalized_data m
            JOIN temp_rrg_updates t ON m.ticker = t.ticker AND m.date = t.date
            SET m.rs = t.rs, m.rm = t.rm
        """))
        conn.commit()
        
        # Drop temp table
        print("    Cleaning up temp table...")
        conn.execute(text("DROP TABLE temp_rrg_updates"))
        conn.commit()
        
    print(f"    OK Main tables updated successfully in {time.time() - t2:.2f} seconds.")
    print("=" * 60)
    print("  DATABASE RRG UPDATE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
