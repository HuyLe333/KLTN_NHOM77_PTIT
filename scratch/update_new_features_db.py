import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
import FiinQuantX as fq
import time

DB_USER = "root"
DB_PASS = "170804"
DB_HOST = "localhost"
DB_NAME = "kltn_stock_db"

def main():
    print("=" * 60)
    print("  FETCHING & UPDATING HISTORICAL BU, SD, FS, FB")
    print("=" * 60)

    # 1. Connect to MySQL and retrieve tickers
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    with engine.connect() as conn:
        res = conn.execute(text("SELECT DISTINCT ticker FROM model_training_data"))
        tickers = [row[0] for row in res.fetchall()]
        
    print(f"Loaded {len(tickers)} tickers from model_training_data.")
    if not tickers:
        print("No tickers found in database!")
        return

    # 2. Login to FiinQuant
    print("\nLogging into FiinQuant API...")
    try:
        session = fq.FiinSession(
            username="anh.phamthitu@fiingroup.vn",
            password="Anhkiet15"
        ).login()
        print("Login Successful.")
    except Exception as e:
        print(f"FiinQuant login failed: {e}")
        return

    # 3. Fetch data from 2021-01-01 to 2026-06-01
    print("\nFetching historical 'bu, sd, fs, fb' data...")
    t0 = time.time()
    try:
        # Fetch in chunks of tickers to avoid timeout
        chunk_size = 25
        all_dfs = []
        for i in range(0, len(tickers), chunk_size):
            ticker_chunk = tickers[i:i+chunk_size]
            print(f"  Fetching chunk {i//chunk_size + 1}/{len(tickers)//chunk_size + 1} for {len(ticker_chunk)} tickers...")
            event = session.Fetch_Trading_Data(
                realtime=False,
                tickers=ticker_chunk,
                fields=["bu", "sd", "fs", "fb"],
                adjusted=True,
                from_date="2021-01-01",
                to_date="2026-06-01",
                by="1d"
            )
            df_chunk = event.get_data()
            if df_chunk is not None and not df_chunk.empty:
                all_dfs.append(df_chunk)
            time.sleep(1) # Be nice to the API

        if not all_dfs:
            print("No data fetched from API!")
            return

        df_api = pd.concat(all_dfs, ignore_index=True)
        print(f"Successfully fetched {len(df_api)} rows in {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"API Fetch failed: {e}")
        return

    # 4. Clean fetched data
    print("\nCleaning data...")
    df_api = df_api.dropna(subset=['ticker', 'timestamp'])
    df_api['date'] = pd.to_datetime(df_api['timestamp']).dt.date
    df_api['bu'] = df_api['bu'].fillna(0.0)
    df_api['sd'] = df_api['sd'].fillna(0.0)
    df_api['fs'] = df_api['fs'].fillna(0.0)
    df_api['fb'] = df_api['fb'].fillna(0.0)
    
    df_api = df_api[['ticker', 'date', 'bu', 'sd', 'fs', 'fb']]
    df_api = df_api.drop_duplicates(subset=['ticker', 'date'])
    
    print(f"Total cleaned rows: {len(df_api)}")
    print(f"Non-zero stats:")
    print(f"  bu non-zero: {(df_api['bu'] != 0).sum()} ({(df_api['bu'] != 0).sum()/len(df_api)*100:.2f}%)")
    print(f"  sd non-zero: {(df_api['sd'] != 0).sum()} ({(df_api['sd'] != 0).sum()/len(df_api)*100:.2f}%)")
    print(f"  fs non-zero: {(df_api['fs'] != 0).sum()} ({(df_api['fs'] != 0).sum()/len(df_api)*100:.2f}%)")
    print(f"  fb non-zero: {(df_api['fb'] != 0).sum()} ({(df_api['fb'] != 0).sum()/len(df_api)*100:.2f}%)")

    # 5. Save to temp table and update model_training_data
    temp_table = "temp_new_features"
    print(f"\nUploading to temporary table '{temp_table}'...")
    df_api.to_sql(
        name=temp_table,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=10000
    )

    print("Merging new features into 'model_training_data'...")
    update_query = f"""
    UPDATE model_training_data m
    JOIN {temp_table} t ON m.ticker = t.ticker AND m.date = t.date
    SET m.bu = t.bu,
        m.sd = t.sd,
        m.fs = t.fs,
        m.fb = t.fb;
    """
    
    with engine.connect() as conn:
        res = conn.execute(text(update_query))
        conn.commit()
        print(f"Updated records in model_training_data.")
        
        # Clean up temp table
        conn.execute(text(f"DROP TABLE {temp_table}"))
        conn.commit()
        print("Dropped temporary table.")

    print("\nVerifying database values for new features...")
    with engine.connect() as conn:
        db_res = conn.execute(text("""
            SELECT COUNT(*),
                   SUM(CASE WHEN bu != 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN sd != 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN fs != 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN fb != 0 THEN 1 ELSE 0 END)
            FROM model_training_data
        """)).fetchone()
        print(f"Verification result: Total Rows={db_res[0]}")
        print(f"  Non-zero bu={db_res[1]}")
        print(f"  Non-zero sd={db_res[2]}")
        print(f"  Non-zero fs={db_res[3]}")
        print(f"  Non-zero fb={db_res[4]}")

    print("=" * 60)
    print("  HISTORICAL UPDATE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
