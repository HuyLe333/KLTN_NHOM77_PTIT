"""
Database Setup Script
1. Connects to local MySQL.
2. Creates the database kltn_stock_db.
3. Creates the 4 required tables.
4. Loads, processes, and imports data2.xlsx into model_training_data.
"""
import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import time
import os

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    print("=" * 60)
    print("  INITIALIZING MYSQL DATABASE SCHEMA")
    print("=" * 60)

    # 1. Connect to MySQL to create the database if not exists
    print("\n[1/5] Connecting to MySQL server...")
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS
        )
        cursor = conn.cursor()
        print("    OK Connected.")
    except Exception as e:
        print(f"    ❌ Connection failed: {e}")
        print("    Please ensure MySQL server is running and credentials are correct.")
        return

    print(f"\n[2/5] Creating database '{DB_NAME}' if not exists...")
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"    OK Database '{DB_NAME}' is ready.")

    # 2. Establish connection to the new database
    db_conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    db_cursor = db_conn.cursor()

    # 3. Create Tables
    print("\n[3/5] Creating tables...")
    
    # Drop tables to apply schema changes
    db_cursor.execute("DROP TABLE IF EXISTS model_training_data;")
    db_cursor.execute("DROP TABLE IF EXISTS daily_normalized_data;")
    
    # Table 1: model_training_data
    db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_training_data (
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        close_LogReturn DOUBLE,
        price_vs_sma50 DOUBLE,
        volatility_20 DOUBLE,
        volume_ratio_20 DOUBLE,
        return_3d DOUBLE,
        return_5d DOUBLE,
        return_10d DOUBLE,
        return_20d DOUBLE,
        sma_50_LogReturn DOUBLE,
        volume_LogReturn DOUBLE,
        PCA_Trend DOUBLE,
        PCA_Oscillators DOUBLE,
        PCA_MACD DOUBLE,
        PCA_ShortReturns DOUBLE,
        atr_14 DOUBLE,
        high_low DOUBLE,
        market_return DOUBLE,
        foreign_net DOUBLE,
        target INT,
        PRIMARY KEY (ticker, date)
    ) ENGINE=InnoDB;
    """)

    # Table 2: daily_raw_data
    db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_raw_data (
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        PRIMARY KEY (ticker, date)
    ) ENGINE=InnoDB;
    """)

    # Table 3: daily_normalized_data
    db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_normalized_data (
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        close_LogReturn DOUBLE,
        price_vs_sma50 DOUBLE,
        volatility_20 DOUBLE,
        volume_ratio_20 DOUBLE,
        return_3d DOUBLE,
        return_5d DOUBLE,
        return_10d DOUBLE,
        return_20d DOUBLE,
        sma_50_LogReturn DOUBLE,
        volume_LogReturn DOUBLE,
        PCA_Trend DOUBLE,
        PCA_Oscillators DOUBLE,
        PCA_MACD DOUBLE,
        PCA_ShortReturns DOUBLE,
        atr_14 DOUBLE,
        high_low DOUBLE,
        market_return DOUBLE,
        foreign_net DOUBLE,
        PRIMARY KEY (ticker, date)
    ) ENGINE=InnoDB;
    """)

    # Table 4: model_predictions
    db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_predictions (
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        prediction INT NOT NULL,
        probability_up DOUBLE,
        probability_down DOUBLE,
        confidence DOUBLE,
        predict_date DATE NOT NULL,
        PRIMARY KEY (ticker, date)
    ) ENGINE=InnoDB;
    """)

    db_conn.commit()
    db_cursor.close()
    db_conn.close()
    print("    OK All tables created.")

    # 4. Load data2.xlsx and process
    print("\n[4/5] Loading and processing training dataset 'data2.xlsx'...")
    if not os.path.exists("data2.xlsx"):
        print("    ❌ 'data2.xlsx' not found. Cannot populate model_training_data.")
        return

    t0 = time.time()
    df_raw = pd.read_excel("data2.xlsx")
    print(f"    OK Loaded data2.xlsx in {time.time() - t0:.2f} seconds. Rows: {df_raw.shape[0]}")

    print("    Loading 'foreign_net' from the Excel export file...")
    excel_path = r"D:\Khóa luận tốt nghiệp\phân tích data\data v1\database_export_20260518_1159.xlsx"
    df_indicators = pd.read_excel(
        excel_path,
        sheet_name='technical_indicators',
        usecols=['ticker', 'date', 'foreign_net']
    )
    df_indicators['date'] = pd.to_datetime(df_indicators['date'])
    df_indicators = df_indicators.drop_duplicates(subset=['ticker', 'date'])

    # Process and build target target
    df = df_raw.copy()
    if 'ticker' not in df.columns or 'date' not in df.columns:
        print("    ❌ 'ticker' and 'date' columns must exist in data2.xlsx.")
        return

    # Clean date
    df['date'] = pd.to_datetime(df['date'])

    # Merge foreign_net
    print("    Merging 'foreign_net' into training data...")
    df = pd.merge(df, df_indicators, on=['ticker', 'date'], how='left')
    df['foreign_net'] = df['foreign_net'].fillna(0.0)

    # Compute target labels (T+5 return prediction target)
    print("    Calculating target labels...")
    df['close_LogReturn_5d'] = (
        df.groupby('ticker')['close_LogReturn'].shift(-1) +
        df.groupby('ticker')['close_LogReturn'].shift(-2) +
        df.groupby('ticker')['close_LogReturn'].shift(-3) +
        df.groupby('ticker')['close_LogReturn'].shift(-4) +
        df.groupby('ticker')['close_LogReturn'].shift(-5)
    )
    df['target'] = (df['close_LogReturn_5d'] > 0).astype(int)
    
    # Drop rows with null values in either training columns or target
    train_cols = [
        'ticker', 'date', 'close_LogReturn', 'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
        'return_3d', 'return_5d', 'return_10d', 'return_20d', 'sma_50_LogReturn', 'volume_LogReturn',
        'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns', 'atr_14', 'high_low', 'market_return', 'foreign_net', 'target'
    ]
    df_db = df[train_cols].dropna()
    print(f"    OK Cleaned training rows for DB ingest: {df_db.shape[0]}")

    # 5. Ingest into MySQL
    print("\n[5/5] Ingesting training data into MySQL 'model_training_data'...")
    t1 = time.time()
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    # Truncate if exists to avoid duplicates during setup rerun
    db_conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    db_cursor = db_conn.cursor()
    db_cursor.execute("TRUNCATE TABLE model_training_data;")
    db_conn.commit()
    db_cursor.close()
    db_conn.close()
    
    df_db.to_sql(
        name='model_training_data',
        con=engine,
        if_exists='append',
        index=False,
        chunksize=5000
    )
    print(f"    OK Ingest completed in {time.time() - t1:.2f} seconds.")
    print("=" * 60)
    print("  DATABASE SCHEMA SET UP AND SEEDED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
