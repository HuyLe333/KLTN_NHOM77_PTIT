"""
Daily Crawler & Prediction Pipeline
1. Connects to MySQL kltn_stock_db.
2. Queries the list of tickers to track.
3. Authenticates with FiinQuant API.
4. Fetches the last 90 calendar days of daily price data for all tickers and VNINDEX.
5. Ingests raw pricing into daily_raw_data.
6. Computes standard technical indicators.
7. Reconstructs PCA features using the trained pca_predictor.pkl.
8. Persists features to daily_normalized_data.
9. Runs XGBoost model (xgb_model.json) on the latest session features to generate forecasts.
10. Saves predictions to model_predictions.
"""
import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine, text
import FiinQuantX as fq
import pickle
import xgboost as xgb
from datetime import datetime, timedelta
import os
import sys
import io

# Setup UTF-8 encoding for Windows console logs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    print("=" * 60)
    print("  RUNNING DAILY CRAWLER & FORECAST PIPELINE")
    print("=" * 60)

    # 1. Connect to MySQL and retrieve tickers
    print("\n[1/7] Querying ticker list from MySQL database...")
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM model_training_data")
        tickers = [row[0] for row in cursor.fetchall()]
        print(f"    OK Found {len(tickers)} tickers to process.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"    ❌ Database connection failed: {e}")
        return

    if not tickers:
        print("    ❌ No tickers found in model_training_data.")
        return

    # 2. Login to FiinQuant
    print("\n[2/7] Authenticating with FiinQuant API...")
    try:
        session = fq.FiinSession(
            username="anh.phamthitu@fiingroup.vn",
            password="Anhkiet15"
        ).login()
        print("    OK Login Successful.")
    except Exception as e:
        print(f"    ❌ FiinQuant authentication failed: {e}")
        return

    # 3. Calculate Date Range (90 calendar days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    from_date_str = start_date.strftime("%Y-%m-%d")
    to_date_str = end_date.strftime("%Y-%m-%d")
    print(f"    Fetching data from {from_date_str} to {to_date_str}...")

    # 4. Fetch Ticker Data and Index Data
    print("\n[3/7] Fetching trading data from FiinQuant API...")
    try:
        # Fetch tickers prices
        print("    Fetching price data for 93 tickers...")
        event = session.Fetch_Trading_Data(
            realtime=False,
            tickers=tickers,
            fields=["open", "high", "low", "close", "volume", "fn", "bu", "sd", "fs", "fb"],
            adjusted=True,
            from_date=from_date_str,
            by="1d"
        )
        df_tickers_raw = event.get_data()
        if not df_tickers_raw.empty:
            if 'fn' in df_tickers_raw.columns:
                df_tickers_raw.rename(columns={'fn': 'foreign_net'}, inplace=True)
            else:
                df_tickers_raw['foreign_net'] = 0.0
            
            for col in ['bu', 'sd', 'fs', 'fb']:
                if col not in df_tickers_raw.columns:
                    df_tickers_raw[col] = 0.0
                else:
                    df_tickers_raw[col] = df_tickers_raw[col].fillna(0.0)
        print(f"    OK Tickers data shape: {df_tickers_raw.shape}")
        
        # Fetch VNINDEX price
        print("    Fetching index data for VNINDEX...")
        event_index = session.Fetch_Trading_Data(
            realtime=False,
            tickers=["VNINDEX"],
            fields=["open", "high", "low", "close", "volume"],
            adjusted=True,
            from_date=from_date_str,
            by="1d"
        )
        df_index_raw = event_index.get_data()
        print(f"    OK VNINDEX data shape: {df_index_raw.shape}")
    except Exception as e:
        print(f"    ❌ API Fetch failed: {e}")
        return

    if df_tickers_raw.empty or df_index_raw.empty:
        print("    ❌ Fetched data is empty. Cannot continue.")
        return

    # Clean and parse dates
    df_tickers_raw['date'] = pd.to_datetime(df_tickers_raw['timestamp']).dt.date
    df_index_raw['date'] = pd.to_datetime(df_index_raw['timestamp']).dt.date
    
    # Sort and clean index
    df_index = df_index_raw[['date', 'close']].sort_values('date').drop_duplicates('date').reset_index(drop=True)
    df_index.rename(columns={'close': 'index_close'}, inplace=True)

    # 5. Load PCA Mapper and XGBoost Model
    print("\n[4/7] Loading mapping and modeling resources...")
    if not os.path.exists("pca_predictor.pkl"):
        print("    ❌ 'pca_predictor.pkl' not found. Run fit_pca_mapper.py first.")
        return
    if not os.path.exists("xgb_model.json"):
        print("    ❌ 'xgb_model.json' not found.")
        return
        
    with open("pca_predictor.pkl", "rb") as f:
        pca_resource = pickle.load(f)
    pca_mapper = pca_resource['model']
    
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    print("    OK Models loaded successfully.")

    # 6. Process each Ticker
    print("\n[5/7] Processing feature engineering and predictions...")
    
    # Establish database engine for insertion
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    raw_insert_count = 0
    norm_insert_count = 0
    pred_insert_count = 0
    
    # We will collect rows for bulk database insert
    raw_rows = []
    norm_rows = []
    pred_rows = []

    for ticker in tickers:
        df_t = df_tickers_raw[df_tickers_raw['ticker'] == ticker].copy()
        if df_t.empty:
            continue
            
        df_t = df_t.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        
        # Merge with VNINDEX to align dates
        df_merged = pd.merge(df_t, df_index, on='date', how='left')
        # ffill VNINDEX close if any missing
        df_merged['index_close'] = df_merged['index_close'].ffill().bfill()
        
        if len(df_merged) < 50:
            # We need at least 50 trading days to compute the rolling SMA 50
            continue
            
        # Add raw rows to save
        for _, row in df_merged.iterrows():
            raw_rows.append({
                'ticker': ticker,
                'date': row['date'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
            
        # Calculate features
        close = df_merged['close'].astype(float)
        high = df_merged['high'].astype(float)
        low = df_merged['low'].astype(float)
        volume = df_merged['volume'].astype(float)
        index_close = df_merged['index_close'].astype(float)
        
        df_merged['close_LogReturn'] = np.log(close / close.shift(1))
        
        sma50 = close.rolling(50).mean()
        df_merged['price_vs_sma50'] = (close / sma50) - 1
        
        df_merged['volatility_20'] = df_merged['close_LogReturn'].rolling(20).std() * np.sqrt(252)
        df_merged['volume_ratio_20'] = volume / volume.rolling(20).mean()
        
        df_merged['return_3d'] = close.pct_change(3)
        df_merged['return_5d'] = close.pct_change(5)
        df_merged['return_10d'] = close.pct_change(10)
        df_merged['return_20d'] = close.pct_change(20)
        
        df_merged['sma_50_LogReturn'] = np.log(sma50 / sma50.shift(1))
        df_merged['volume_LogReturn'] = np.log(volume / volume.shift(1))
        
        # ATR 14 (Normalized: ATR 14 / close)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df_merged['atr_14'] = tr.rolling(14).mean() / close
        
        # high_low: (high - low) / close
        df_merged['high_low'] = (high - low) / close
        df_merged['market_return'] = index_close.pct_change(1)
        
        # Clean infinite and null values from feature calculations
        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        feature_cols_base = [
            'price_vs_sma50', 'volatility_20', 'volume_ratio_20', 
            'return_3d', 'return_5d', 'return_10d', 'return_20d', 
            'sma_50_LogReturn', 'volume_LogReturn', 'atr_14', 
            'high_low', 'market_return'
        ]
        
        df_features = df_merged.dropna(subset=['close_LogReturn'] + feature_cols_base).copy()
        if df_features.empty:
            continue
            
        # Reconstruct PCA components
        X_base = df_features[feature_cols_base]
        pca_preds = pca_mapper.predict(X_base)
        
        df_features['PCA_Trend'] = pca_preds[:, 0]
        df_features['PCA_Oscillators'] = pca_preds[:, 1]
        df_features['PCA_MACD'] = pca_preds[:, 2]
        df_features['PCA_ShortReturns'] = pca_preds[:, 3]
        
        # Add to normalized rows
        for _, row in df_features.iterrows():
            norm_rows.append({
                'ticker': ticker,
                'date': row['date'],
                'close_LogReturn': float(row['close_LogReturn']),
                'price_vs_sma50': float(row['price_vs_sma50']),
                'volatility_20': float(row['volatility_20']),
                'volume_ratio_20': float(row['volume_ratio_20']),
                'return_3d': float(row['return_3d']),
                'return_5d': float(row['return_5d']),
                'return_10d': float(row['return_10d']),
                'return_20d': float(row['return_20d']),
                'sma_50_LogReturn': float(row['sma_50_LogReturn']),
                'volume_LogReturn': float(row['volume_LogReturn']),
                'PCA_Trend': float(row['PCA_Trend']),
                'PCA_Oscillators': float(row['PCA_Oscillators']),
                'PCA_MACD': float(row['PCA_MACD']),
                'PCA_ShortReturns': float(row['PCA_ShortReturns']),
                'atr_14': float(row['atr_14']),
                'high_low': float(row['high_low']),
                'market_return': float(row['market_return']),
                'foreign_net': float(row.get('foreign_net', 0.0)),
                'bu': float(row.get('bu', 0.0)),
                'sd': float(row.get('sd', 0.0)),
                'fs': float(row.get('fs', 0.0)),
                'fb': float(row.get('fb', 0.0))
            })
            
        # Get latest day row for prediction
        latest_row = df_features.iloc[-1]
        latest_date = latest_row['date']
        
        # Features for model input (21 columns)
        feature_cols_model = [
            'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
            'return_3d', 'return_5d', 'return_10d', 'return_20d',
            'sma_50_LogReturn', 'volume_LogReturn',
            'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
            'atr_14', 'high_low', 'market_return', 'foreign_net',
            'bu', 'sd', 'fs', 'fb'
        ]
        
        X_pred = np.array([[float(latest_row[f]) for f in feature_cols_model]])
        
        pred_label = int(model.predict(X_pred)[0])
        proba = model.predict_proba(X_pred)[0].tolist()
        proba_up = float(proba[1])
        proba_down = float(proba[0])
        confidence = abs(proba_up - proba_down)
        
        # Save prediction
        pred_rows.append({
            'ticker': ticker,
            'date': latest_date,
            'prediction': pred_label,
            'probability_up': proba_up,
            'probability_down': proba_down,
            'confidence': confidence,
            'predict_date': datetime.now().date()
        })

    # 7. Persist to Database using INSERT IGNORE / REPLACE
    print("\n[6/7] Ingesting parsed daily data to MySQL database...")
    
    # Ingest Raw prices
    if raw_rows:
        df_raw = pd.DataFrame(raw_rows)
        # Use mysql connection to execute replace / insert ignore
        with engine.connect() as conn_sql:
            # We can write a custom REPLACE/INSERT IGNORE function, or simply use df.to_sql with a temporary table
            # But the easiest way for key constraints in pandas is to dump into a temp table, then insert ignore
            print("    Ingesting raw pricing...")
            df_raw.to_sql('temp_raw', con=engine, if_exists='replace', index=False)
            conn_sql.execute(text("INSERT IGNORE INTO daily_raw_data SELECT * FROM temp_raw"))
            conn_sql.execute(text("DROP TABLE temp_raw"))
            conn_sql.commit()
            raw_insert_count = len(df_raw)
            print(f"      - daily_raw_data: {raw_insert_count} records processed.")

    # Ingest Normalized features
    if norm_rows:
        df_norm = pd.DataFrame(norm_rows)
        with engine.connect() as conn_sql:
            print("    Ingesting normalized features...")
            df_norm.to_sql('temp_norm', con=engine, if_exists='replace', index=False)
            conn_sql.execute(text("INSERT IGNORE INTO daily_normalized_data SELECT * FROM temp_norm"))
            conn_sql.execute(text("DROP TABLE temp_norm"))
            conn_sql.commit()
            norm_insert_count = len(df_norm)
            print(f"      - daily_normalized_data: {norm_insert_count} records processed.")

    # Ingest predictions
    if pred_rows:
        df_pred = pd.DataFrame(pred_rows)
        with engine.connect() as conn_sql:
            print("    Ingesting XGBoost model predictions...")
            df_pred.to_sql('temp_pred', con=engine, if_exists='replace', index=False)
            conn_sql.execute(text("REPLACE INTO model_predictions SELECT * FROM temp_pred"))
            conn_sql.execute(text("DROP TABLE temp_pred"))
            conn_sql.commit()
            pred_insert_count = len(df_pred)
            print(f"      - model_predictions: {pred_insert_count} predictions saved.")

    print("\n[7/7] Verifying latest prediction results...")
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, date, prediction, confidence FROM model_predictions ORDER BY confidence DESC LIMIT 10")
        rows = cursor.fetchall()
        print("\nTop 10 Predictions with Highest Confidence:")
        print(f"  {'Ticker':8s} | {'Target Date':11s} | {'Prediction':11s} | {'Confidence':10s}")
        print("  " + "-" * 50)
        for r in rows:
            pred_text = "Tang/Mua (1)" if r[2] == 1 else "Giam/Ban (0)"
            print(f"  {r[0]:8s} | {str(r[1]):11s} | {pred_text:11s} | {r[3]:.4f}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"    Verification failed: {e}")

    print("=" * 60)
    print("  DAILY CRAWLER & FORECAST PIPELINE RUN COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
