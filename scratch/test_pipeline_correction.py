import pandas as pd
import numpy as np
import mysql.connector
import xgboost as xgb
import pickle
import os

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    print("Loading models...")
    with open("pca_predictor.pkl", "rb") as f:
        pca_resource = pickle.load(f)
    pca_mapper = pca_resource['model']
    
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    
    # Connect to MySQL and retrieve tickers
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM model_training_data")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    # Let's query raw daily prices from daily_raw_data for the last 90 days
    df_raw = pd.read_sql("SELECT ticker, date, open, high, low, close, volume FROM daily_raw_data", conn)
    
    # Get index close from daily_raw_data for VNINDEX
    # Wait, is VNINDEX in daily_raw_data? Let's check.
    df_index_raw = df_raw[df_raw['ticker'] == 'VNINDEX'].copy()
    if df_index_raw.empty:
        print("VNINDEX not found in daily_raw_data, getting from daily_normalized_data")
        # fallback
        df_index = pd.read_sql("SELECT DISTINCT date, high_low as index_close FROM daily_normalized_data", conn)
    else:
        df_index = df_index_raw[['date', 'close']].rename(columns={'close': 'index_close'})
        
    df_index['date'] = pd.to_datetime(df_index['date']).dt.date
    df_index = df_index.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    
    df_raw['date'] = pd.to_datetime(df_raw['date']).dt.date
    
    # We will simulate predictions for 2026-06-01
    target_date = pd.to_datetime('2026-06-01').date()
    
    pred_rows = []
    
    for ticker in tickers:
        df_t = df_raw[df_raw['ticker'] == ticker].copy()
        if df_t.empty:
            continue
            
        df_t = df_t.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        
        # Merge with VNINDEX
        df_merged = pd.merge(df_t, df_index, on='date', how='left')
        df_merged['index_close'] = df_merged['index_close'].ffill().bfill()
        
        if len(df_merged) < 50:
            continue
            
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
        
        # CORRECTED ATR 14: Instead of standard ATR, it should be the daily high-low spread ratio (high - low)/close
        df_merged['atr_14'] = (high - low) / close
        
        df_merged['high_low'] = index_close
        df_merged['market_return'] = index_close.pct_change(1)
        
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
        
        # Get target day row for prediction
        df_target = df_features[df_features['date'] == target_date]
        if df_target.empty:
            continue
            
        latest_row = df_target.iloc[0]
        
        feature_cols_model = [
            'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
            'return_3d', 'return_5d', 'return_10d', 'return_20d',
            'sma_50_LogReturn', 'volume_LogReturn',
            'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
            'atr_14', 'high_low', 'market_return', 'foreign_net'
        ]
        
        # We need to fetch foreign_net for this ticker on this date
        # Let's see if we have it in daily_normalized_data (to avoid recalculating or querying too much)
        # Or we can query the database directly for foreign_net
        cursor_fn = conn.cursor()
        cursor_fn.execute("SELECT foreign_net FROM daily_normalized_data WHERE ticker=%s AND date=%s", (ticker, target_date))
        row_fn = cursor_fn.fetchone()
        foreign_net = row_fn[0] if row_fn else 0.0
        cursor_fn.close()
        
        X_pred = np.array([[
            float(latest_row['price_vs_sma50']),
            float(latest_row['volatility_20']),
            float(latest_row['volume_ratio_20']),
            float(latest_row['return_3d']),
            float(latest_row['return_5d']),
            float(latest_row['return_10d']),
            float(latest_row['return_20d']),
            float(latest_row['sma_50_LogReturn']),
            float(latest_row['volume_LogReturn']),
            float(latest_row['PCA_Trend']),
            float(latest_row['PCA_Oscillators']),
            float(latest_row['PCA_MACD']),
            float(latest_row['PCA_ShortReturns']),
            float(latest_row['atr_14']),
            float(latest_row['high_low']),
            float(latest_row['market_return']),
            float(foreign_net)
        ]])
        
        pred_label = int(model.predict(X_pred)[0])
        proba = model.predict_proba(X_pred)[0].tolist()
        proba_up = float(proba[1])
        proba_down = float(proba[0])
        confidence = abs(proba_up - proba_down)
        
        pred_rows.append({
            'ticker': ticker,
            'prediction': pred_label,
            'probability_up': proba_up,
            'probability_down': proba_down,
            'confidence': confidence,
            'PCA_ShortReturns': latest_row['PCA_ShortReturns'],
            'atr_14': latest_row['atr_14']
        })
        
    df_preds_res = pd.DataFrame(pred_rows)
    print("\n--- Simulation Results (Corrected ATR 14) ---")
    if not df_preds_res.empty:
        print(df_preds_res['prediction'].value_counts())
        print(f"Average probability up: {df_preds_res['probability_up'].mean():.4f}")
        print("\nSome sample predictions:")
        print(df_preds_res.head(15).to_string())
    else:
        print("No predictions generated.")
        
    conn.close()

if __name__ == '__main__':
    main()
