import pandas as pd
import numpy as np
import mysql.connector
import xgboost as xgb
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    print("1. Loading training data to retrain models without high_low...")
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    df_train = pd.read_sql("SELECT * FROM model_training_data", conn)
    df_train['_ticker'] = df_train['ticker']
    df_train['_date'] = pd.to_datetime(df_train['date'])
    df_train = df_train.sort_values('_date').reset_index(drop=True)
    
    # Define features without high_low
    non_pca_cols = [
        'price_vs_sma50', 'volatility_20', 'volume_ratio_20', 
        'return_3d', 'return_5d', 'return_10d', 'return_20d', 
        'sma_50_LogReturn', 'volume_LogReturn', 'atr_14', 
        'market_return'
    ]
    pca_cols = ['PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns']
    
    # 2. Retrain PCA Mapper without high_low
    print("Retraining PCA mapper...")
    df_clean = df_train[non_pca_cols + pca_cols].dropna()
    X_pca = df_clean[non_pca_cols]
    y_pca = df_clean[pca_cols]
    X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)
    
    pca_mapper = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    pca_mapper.fit(X_pca_train, y_pca_train)
    print("PCA mapper retrained.")
    
    # 3. Retrain XGBoost Classifier without high_low
    print("Retraining XGBoost classifier...")
    feature_cols_model = [
        'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
        'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'sma_50_LogReturn', 'volume_LogReturn',
        'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
        'atr_14', 'market_return', 'foreign_net'
    ]
    
    X_xgb = df_train[feature_cols_model]
    y_xgb = df_train['target']
    
    train_mask = df_train['_date'] < '2025-01-01'
    X_train = X_xgb[train_mask]
    y_train = y_xgb[train_mask]
    
    train_class_counts = y_train.value_counts()
    scale_pos_weight = train_class_counts.get(0, 1) / train_class_counts.get(1, 1)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("XGBoost classifier retrained.")
    
    # 4. Simulate daily pipeline for 2026-06-01
    print("Simulating daily pipeline for 2026-06-01...")
    tickers = sorted(df_train['ticker'].unique())
    df_raw = pd.read_sql("SELECT ticker, date, open, high, low, close, volume FROM daily_raw_data", conn)
    
    df_index_raw = df_raw[df_raw['ticker'] == 'VNINDEX'].copy()
    if df_index_raw.empty:
        df_index = pd.read_sql("SELECT DISTINCT date, high_low as index_close FROM daily_normalized_data", conn)
    else:
        df_index = df_index_raw[['date', 'close']].rename(columns={'close': 'index_close'})
    df_index['date'] = pd.to_datetime(df_index['date']).dt.date
    df_index = df_index.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    
    df_raw['date'] = pd.to_datetime(df_raw['date']).dt.date
    target_date = pd.to_datetime('2026-06-01').date()
    
    pred_rows = []
    for ticker in tickers:
        df_t = df_raw[df_raw['ticker'] == ticker].copy()
        if df_t.empty:
            continue
        df_t = df_t.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        df_merged = pd.merge(df_t, df_index, on='date', how='left')
        df_merged['index_close'] = df_merged['index_close'].ffill().bfill()
        if len(df_merged) < 50:
            continue
            
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
        
        # Corrected atr_14: (high - low)/close
        df_merged['atr_14'] = (high - low) / close
        df_merged['market_return'] = index_close.pct_change(1)
        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df_features = df_merged.dropna(subset=['close_LogReturn'] + non_pca_cols).copy()
        if df_features.empty:
            continue
            
        X_base = df_features[non_pca_cols]
        pca_preds = pca_mapper.predict(X_base)
        df_features['PCA_Trend'] = pca_preds[:, 0]
        df_features['PCA_Oscillators'] = pca_preds[:, 1]
        df_features['PCA_MACD'] = pca_preds[:, 2]
        df_features['PCA_ShortReturns'] = pca_preds[:, 3]
        
        df_target = df_features[df_features['date'] == target_date]
        if df_target.empty:
            continue
        latest_row = df_target.iloc[0]
        
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
            'confidence': confidence
        })
        
    df_res = pd.DataFrame(pred_rows)
    print("\n--- Simulation Results (No high_low + Corrected ATR 14) ---")
    if not df_res.empty:
        print(df_res['prediction'].value_counts())
        print(f"Average probability up: {df_res['probability_up'].mean():.4f}")
        print("\nTop 10 highest confidence predictions:")
        print(df_res.sort_values('confidence', ascending=False).head(10).to_string())
    else:
        print("No predictions generated.")
        
    conn.close()

if __name__ == '__main__':
    main()
