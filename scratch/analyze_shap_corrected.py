import pandas as pd
import numpy as np
import mysql.connector
import xgboost as xgb
import shap

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    with open("pca_predictor.pkl", "rb") as f:
        pca_resource = pickle_load = None
        import pickle
        with open("pca_predictor.pkl", "rb") as f2:
            pca_resource = pickle.load(f2)
    pca_mapper = pca_resource['model']
    
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    
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
    rows = []
    
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
            
        X_base = df_features[feature_cols_base]
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
        
        rows.append({
            'ticker': ticker,
            'price_vs_sma50': float(latest_row['price_vs_sma50']),
            'volatility_20': float(latest_row['volatility_20']),
            'volume_ratio_20': float(latest_row['volume_ratio_20']),
            'return_3d': float(latest_row['return_3d']),
            'return_5d': float(latest_row['return_5d']),
            'return_10d': float(latest_row['return_10d']),
            'return_20d': float(latest_row['return_20d']),
            'sma_50_LogReturn': float(latest_row['sma_50_LogReturn']),
            'volume_LogReturn': float(latest_row['volume_LogReturn']),
            'PCA_Trend': float(latest_row['PCA_Trend']),
            'PCA_Oscillators': float(latest_row['PCA_Oscillators']),
            'PCA_MACD': float(latest_row['PCA_MACD']),
            'PCA_ShortReturns': float(latest_row['PCA_ShortReturns']),
            'atr_14': float(latest_row['atr_14']),
            'high_low': float(latest_row['high_low']),
            'market_return': float(latest_row['market_return']),
            'foreign_net': float(foreign_net)
        })
        
    df_eval = pd.DataFrame(rows)
    feature_cols_model = [
        'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
        'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'sma_50_LogReturn', 'volume_LogReturn',
        'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
        'atr_14', 'high_low', 'market_return', 'foreign_net'
    ]
    
    X = df_eval[feature_cols_model]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    mean_signed_shap = np.mean(shap_values, axis=0)
    
    shap_df = pd.DataFrame({
        'feature': feature_cols_model,
        'mean_abs_shap': mean_shap,
        'mean_signed_shap': mean_signed_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n--- Corrected SHAP feature contributions for 2026-06-01 ---")
    print(shap_df.to_string(index=False))
    conn.close()

if __name__ == '__main__':
    main()
