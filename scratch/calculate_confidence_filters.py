import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import create_engine

def main():
    print("==========================================================")
    print("  RECALCULATING CONFIDENCE FILTER STATS (v4 Model)")
    print("==========================================================")

    # 1. Load data from MySQL
    engine = create_engine("mysql+mysqlconnector://root:170804@localhost/kltn_stock_db")
    df = pd.read_sql("SELECT * FROM model_training_data", engine)
    
    # 2. Preprocess to match training/validation
    df['_ticker'] = df['ticker']
    df['_date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['rs', 'rm'])
    
    # Define features
    feature_cols = [
        'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
        'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'sma_50_LogReturn', 'volume_LogReturn',
        'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
        'atr_14', 'high_low', 'market_return', 'foreign_net',
        'bu', 'sd', 'fs', 'fb', 'rs', 'rm'
    ]
    
    # 3. Split train/test
    test_mask = df['_date'] >= '2025-01-01'
    df_test = df[test_mask].copy()
    
    X_test = df_test[feature_cols]
    y_test = df_test['target'].values
    
    print(f"Total Test Samples (2025-2026): {len(df_test):,}")

    # 4. Load Model and Predict
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Definition A: Confidence = |2 * prob - 1| (Main Dashboard / app.py)
    # Filter: |2 * p - 1| >= t  <=>  p <= 0.5 - t/2  or  p >= 0.5 + t/2
    # e.g., t = 0.20 <=> p <= 40% or p >= 60%
    print("\n--- DEFINITION A: Main Dashboard Filter (|2p-1| >= t) ---")
    print(f"{'Threshold (t)':<15} | {'Prob Range':<22} | {'Samples':<12} | {'Coverage (%)':<12} | {'Accuracy (%)':<12} | {'Precision (%)':<12}")
    print("-" * 92)
    
    prob_1 = y_proba
    prob_0 = 1.0 - y_proba
    conf_a = np.abs(prob_1 - prob_0)
    
    thresholds_a = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for t in thresholds_a:
        mask = conf_a >= t
        y_test_sub = y_test[mask]
        y_pred_sub = y_pred[mask]
        
        if len(y_test_sub) > 0:
            acc = accuracy_score(y_test_sub, y_pred_sub) * 100
            prec = precision_score(y_test_sub, y_pred_sub, average='weighted', zero_division=0) * 100
            cov = len(y_test_sub) / len(y_test) * 100
            prob_min = 0.5 - t/2
            prob_max = 0.5 + t/2
            range_str = f"<= {prob_min*100:.1f}% or >= {prob_max*100:.1f}%"
            print(f"{t:<15.2f} | {range_str:<22} | {len(y_test_sub):<12,} | {cov:<12.2f} | {acc:<12.2f} | {prec:<12.2f}")
            
    # Definition B: Offset from 0.5 (Backtest Dashboard / backtest_app.py)
    # Filter: |p - 0.5| >= offset  <=>  p <= 0.5 - offset  or  p >= 0.5 + offset
    # e.g., offset = 0.20 <=> p <= 30% or p >= 70%
    print("\n--- DEFINITION B: Backtest Dashboard Filter (|p - 0.5| >= offset) ---")
    print(f"{'Offset (t)':<15} | {'Prob Range':<22} | {'Samples':<12} | {'Coverage (%)':<12} | {'Accuracy (%)':<12} | {'Precision (%)':<12}")
    print("-" * 92)
    
    offsets_b = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for t in offsets_b:
        mask = (y_proba <= 0.5 - t) | (y_proba >= 0.5 + t)
        y_test_sub = y_test[mask]
        y_pred_sub = y_pred[mask]
        
        if len(y_test_sub) > 0:
            acc = accuracy_score(y_test_sub, y_pred_sub) * 100
            prec = precision_score(y_test_sub, y_pred_sub, average='weighted', zero_division=0) * 100
            cov = len(y_test_sub) / len(y_test) * 100
            prob_min = 0.5 - t
            prob_max = 0.5 + t
            range_str = f"<= {prob_min*100:.1f}% or >= {prob_max*100:.1f}%"
            print(f"{t:<15.2f} | {range_str:<22} | {len(y_test_sub):<12,} | {cov:<12.2f} | {acc:<12.2f} | {prec:<12.2f}")

if __name__ == '__main__':
    main()
