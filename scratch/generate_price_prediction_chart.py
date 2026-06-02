import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# Matplotlib styling for thesis
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Georgia']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    print("=" * 60)
    print("  GENERATING ACTUAL VS PREDICTED PRICE CHART FOR THESIS")
    print("=" * 60)
    
    # 1. Connect and Load Model
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
        
    # 2. Select a ticker (e.g. ACB)
    ticker = 'ACB'
    print(f"Loading data for ticker: {ticker}...")
    
    # Load all normalized features and raw close prices
    # We join model_training_data and daily_raw_data to get both close prices and model features
    query = text("""
        SELECT n.date, r.close, n.close_LogReturn,
               n.price_vs_sma50, n.volatility_20, n.volume_ratio_20,
               n.return_3d, n.return_5d, n.return_10d, n.return_20d,
               n.sma_50_LogReturn, n.volume_LogReturn,
               n.PCA_Trend, n.PCA_Oscillators, n.PCA_MACD, n.PCA_ShortReturns,
               n.atr_14, n.high_low, n.market_return, n.foreign_net,
               n.bu, n.sd, n.fs, n.fb, n.rs, n.rm
        FROM model_training_data n
        JOIN daily_raw_data r ON n.ticker = r.ticker AND n.date = r.date
        WHERE n.ticker = :ticker AND n.date >= '2024-12-01'
        ORDER BY n.date
    """)
    
    df = pd.read_sql(query, engine, params={'ticker': ticker})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    if df.empty:
        print("❌ Error: No data found for ticker FPT")
        return
        
    # Filter out NaNs in features
    df_clean = df.dropna(subset=feature_cols).copy()
    
    # 3. Predict Up Probability (y_proba)
    X = df_clean[feature_cols]
    df_clean['y_proba'] = model.predict_proba(X)[:, 1]
    
    # 4. Reconstruct Predicted Close Price at t+5 using prediction at t
    # Expected log return = (proba - 0.5) * scaling_factor
    # We use a scaling factor of 0.08 (approx 8% max expected return over 5 sessions)
    scaling_factor = 0.08
    df_clean['expected_log_ret'] = (df_clean['y_proba'] - 0.5) * scaling_factor
    
    # Predicted close price for day t+5 is Close_t * exp(expected_log_ret)
    df_clean['predicted_close_t5'] = df_clean['close'] * np.exp(df_clean['expected_log_ret'])
    
    # Shift predicted price by 5 days so that the prediction made at t-5 aligns with actual price at t
    df_clean['predicted_close'] = df_clean['predicted_close_t5'].shift(5)
    
    # 5. Filter for Test Set (from 2025-01-01 onwards)
    test_df = df_clean[df_clean['date'] >= '2025-01-01'].copy().dropna(subset=['predicted_close'])
    
    # 6. Plotting
    plt.figure(figsize=(10, 5.5), dpi=300)
    
    plt.plot(test_df['date'], test_df['close'], label='Dữ liệu cổ phiếu thực tế (Actual)', color='#1f77b4', linewidth=1.8)
    plt.plot(test_df['date'], test_df['predicted_close'], label='Mô hình XGBoost v4 dự đoán (Predicted T+5)', color='#ff7f0e', linewidth=1.5, linestyle='-.')
    
    plt.title(f'Giá đóng cửa thực tế và dự báo T+5 của cổ phiếu {ticker} (01/2025 - 06/2026)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Thời gian', labelpad=10)
    plt.ylabel('Giá đóng cửa (VNĐ)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Create directory if not exists
    os.makedirs('reports/charts', exist_ok=True)
    
    output_path = 'reports/charts/chart4_price_prediction.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"SUCCESS: Saved price prediction chart to {output_path}")

if __name__ == '__main__':
    main()
