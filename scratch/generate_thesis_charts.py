import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine, text

# Set professional style parameters for matplotlib
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Georgia']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

FEATURE_LABELS_VI = {
    'price_vs_sma50': 'Giá so với đường SMA50',
    'volatility_20': 'Độ biến động (20 phiên)',
    'volume_ratio_20': 'Tỷ lệ khối lượng (20 phiên)',
    'return_3d': 'Tỷ suất sinh lợi 3 phiên',
    'return_5d': 'Tỷ suất sinh lợi 5 phiên',
    'return_10d': 'Tỷ suất sinh lợi 10 phiên',
    'return_20d': 'Tỷ suất sinh lợi 20 phiên',
    'sma_50_LogReturn': 'Tỷ suất log đường SMA50',
    'volume_LogReturn': 'Tỷ suất log khối lượng',
    'PCA_Trend': 'PCA - Xu hướng dài hạn',
    'PCA_Oscillators': 'PCA - Chỉ báo động lượng',
    'PCA_MACD': 'PCA - Chỉ báo MACD',
    'PCA_ShortReturns': 'PCA - Tỷ suất ngắn hạn',
    'atr_14': 'Độ biến động ATR (14 phiên)',
    'high_low': 'Biên độ cao - thấp trong ngày',
    'market_return': 'Tỷ suất sinh lợi thị trường',
    'foreign_net': 'Giao dịch ròng khối ngoại',
    'bu': 'Khối lượng mua chủ động',
    'sd': 'Khối lượng bán chủ động',
    'fs': 'Khối lượng tự doanh mua',
    'fb': 'Khối lượng tự doanh bán',
    'rs': 'RRG RS-Ratio (Sức mạnh tương đối)',
    'rm': 'RRG RS-Momentum (Động lượng xoay vòng)'
}

def get_engine():
    return create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

def load_model():
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

def generate_chart_1_accuracy_tradeoff():
    print("[1/3] Generating Chart 1: Accuracy vs Trade Coverage Trade-off...")
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    thresholds = []
    accuracies = []
    coverage = []
    
    for t_str, m in metrics['threshold_metrics'].items():
        thresholds.append(float(t_str))
        accuracies.append(m['accuracy'] * 100)
        coverage.append(m['pct_samples'])
        
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    # Plot Accuracy
    color = '#1f77b4'
    ax1.set_xlabel('Ngưỡng lệch bộ lọc tự tin (Threshold Offset)', labelpad=10)
    ax1.set_ylabel('Độ chính xác (Accuracy, %)', color=color)
    line1 = ax1.plot(thresholds, accuracies, marker='o', color=color, linewidth=2, label='Độ chính xác (Accuracy)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Coverage
    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Tỷ lệ mẫu giữ lại (Coverage, %)', color=color)
    line2 = ax2.plot(thresholds, coverage, marker='s', color=color, linewidth=2, linestyle='--', label='Tỷ lệ cơ hội giao dịch')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight baseline
    ax1.axhline(y=accuracies[0], color='gray', linestyle=':', alpha=0.7, label='Mức cơ sở (Không lọc)')
    
    # Title & Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    
    plt.title('Đánh đổi giữa Độ chính xác và Tỷ lệ cơ hội giao dịch', pad=15, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'reports/charts/chart1_accuracy_tradeoff.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"    OK Saved to {output_path}")

def generate_chart_2_backtest_cumulative():
    print("[2/3] Generating Chart 2: Backtest Cumulative Returns...")
    engine = get_engine()
    model, feature_cols = load_model()
    
    # Load raw training data table to isolate Test set
    df = pd.read_sql("SELECT * FROM model_training_data ORDER BY date", engine)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    if 'rs' in df.columns and 'rm' in df.columns:
        df = df.dropna(subset=['rs', 'rm'])
        
    test_df = df[df['date'] >= '2025-01-01'].copy()
    
    # Run predictions
    test_df['y_pred'] = model.predict(test_df[feature_cols]).astype(int)
    test_df['y_proba'] = model.predict_proba(test_df[feature_cols])[:, 1]
    
    # Get VNINDEX data for benchmark comparison
    df_vnindex = pd.read_sql("""
        SELECT date, close FROM daily_raw_data 
        WHERE ticker = 'VNINDEX' AND date >= '2025-01-01'
        ORDER BY date
    """, engine)
    df_vnindex['date'] = pd.to_datetime(df_vnindex['date'])
    df_vnindex = df_vnindex.sort_values('date').reset_index(drop=True)
    df_vnindex['vnindex_return'] = df_vnindex['close'].pct_change().fillna(0.0)
    df_vnindex['vnindex_cum'] = (1 + df_vnindex['vnindex_return']).cumprod() - 1
    
    # Compile daily model returns
    dates = sorted(test_df['date'].unique())
    
    results = []
    for d in dates:
        day_df = test_df[test_df['date'] == d]
        
        # Scenario 1: No Filter
        buy_signals_all = day_df[day_df['y_pred'] == 1]
        ret_all = 0.0
        if not buy_signals_all.empty:
            # We approximate daily return of holding 5 sessions by dividing the 5-day log return by 5
            # to generate a daily return proxy for compounding.
            ret_all = np.mean(np.exp(buy_signals_all['close_LogReturn']) - 1)
            
        # Scenario 2: recommended offset (>= 0.10)
        # 10% offset means proba_up <= 0.40 or proba_up >= 0.60
        day_df_filtered = day_df[(day_df['y_proba'] <= 0.40) | (day_df['y_proba'] >= 0.60)]
        buy_signals_filter = day_df_filtered[day_df_filtered['y_pred'] == 1]
        ret_filter = 0.0
        if not buy_signals_filter.empty:
            ret_filter = np.mean(np.exp(buy_signals_filter['close_LogReturn']) - 1)
            
        results.append({
            'date': d,
            'ret_all': ret_all,
            'ret_filter': ret_filter
        })
        
    df_strat = pd.DataFrame(results)
    df_strat['cum_all'] = (1 + df_strat['ret_all']).cumprod() - 1
    df_strat['cum_filter'] = (1 + df_strat['ret_filter']).cumprod() - 1
    
    # Merge with VNINDEX benchmark
    df_merged = pd.merge(df_strat, df_vnindex[['date', 'vnindex_cum']], on='date', how='left')
    df_merged['vnindex_cum'] = df_merged['vnindex_cum'].ffill().fillna(0.0)
    
    plt.figure(figsize=(9, 5), dpi=300)
    
    # Plot strategy cumulative returns
    plt.plot(df_merged['date'], df_merged['cum_filter'] * 100, label='Mô hình XGBoost + Bộ lọc tự tin (Offset >= 0.10)', color='#2ca02c', linewidth=2.2)
    plt.plot(df_merged['date'], df_merged['cum_all'] * 100, label='Mô hình XGBoost (Không lọc)', color='#1f77b4', linewidth=1.5, linestyle='--')
    plt.plot(df_merged['date'], df_merged['vnindex_cum'] * 100, label='Chỉ số VNINDEX (Buy & Hold)', color='#d62728', linewidth=1.5, alpha=0.8)
    
    plt.title('Hiệu năng Lũy kế Chiến lược đầu tư trên tập Test (2025 - 2026)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Thời gian', labelpad=10)
    plt.ylabel('Hiệu suất lũy kế (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    output_path = 'reports/charts/chart2_backtest_performance.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"    OK Saved to {output_path}")

def generate_chart_3_feature_importance():
    print("[3/3] Generating Chart 3: Feature Importance...")
    model, feature_cols = load_model()
    
    importance = model.feature_importances_
    
    df_fi = pd.DataFrame({
        'raw_feature': feature_cols,
        'importance': importance
    })
    
    # Translate feature names to Vietnamese
    df_fi['feature'] = df_fi['raw_feature'].map(FEATURE_LABELS_VI).fillna(df_fi['raw_feature'])
    
    # Sort and take top 15
    df_fi = df_fi.sort_values('importance', ascending=True).tail(15)
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    # Create horizontal bar plot
    bars = plt.barh(df_fi['feature'], df_fi['importance'] * 100, color='#17becf', edgecolor='none', height=0.6)
    
    # Highlight new RRG features with a different color
    for idx, row in df_fi.reset_index(drop=True).iterrows():
        if row['raw_feature'] in ['rs', 'rm']:
            bars[idx].set_color('#bcbd22') # Highlight color
            
    plt.title('Đánh giá Tầm quan trọng của các Đặc trưng (Top 15)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Tỷ lệ đóng góp quan trọng (Gini Importance, %)', labelpad=10)
    plt.ylabel('Các đặc trưng đầu vào')
    plt.grid(True, axis='x', linestyle=':', alpha=0.5)
    
    # Add values at the end of bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                 va='center', ha='left', fontsize=8.5, color='#333333')
        
    plt.xlim(0, max(df_fi['importance'] * 100) + 1.2)
    plt.tight_layout()
    
    output_path = 'reports/charts/chart3_feature_importance.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"    OK Saved to {output_path}")

def main():
    print("=" * 60)
    print("  MATPLOTLIB ACADEMIC CHARTS GENERATOR FOR THESIS")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('reports/charts', exist_ok=True)
    
    try:
        generate_chart_1_accuracy_tradeoff()
        generate_chart_2_backtest_cumulative()
        generate_chart_3_feature_importance()
        print("\nSUCCESS: All graduation thesis charts generated successfully under 'reports/charts/'!")
    except Exception as e:
        print(f"\nERROR: Generating charts failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
