import mysql.connector
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def main():
    print("=" * 70)
    print("  A/B TESTING MODEL: ADDING RRG FEATURES (RS-Ratio & RS-Momentum)")
    print("=" * 70)

    # 1. Connect to Database & Load Data
    print("\n[1/6] Loading data from MySQL...")
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    
    # Load model_training_data
    df_train_data = pd.read_sql("SELECT * FROM model_training_data", conn)
    df_train_data['date'] = pd.to_datetime(df_train_data['date'])
    print(f"    OK Loaded {len(df_train_data):,} rows of training data.")
    
    # Load daily_raw_data for RRG calculation
    df_raw = pd.read_sql("SELECT ticker, date, close FROM daily_raw_data", conn)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.drop_duplicates(subset=['ticker', 'date'])
    print(f"    OK Loaded {len(df_raw):,} rows of raw price data.")
    
    conn.close()

    # 2. RRG Calculation
    print("\n[2/6] Calculating RRG (RS-Ratio & RS-Momentum) daily for all tickers...")
    
    # Extract VNINDEX benchmark close prices
    df_bench = df_raw[df_raw['ticker'] == 'VNINDEX'].sort_values('date').reset_index(drop=True)
    df_bench = df_bench.rename(columns={'close': 'close_bench'})
    
    rrg_dfs = []
    unique_tickers = [t for t in df_train_data['ticker'].unique() if t != 'VNINDEX']
    
    for ticker in unique_tickers:
        df_ticker = df_raw[df_raw['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df_ticker) < 35:
            continue
            
        # Align on date
        merged = pd.merge(df_ticker[['date', 'close']], df_bench[['date', 'close_bench']], on='date', how='inner')
        merged = merged.sort_values('date').reset_index(drop=True)
        
        if len(merged) < 35:
            continue
            
        # Compute RS and RM (Rolling windows: RS = 12/26, RM = 9)
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
        
    df_rrg = pd.concat(rrg_dfs, ignore_index=True)
    print(f"    OK Calculated RRG for {len(rrg_dfs)} tickers. Total RRG rows: {len(df_rrg):,}")

    # 3. Merge RRG into Training Dataset
    print("\n[3/6] Merging RRG features into training dataset...")
    df_merged = pd.merge(df_train_data, df_rrg, on=['ticker', 'date'], how='inner')
    print(f"    OK Merged dataset size: {len(df_merged):,} (after inner join with calculated RRG)")
    
    # Drop rows with NaN in RRG (warm-up periods)
    before_dropna = len(df_merged)
    df_merged = df_merged.dropna(subset=['rs', 'rm'])
    after_dropna = len(df_merged)
    print(f"    OK Dropped {before_dropna - after_dropna:,} rows with NaN in RRG (warm-up). Remaining: {after_dropna:,} rows.")

    # 4. Correlation Analysis
    print("\n[4/6] Correlation Analysis:")
    # Correlation between rs, rm and existing trend features
    corr_features = ['rs', 'rm', 'price_vs_sma50', 'PCA_Trend', 'close_LogReturn']
    corr_matrix = df_merged[corr_features].corr(method='pearson')
    print("    Pearson Correlation Matrix:")
    print(corr_matrix.round(4).to_string())

    # 5. A/B Testing Models Setup
    print("\n[5/6] Splitting Train/Test chronologically (Train: < 2025-01-01, Test: >= 2025-01-01)...")
    
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    train_mask = df_merged['date'] < '2025-01-01'
    test_mask = df_merged['date'] >= '2025-01-01'
    
    df_train = df_merged[train_mask]
    df_test = df_merged[test_mask]
    
    print(f"    Train sample count: {len(df_train):,} ({df_train['date'].min().strftime('%Y-%m-%d')} to {df_train['date'].max().strftime('%Y-%m-%d')})")
    print(f"    Test sample count:  {len(df_test):,} ({df_test['date'].min().strftime('%Y-%m-%d')} to {df_test['date'].max().strftime('%Y-%m-%d')})")
    
    # Define Feature Sets
    baseline_features = [
        'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
        'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'sma_50_LogReturn', 'volume_LogReturn',
        'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
        'atr_14', 'high_low', 'market_return', 'foreign_net',
        'bu', 'sd', 'fs', 'fb'
    ]
    baseline_features = [c for c in baseline_features if c in df_merged.columns]
    rrg_features = baseline_features + ['rs', 'rm']
    
    # Target label
    y_train = df_train['target']
    y_test = df_test['target']
    
    # Compute scale_pos_weight
    train_class_counts = y_train.value_counts()
    scale_pos_weight = train_class_counts.get(0, 1) / train_class_counts.get(1, 1)
    
    # XGBoost Parameters
    xgb_params = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("\nTraining Baseline Model (without RRG)...")
    X_train_base = df_train[baseline_features]
    X_test_base = df_test[baseline_features]
    
    model_base = xgb.XGBClassifier(**xgb_params)
    model_base.fit(X_train_base, y_train)
    print("    OK Baseline training done.")
    
    print("\nTraining Experimental Model (with RRG: rs, rm)...")
    X_train_rrg = df_train[rrg_features]
    X_test_rrg = df_test[rrg_features]
    
    model_rrg = xgb.XGBClassifier(**xgb_params)
    model_rrg.fit(X_train_rrg, y_train)
    print("    OK RRG training done.")

    # 6. Evaluation and Comparison
    print("\n[6/6] Evaluating & Comparing Models...")
    
    # Baseline Predictions
    y_pred_base = model_base.predict(X_test_base)
    y_proba_base = model_base.predict_proba(X_test_base)[:, 1]
    
    # RRG Predictions
    y_pred_rrg = model_rrg.predict(X_test_rrg)
    y_proba_rrg = model_rrg.predict_proba(X_test_rrg)[:, 1]
    
    # Standard Metrics
    def get_metrics(y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        # We care about Precision on class 1 (Buy signals)
        prec_weighted = precision_score(y_true, y_pred, average='weighted')
        prec_class1 = precision_score(y_true, y_pred, pos_label=1)
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_proba)
        return acc, prec_weighted, prec_class1, rec, f1, auc
        
    metrics_base = get_metrics(y_test, y_pred_base, y_proba_base)
    metrics_rrg = get_metrics(y_test, y_pred_rrg, y_proba_rrg)
    
    print("\n=== OVERALL TEST METRICS COMPARISON ===")
    print(f"{'Metric':<25} | {'Baseline Model':<16} | {'RRG Model':<16} | {'Delta':<10}")
    print("-" * 75)
    names = ['Accuracy', 'Weighted Precision', 'Class 1 Precision', 'Weighted Recall', 'Weighted F1-Score', 'AUC-ROC']
    for name, val_base, val_rrg in zip(names, metrics_base, metrics_rrg):
        delta = val_rrg - val_base
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} | {val_base*100:13.2f}% | {val_rrg*100:13.2f}% | {sign}{delta*100:.2f}%")
        
    # Feature Importance of RRG Model
    print("\n=== FEATURE IMPORTANCE (RRG MODEL) ===")
    fi = pd.DataFrame({
        'feature': rrg_features,
        'importance': model_rrg.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    for idx, row in fi.iterrows():
        bar = '#' * int(row['importance'] * 100)
        marker = "* [NEW]" if row['feature'] in ['rs', 'rm'] else "      "
        print(f"    {idx+1:2d}. {row['feature']:20s} | {row['importance']:.4f} | {bar} {marker}")

    # Confidence Filter Performance
    print("\n=== CONFIDENCE FILTER PERFORMANCE (THRESHOLD: 0.15) ===")
    
    def eval_threshold(y_true, y_pred, y_proba, threshold):
        confidences = np.abs(y_proba - (1.0 - y_proba))
        mask = confidences >= threshold
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        
        if len(y_true_sub) > 0:
            acc = accuracy_score(y_true_sub, y_pred_sub)
            prec1 = precision_score(y_true_sub, y_pred_sub, pos_label=1, zero_division=0)
            pct = len(y_true_sub) / len(y_true) * 100
            return acc, prec1, len(y_true_sub), pct
        else:
            return 0.0, 0.0, 0, 0.0

    base_t_acc, base_t_prec1, base_t_cnt, base_t_pct = eval_threshold(y_test, y_pred_base, y_proba_base, 0.15)
    rrg_t_acc, rrg_t_prec1, rrg_t_cnt, rrg_t_pct = eval_threshold(y_test, y_pred_rrg, y_proba_rrg, 0.15)
    
    print(f"Baseline (Conf >= 0.15): Acc = {base_t_acc*100:.2f}%, Buy Precision = {base_t_prec1*100:.2f}%, Samples = {base_t_cnt} ({base_t_pct:.1f}%)")
    print(f"RRG Model (Conf >= 0.15): Acc = {rrg_t_acc*100:.2f}%, Buy Precision = {rrg_t_prec1*100:.2f}%, Samples = {rrg_t_cnt} ({rrg_t_pct:.1f}%)")

if __name__ == '__main__':
    main()
