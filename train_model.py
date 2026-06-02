"""
XGBoost Training Script v2 - No Data Leakage
Loại bỏ rsi_14_LogReturn và return_1d để tránh data leakage
Thêm per-ticker statistics để web app sử dụng
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  XGBoost Training v2 - Stock Market Prediction")
print("  (Da loai bo cac cot gay data leakage)")
print("=" * 60)

# ─── 1. LOAD DATA ────────────────────────────────────────────
print("\n[1/7] Loading training dataset ...")
loaded_from_db = False

try:
    from sqlalchemy import create_engine
    engine = create_engine("mysql+mysqlconnector://root:170804@localhost/kltn_stock_db")
    df = pd.read_sql("SELECT * FROM model_training_data", engine)
    df['_ticker'] = df['ticker']
    df['_date'] = pd.to_datetime(df['date'])
    print(f"    OK Loaded dataset from MySQL database table 'model_training_data'. Total rows: {len(df):,}")
    # Drop rows with NaN in RRG features
    if 'rs' in df.columns and 'rm' in df.columns:
        before_drop = len(df)
        df = df.dropna(subset=['rs', 'rm'])
        print(f"    OK Dropped {before_drop - len(df):,} rows with NaN in RRG features. Remaining rows: {len(df):,}")
    loaded_from_db = True
except Exception as e:
    print(f"    ⚠️ Failed to load from MySQL database: {e}")
    print("    Falling back to Excel files (data2.xlsx and database_export)...")

if not loaded_from_db:
    df_raw = pd.read_excel('data2.xlsx')
    print(f"    OK Tong so dong : {len(df_raw):,}")
    print(f"    OK Tong so cot  : {df_raw.shape[1]}")
    
    print("    Loading 'foreign_net' from the Excel export file...")
    excel_path = r"D:\Khóa luận tốt nghiệp\phân tích data\data v1\database_export_20260518_1159.xlsx"
    df_indicators = pd.read_excel(
        excel_path,
        sheet_name='technical_indicators',
        usecols=['ticker', 'date', 'foreign_net']
    )
    df_indicators['date'] = pd.to_datetime(df_indicators['date'])
    df_indicators = df_indicators.drop_duplicates(subset=['ticker', 'date'])
    
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    print("    Merging 'foreign_net' into training data...")
    df_raw = pd.merge(df_raw, df_indicators, on=['ticker', 'date'], how='left')
    df_raw['foreign_net'] = df_raw['foreign_net'].fillna(0.0)
    
    ticker_col = df_raw['ticker'].copy() if 'ticker' in df_raw.columns else pd.Series(['UNKNOWN'] * len(df_raw))
    
    # ─── 2. XỬ LÝ DỮ LIỆU ───────────────────────────────────────
    print("\n[2/7] Xu ly du lieu ...")
    drop_cols = [
        'id', 'ticker_id', 'ticker', 'date',
        'Unnamed: 23', 'Unnamed: 24',
        'rsi_14_LogReturn',
        'return_1d',
    ]
    df = df_raw.copy()
    df['_ticker'] = ticker_col
    df['_date'] = pd.to_datetime(df_raw['date']) if 'date' in df_raw.columns else pd.to_datetime('2021-01-01')
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"    OK Da xoa {before - after:,} hang co gia tri null (con lai: {after:,} hang)")
    
    df['close_LogReturn_5d'] = (
        df.groupby('_ticker')['close_LogReturn'].shift(-1) +
        df.groupby('_ticker')['close_LogReturn'].shift(-2) +
        df.groupby('_ticker')['close_LogReturn'].shift(-3) +
        df.groupby('_ticker')['close_LogReturn'].shift(-4) +
        df.groupby('_ticker')['close_LogReturn'].shift(-5)
    )
    
    before_drop = len(df)
    df = df.dropna(subset=['close_LogReturn_5d'])
    after_drop = len(df)
    print(f"    OK Da xoa {before_drop - after_drop:,} hang o cuoi chuoi do thieu du lieu 5 phien tiep theo")
    
    df['target'] = (df['close_LogReturn_5d'] > 0).astype(int)

# Sắp xếp toàn bộ DataFrame theo dòng thời gian tăng dần
df = df.sort_values('_date').reset_index(drop=True)

class_counts = df['target'].value_counts()
print(f"    OK Phan bo nhan (du bao 5 phien):")
print(f"      - Tang (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"      - Giam (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")

# Các cột feature (23 cột, đã bỏ 2 cột leaky, thêm foreign_net, bu, sd, fs, fb và rs, rm)
feature_cols = [
    'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
    'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'sma_50_LogReturn', 'volume_LogReturn',
    'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
    'atr_14', 'high_low', 'market_return', 'foreign_net',
    'bu', 'sd', 'fs', 'fb', 'rs', 'rm'
]
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"    OK So features (sau khi loai leaky): {len(feature_cols)}")
print(f"    OK Da bo: rsi_14_LogReturn, return_1d")

# ─── 3. PER-TICKER STATISTICS ────────────────────────────────
print("\n[3/7] Tinh thong ke per-ticker ...")
ticker_stats = {}
unique_tickers = sorted(df['_ticker'].dropna().unique().tolist())
print(f"    OK Tim thay {len(unique_tickers)} ma co phieu")

for t in unique_tickers:
    mask = df['_ticker'] == t
    sub = df[mask]
    if len(sub) < 5:
        continue
    medians = sub[feature_cols].median().round(6).to_dict()
    up_count = int(sub['target'].sum())
    down_count = int((~sub['target'].astype(bool)).sum())
    total = len(sub)
    ticker_stats[t] = {
        "ticker": t,
        "total_samples": total,
        "up": up_count,
        "down": down_count,
        "up_pct": round(up_count / total * 100, 1),
        "down_pct": round(down_count / total * 100, 1),
        "median_features": medians
    }

print(f"    OK Da tinh stats cho {len(ticker_stats)} ma")

X = df[feature_cols]
y = df['target']

# ─── 4. CHIA TRAIN/TEST CHRONOLOGICAL ─────────────────────────
print("\n[4/7] Chia train/test theo thoi gian (Train: 2021-2024, Test: 2025-2026) ...")
train_mask = df['_date'] < '2025-01-01'
test_mask = df['_date'] >= '2025-01-01'

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# In ra thong tin thoi gian de de dang bao cao
print(f"    OK Train (2021-2024): {len(X_train):,} mau ({df.loc[train_mask, '_date'].min().strftime('%Y-%m-%d')} -> {df.loc[train_mask, '_date'].max().strftime('%Y-%m-%d')})")
print(f"    OK Test (tu 2025): {len(X_test):,} mau ({df.loc[test_mask, '_date'].min().strftime('%Y-%m-%d')} -> {df.loc[test_mask, '_date'].max().strftime('%Y-%m-%d')})")

# ─── 5. TRAINING XGBOOST ─────────────────────────────────────
print("\n[5/7] Huan luyen mo hinh XGBoost ...")
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
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# eval_set co train va test
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=50
)
print("    OK Training hoan thanh!")

# ─── 6. ĐÁNH GIÁ ─────────────────────────────────────────────
print("\n[6/7] Danh gia mo hinh ...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Calculate test accuracy per ticker
test_df = df[test_mask].copy()
test_df['y_true'] = y_test
test_df['y_pred'] = y_pred

for t in unique_tickers:
    t_mask = test_df['_ticker'] == t
    t_sub = test_df[t_mask]
    if len(t_sub) > 0:
        ta = accuracy_score(t_sub['y_true'], t_sub['y_pred'])
        if t in ticker_stats:
            ticker_stats[t]['test_accuracy'] = round(float(ta), 4)
            ticker_stats[t]['test_samples'] = int(len(t_sub))
            # classify predictability
            if ta >= 0.53:
                ticker_stats[t]['predictability'] = 'high'
            elif ta < 0.50:
                ticker_stats[t]['predictability'] = 'low'
            else:
                ticker_stats[t]['predictability'] = 'medium'
    else:
        if t in ticker_stats:
            ticker_stats[t]['test_accuracy'] = 0.0
            ticker_stats[t]['test_samples'] = 0
            ticker_stats[t]['predictability'] = 'low'

print(f"\n    === KET QUA DANH GIA MO HINH ===")
print(f"    Accuracy  : {acc*100:.2f}%")
print(f"    Precision : {precision*100:.2f}%")
print(f"    Recall    : {recall*100:.2f}%")
print(f"    F1-Score  : {f1*100:.2f}%")
print(f"    AUC-ROC   : {auc:.4f}")

# Feature Importance
fi = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n    Top Features quan trong nhat:")
for _, row in fi.head(8).iterrows():
    bar = '#' * int(row['importance'] * 100)
    print(f"      {row['feature']:25s} {row['importance']:.4f} {bar}")

# Learning curves
evals_result = model.evals_result()
train_logloss = evals_result['validation_0']['logloss']
val_logloss   = evals_result['validation_1']['logloss']

# Calculate metrics at different confidence thresholds on the test set
thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
threshold_metrics = {}
prob_1 = y_pred_proba
prob_0 = 1.0 - y_pred_proba
confidences = np.abs(prob_1 - prob_0)

for t in thresholds:
    mask = confidences >= t
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    y_pred_proba_sub = y_pred_proba[mask]
    
    if len(y_test_sub) > 0:
        if len(np.unique(y_test_sub)) > 1:
            sub_auc = roc_auc_score(y_test_sub, y_pred_proba_sub)
        else:
            sub_auc = 0.5
            
        sub_acc = accuracy_score(y_test_sub, y_pred_sub)
        sub_precision = precision_score(y_test_sub, y_pred_sub, average='weighted', zero_division=0)
        sub_recall = recall_score(y_test_sub, y_pred_sub, average='weighted', zero_division=0)
        sub_f1 = f1_score(y_test_sub, y_pred_sub, average='weighted', zero_division=0)
        sub_cm = confusion_matrix(y_test_sub, y_pred_sub)
        sub_report = classification_report(y_test_sub, y_pred_sub, output_dict=True, zero_division=0)
        
        threshold_metrics[str(t)] = {
            "accuracy": round(float(sub_acc), 4),
            "precision": round(float(sub_precision), 4),
            "recall": round(float(sub_recall), 4),
            "f1_score": round(float(sub_f1), 4),
            "auc_roc": round(float(sub_auc), 4),
            "samples": int(len(y_test_sub)),
            "pct_samples": round(float(len(y_test_sub) / len(y_test) * 100), 2),
            "confusion_matrix": sub_cm.tolist(),
            "classification_report": {
                "class_0": {
                    "precision": round(float(sub_report.get('0', {}).get('precision', 0)), 4),
                    "recall": round(float(sub_report.get('0', {}).get('recall', 0)), 4),
                    "f1_score": round(float(sub_report.get('0', {}).get('f1-score', 0)), 4)
                },
                "class_1": {
                    "precision": round(float(sub_report.get('1', {}).get('precision', 0)), 4),
                    "recall": round(float(sub_report.get('1', {}).get('recall', 0)), 4),
                    "f1_score": round(float(sub_report.get('1', {}).get('f1-score', 0)), 4)
                }
            }
        }
    else:
        threshold_metrics[str(t)] = None

# ─── 7. LƯU ─────────────────────────────────────────────────
print("\n[7/7] Luu model va thong so ...")
model.save_model('xgb_model.json')

with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

metrics = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(precision), 4),
    "threshold_metrics": threshold_metrics,
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "auc_roc": round(float(auc), 4),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "total_samples": int(len(df)),
    "n_features": int(len(feature_cols)),
    "feature_cols": feature_cols,
    "class_distribution": {
        "tang": int(class_counts.get(1, 0)),
        "giam": int(class_counts.get(0, 0))
    },
    "confusion_matrix": cm.tolist(),
    "classification_report": {
        "class_0": {
            "precision": round(report['0']['precision'], 4),
            "recall": round(report['0']['recall'], 4),
            "f1_score": round(report['0']['f1-score'], 4),
            "support": int(report['0']['support'])
        },
        "class_1": {
            "precision": round(report['1']['precision'], 4),
            "recall": round(report['1']['recall'], 4),
            "f1_score": round(report['1']['f1-score'], 4),
            "support": int(report['1']['support'])
        }
    },
    "feature_importance": [
        {"feature": row['feature'], "importance": round(float(row['importance']), 6)}
        for _, row in fi.iterrows()
    ],
    "learning_curves": {
        "train_logloss": [round(v, 6) for v in train_logloss[::5]],
        "val_logloss":   [round(v, 6) for v in val_logloss[::5]],
        "epochs": list(range(0, len(train_logloss), 5))
    },
    "model_params": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "scale_pos_weight": round(float(scale_pos_weight), 4)
    },
    "removed_features": ["rsi_14_LogReturn", "return_1d"],
    "unique_tickers": unique_tickers
}

with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

with open('ticker_stats.json', 'w', encoding='utf-8') as f:
    json.dump(ticker_stats, f, ensure_ascii=False, indent=2)

print("    OK Da luu: xgb_model.json")
print("    OK Da luu: model_metrics.json")
print("    OK Da luu: ticker_stats.json")
print(f"\n{'='*60}")
print("  HOAN TAT! Mo hinh v2 da san sang.")
print(f"{'='*60}")
