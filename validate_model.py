"""
=============================================================
  SCRIPT XÁC THỰC ĐỘ CHÍNH XÁC MÔ HÌNH XGBOOST v2
  Chạy cùng thư mục với: xgb_model.json, data2.xlsx
=============================================================
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.dummy import DummyClassifier

SEP = "=" * 65

# ─── LOAD DỮ LIỆU ─────────────────────────────────────────────
print(SEP)
print("  LOAD DỮ LIỆU & MODEL")
print(SEP)

loaded_from_db = False

try:
    from sqlalchemy import create_engine
    engine = create_engine("mysql+mysqlconnector://root:170804@localhost/kltn_stock_db")
    df = pd.read_sql("SELECT * FROM model_training_data", engine)
    df['_ticker'] = df['ticker']
    df['_date'] = pd.to_datetime(df['date'])
    print(f"    OK Loaded dataset from MySQL database table 'model_training_data'. Total rows: {len(df):,}")
    loaded_from_db = True
except Exception as e:
    print(f"    ⚠️ Failed to load from MySQL database: {e}")
    print("    Falling back to Excel files...")

if not loaded_from_db:
    df_raw = pd.read_excel('data2.xlsx')
    
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
    print("    Merging 'foreign_net' into validation data...")
    df_raw = pd.merge(df_raw, df_indicators, on=['ticker', 'date'], how='left')
    df_raw['foreign_net'] = df_raw['foreign_net'].fillna(0.0)
    
    ticker_col = df_raw['ticker'].copy() if 'ticker' in df_raw.columns else pd.Series(['?'] * len(df_raw))
    
    drop_cols = ['id','ticker_id','ticker','date','Unnamed: 23','Unnamed: 24',
                 'rsi_14_LogReturn','return_1d']
    df = df_raw.copy()
    df['_ticker'] = ticker_col
    df['_date'] = pd.to_datetime(df_raw['date']) if 'date' in df_raw.columns else pd.to_datetime('2021-01-01')
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna()
    
    # Tạo nhãn mục tiêu cho 5 phiên tiếp theo (T+1 -> T+5)
    df['close_LogReturn_5d'] = (
        df.groupby('_ticker')['close_LogReturn'].shift(-1) +
        df.groupby('_ticker')['close_LogReturn'].shift(-2) +
        df.groupby('_ticker')['close_LogReturn'].shift(-3) +
        df.groupby('_ticker')['close_LogReturn'].shift(-4) +
        df.groupby('_ticker')['close_LogReturn'].shift(-5)
    )
    
    # Loại bỏ hàng không đủ dữ liệu 5 phiên tiếp theo ở cuối chuỗi của các mã
    df = df.dropna(subset=['close_LogReturn_5d'])
    df['target'] = (df['close_LogReturn_5d'] > 0).astype(int)
    
    # Sắp xếp toàn bộ DataFrame theo dòng thời gian tăng dần
    df = df.sort_values('_date').reset_index(drop=True)

feature_cols = ['price_vs_sma50','volatility_20','volume_ratio_20',
                'return_3d','return_5d','return_10d','return_20d',
                'sma_50_LogReturn','volume_LogReturn',
                'PCA_Trend','PCA_Oscillators','PCA_MACD','PCA_ShortReturns',
                'atr_14','high_low','market_return','foreign_net']
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['target']
ticker_series = df['_ticker']  # Lưu riêng trước khi split

model = xgb.XGBClassifier()
model.load_model('xgb_model.json')

# Chia train/test theo thời gian (Train: 2021-2024, Test: 2025-2026)
train_mask = df['_date'] < '2025-01-01'
test_mask = df['_date'] >= '2025-01-01'

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]
idx_train = X_train.index
idx_test = X_test.index

y_pred       = model.predict(X_test)
y_pred_prob  = model.predict_proba(X_test)[:,1]
y_train_pred = model.predict(X_train)

test_acc  = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, y_train_pred)

print(f"  Tổng mẫu : {len(df):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
if 'date' in df.columns:
    print(f"  Khoảng ngày: {df['date'].min()} → {df['date'].max()}")


# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TẦNG 1 — SANITY CHECK CƠ BẢN")
print(SEP)

gap = train_acc - test_acc
print(f"\n  [1A] Train vs Test accuracy:")
print(f"       Train : {train_acc*100:.2f}%")
print(f"       Test  : {test_acc*100:.2f}%")
print(f"       Gap   : {gap*100:.2f}%")
if gap > 0.10:
    print("       ⚠️  CẢNH BÁO: Gap > 10% → khả năng OVERFIT cao")
elif gap > 0.05:
    print("       ⚡ Gap 5-10% → overfit nhẹ, chấp nhận được")
else:
    print("       ✅ Gap < 5% → không overfit đáng kể")

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"\n  [1B] So sánh với baseline:")
print(f"       Baseline accuracy : {dummy_acc*100:.2f}%")
print(f"       Model accuracy    : {test_acc*100:.2f}%")
print(f"       Cải thiện         : +{(test_acc-dummy_acc)*100:.2f}%")
if test_acc - dummy_acc < 0.03:
    print("       ⚠️  Cải thiện < 3% so với baseline → rất đáng ngờ")
else:
    print("       ✅ Model có cải thiện thực chất so với baseline")

print(f"\n  [1C] Permutation test (shuffle label 5 lần):")
shuffle_accs = []
for i in range(5):
    y_shuffle = y_test.sample(frac=1, random_state=i).values
    shuffle_accs.append(accuracy_score(y_shuffle, y_pred))
print(f"       Accuracy khi label ngẫu nhiên: {np.mean(shuffle_accs)*100:.2f}% ± {np.std(shuffle_accs)*100:.2f}%")

auc = roc_auc_score(y_test, y_pred_prob)
print(f"\n  [1D] AUC-ROC: {auc:.4f}")
if auc > 0.65:
    print("       ✅ AUC > 0.65 → model có tín hiệu rõ ràng")
elif auc > 0.55:
    print("       ⚡ AUC 0.55-0.65 → tín hiệu yếu nhưng có thực")
else:
    print("       ⚠️  AUC < 0.55 → model gần như không phân biệt được")


# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TẦNG 2 — KIỂM TRA DATA LEAKAGE")
print(SEP)

print("\n  [2A] Tương quan giữa từng feature và target:")
corr_series = X_test.corrwith(y_test.reset_index(drop=True)).abs().sort_values(ascending=False)
for feat, corr in corr_series.items():
    flag = " ⚠️  NGHI NGỜ LEAKAGE" if corr > 0.5 else (" ⚡ Cao" if corr > 0.3 else "")
    print(f"       {feat:25s}: {corr:.4f}{flag}")

print(f"\n  [2B] Shuffle từng feature — drop accuracy:")
for feat in feature_cols:
    X_test_shuf = X_test.copy()
    X_test_shuf[feat] = X_test_shuf[feat].sample(frac=1, random_state=0).values
    acc_shuf = accuracy_score(y_test, model.predict(X_test_shuf))
    drop = (test_acc - acc_shuf) * 100
    flag = " ⚠️  Leaky?" if drop < 0.5 else ""
    print(f"       {feat:25s}: acc={acc_shuf*100:.2f}%  drop={drop:+.2f}%{flag}")


# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TẦNG 3 — TIME-SERIES VALIDATION (QUAN TRỌNG NHẤT)")
print(SEP)

print(f"\n  [3A] TimeSeriesSplit 5-fold (gap de tranh overlap target):")
n_tickers = df['_ticker'].nunique()
tscv = TimeSeriesSplit(n_splits=5, gap=int(5 * n_tickers))
X_all = df[feature_cols]
y_all = df['target']
ts_accs, ts_aucs = [], []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_all)):
    X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
    y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=5, gamma=0.5,
                           reg_alpha=0.5, reg_lambda=2.0,
                           use_label_encoder=False, eval_metric='logloss',
                           random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, m.predict(X_te))
    auc_fold = roc_auc_score(y_te, m.predict_proba(X_te)[:,1])
    ts_accs.append(acc)
    ts_aucs.append(auc_fold)
    print(f"       Fold {fold+1}: Train={len(tr_idx):,}  Test={len(te_idx):,}  "
          f"Acc={acc*100:.2f}%  AUC={auc_fold:.4f}")

print(f"\n       Trung bình → Acc: {np.mean(ts_accs)*100:.2f}% ± {np.std(ts_accs)*100:.2f}%")
print(f"                    AUC: {np.mean(ts_aucs):.4f} ± {np.std(ts_aucs):.4f}")

gap_ts = test_acc - np.mean(ts_accs)
if abs(gap_ts) < 0.05:
    print(f"\n  ✅ Time-series acc tương đương chronological split")
elif gap_ts > 0.10:
    print(f"\n  ⚠️  Chronological split overestimate {gap_ts*100:.1f}%")
else:
    print(f"\n  ⚡ Chronological split cao hơn time-series {gap_ts*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TẦNG 4 — ROBUSTNESS & STABILITY")
print(SEP)

report = classification_report(y_test, y_pred, output_dict=True)
print(f"\n  [4A] Accuracy từng lớp:")
print(f"       Dự đoán GIẢM (0): precision={report['0']['precision']:.3f}  "
      f"recall={report['0']['recall']:.3f}  f1={report['0']['f1-score']:.3f}")
print(f"       Dự đoán TĂNG (1): precision={report['1']['precision']:.3f}  "
      f"recall={report['1']['recall']:.3f}  f1={report['1']['f1-score']:.3f}")

# FIX: dùng .loc với index thực thay vì .iloc
print(f"\n  [4B] Accuracy theo từng mã cổ phiếu:")
df_test = X_test.copy()
df_test['_ticker'] = ticker_series.loc[X_test.index].values  # FIX đây
df_test['y_true']  = y_test.values
df_test['y_pred']  = y_pred

ticker_accs = []
for t, grp in df_test.groupby('_ticker'):
    if len(grp) >= 20:
        ta = accuracy_score(grp['y_true'], grp['y_pred'])
        ticker_accs.append((t, ta, len(grp)))

ticker_accs.sort(key=lambda x: x[1], reverse=True)
print(f"       Top 5 mã tốt nhất:")
for t, ta, n in ticker_accs[:5]:
    print(f"         {str(t):10s}: {ta*100:.1f}%  (n={n})")
print(f"       Bottom 5 mã tệ nhất:")
for t, ta, n in ticker_accs[-5:]:
    print(f"         {str(t):10s}: {ta*100:.1f}%  (n={n})")

avg_ticker_acc = np.mean([x[1] for x in ticker_accs])
std_ticker_acc = np.std([x[1] for x in ticker_accs])
print(f"\n       Trung bình qua các mã : {avg_ticker_acc*100:.2f}%")
print(f"       Độ lệch chuẩn          : ±{std_ticker_acc*100:.2f}%")
if std_ticker_acc > 0.10:
    print(f"       ⚠️  Std > 10% → hiệu quả không đồng đều giữa các mã")
else:
    print(f"       ✅ Model ổn định qua các mã")

# Accuracy theo năm
print(f"\n  [4C] Accuracy theo từng năm:")
if 'date' in df.columns:
    df_test2 = X_test.copy()
    df_test2['y_true'] = y_test.values
    df_test2['y_pred'] = y_pred
    df_test2['year'] = df.loc[X_test.index, '_date'].dt.year
    for yr, grp in df_test2.groupby('year'):
        ya = accuracy_score(grp['y_true'], grp['y_pred'])
        print(f"         {yr}: {ya*100:.2f}%  (n={len(grp):,})")
else:
    print("       (Không tìm thấy cột date để phân tích theo năm)")


# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TÓM TẮT CUỐI CÙNG")
print(SEP)
print(f"""
  Chronological split accuracy: {test_acc*100:.2f}%
  TimeSeriesSplit accuracy    : {np.mean(ts_accs)*100:.2f}% ± {np.std(ts_accs)*100:.2f}%
  Baseline (majority class)   : {dummy_acc*100:.2f}%
  AUC-ROC                     : {auc:.4f}
  AUC-ROC (time-series avg)   : {np.mean(ts_aucs):.4f}
  Accuracy trung bình/ticker  : {avg_ticker_acc*100:.2f}% ± {std_ticker_acc*100:.2f}%
""")