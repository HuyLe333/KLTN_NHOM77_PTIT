"""
Export script: Xuất dữ liệu đã xử lý ra file Excel
- Giữ lại ticker, date để dễ đọc
- Thêm cột target (0/1) và target_label (Tang/Giam)
- Loại bỏ các cột không cần + 2 cột leaky (rsi_14_LogReturn, return_1d)
- Loại bỏ hàng null
- Tách thành 2 sheet: All_Data, Summary_Stats
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 55)
print("  XUAT DU LIEU DA XU LY RA EXCEL")
print("=" * 55)

# ── 1. Load ──────────────────────────────────────────────
print("\n[1/4] Dang tai data2.xlsx ...")
df_raw = pd.read_excel('data2.xlsx')
print(f"    Raw: {df_raw.shape[0]:,} dong x {df_raw.shape[1]} cot")

# ── 2. Xu ly ─────────────────────────────────────────────
print("\n[2/4] Xu ly du lieu ...")

# Cac cot bi loai
drop_cols = ['id', 'ticker_id', 'Unnamed: 23', 'Unnamed: 24',
             'rsi_14_LogReturn', 'return_1d']

df = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])

# Drop null
before = len(df)
df = df.dropna()
after  = len(df)
print(f"    Xoa null: {before - after:,} hang -> con lai {after:,} hang")

# Sap xep cot: ticker, date truoc, roi features, roi target cuoi
feature_cols = [
    'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
    'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'sma_50_LogReturn', 'volume_LogReturn',
    'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
    'atr_14', 'high_low', 'market_return'
]
feature_cols = [c for c in feature_cols if c in df.columns]

# Them cot target
df['target']       = (df['close_LogReturn'] > 0).astype(int)
df['target_label'] = df['target'].map({1: 'Tang', 0: 'Giam'})

# Sap xep lai cot
base_cols    = [c for c in ['ticker', 'date', 'close_LogReturn'] if c in df.columns]
ordered_cols = base_cols + feature_cols + ['target', 'target_label']
df = df[ordered_cols]

# Sap xep theo ticker va date
if 'ticker' in df.columns and 'date' in df.columns:
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

print(f"    So features giu lai: {len(feature_cols)}")
print(f"    Cot bi loai (leaky): rsi_14_LogReturn, return_1d")
print(f"    Cot bi loai (khac) : id, ticker_id, Unnamed:23, Unnamed:24")
print(f"    Shape cuoi: {df.shape[0]:,} x {df.shape[1]}")

class_counts = df['target_label'].value_counts()
print(f"    Phan bo: Tang={class_counts.get('Tang',0):,}  Giam={class_counts.get('Giam',0):,}")

# ── 3. Tao Summary Stats ──────────────────────────────────
print("\n[3/4] Tao sheet thong ke ...")

# Per-ticker summary
ticker_summary = df.groupby('ticker').agg(
    So_phien       =('ticker',      'count'),
    So_phien_Tang  =('target',      'sum'),
    Ty_le_Tang_pct =('target',      lambda x: round(x.mean()*100, 2)),
    close_LR_mean  =('close_LogReturn', 'mean'),
    close_LR_std   =('close_LogReturn', 'std'),
    **{f + '_median': (f, 'median') for f in feature_cols}
).reset_index()
ticker_summary['So_phien_Giam'] = ticker_summary['So_phien'] - ticker_summary['So_phien_Tang']
ticker_summary['Ty_le_Giam_pct'] = 100 - ticker_summary['Ty_le_Tang_pct']

# Feature stats tong the
feat_stats = df[feature_cols + ['close_LogReturn']].describe().T.reset_index()
feat_stats.columns = ['Feature', 'Count', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max']
feat_stats = feat_stats.round(6)

# ── 4. Xuat Excel ─────────────────────────────────────────
print("\n[4/4] Ghi file Excel ...")
out_file = 'data2_processed.xlsx'

with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
    # Sheet 1: Toan bo du lieu
    df.to_excel(writer, sheet_name='Du_Lieu_Da_Xu_Ly', index=False)

    # Sheet 2: Thong ke per-ticker
    ticker_summary.to_excel(writer, sheet_name='Thong_Ke_Ticker', index=False)

    # Sheet 3: Feature statistics
    feat_stats.to_excel(writer, sheet_name='Feature_Statistics', index=False)

    # Sheet 4: Mo ta cac buoc xu ly
    info_rows = [
        ['=== THONG TIN XU LY DU LIEU ===', ''],
        ['Thoi gian xuat', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['File goc', 'data2.xlsx'],
        ['File xuat', out_file],
        ['', ''],
        ['=== KET QUA ===', ''],
        ['Tong so dong goc', df_raw.shape[0]],
        ['Tong so dong sau xu ly', len(df)],
        ['So dong da xoa (null)', df_raw.shape[0] - len(df)],
        ['So cot goc', df_raw.shape[1]],
        ['So cot sau xu ly', df.shape[1]],
        ['So features su dung', len(feature_cols)],
        ['', ''],
        ['=== COT DA LOAI BO ===', ''],
        ['id',               'Khong can thiet'],
        ['ticker_id',        'Khong can thiet'],
        ['Unnamed: 23',      'Cot rong'],
        ['Unnamed: 24',      'Cot rong'],
        ['rsi_14_LogReturn', 'DATA LEAKAGE - tuong quan truc tiep voi target'],
        ['return_1d',        'DATA LEAKAGE - xap xi bang close_LogReturn'],
        ['', ''],
        ['=== COT THEM MOI ===', ''],
        ['target',       '1 = Tang (close_LogReturn > 0), 0 = Giam'],
        ['target_label', 'Tang / Giam (nhan chu)'],
        ['', ''],
        ['=== FEATURES SU DUNG (16 COT) ===', ''],
    ] + [[f, ''] for f in feature_cols]

    info_df = pd.DataFrame(info_rows, columns=['Muc', 'Mo_ta'])
    info_df.to_excel(writer, sheet_name='Huong_Dan', index=False)

print(f"    OK Da xuat: {out_file}")
print(f"\n    Cac sheet:")
print(f"      1. Du_Lieu_Da_Xu_Ly    - {len(df):,} dong x {df.shape[1]} cot")
print(f"      2. Thong_Ke_Ticker     - {len(ticker_summary)} ma co phieu")
print(f"      3. Feature_Statistics  - Thong ke cac feature")
print(f"      4. Huong_Dan           - Mo ta qua trinh xu ly")
print(f"\n{'='*55}")
print(f"  HOAN TAT! Mo file: {out_file}")
print(f"{'='*55}")
