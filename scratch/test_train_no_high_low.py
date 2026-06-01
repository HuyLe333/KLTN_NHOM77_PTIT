import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import json

print("Loading dataset...")
from sqlalchemy import create_engine
engine = create_engine("mysql+mysqlconnector://root:170804@localhost/kltn_stock_db")
df = pd.read_sql("SELECT * FROM model_training_data", engine)
df['_ticker'] = df['ticker']
df['_date'] = pd.to_datetime(df['date'])

# Sắp xếp toàn bộ DataFrame theo dòng thời gian tăng dần
df = df.sort_values('_date').reset_index(drop=True)

# Features (removed high_low)
feature_cols = [
    'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
    'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'sma_50_LogReturn', 'volume_LogReturn',
    'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
    'atr_14', 'market_return', 'foreign_net'
]

X = df[feature_cols]
y = df['target']

train_mask = df['_date'] < '2025-01-01'
test_mask = df['_date'] >= '2025-01-01'

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

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

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=50
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== EVALUATION WITHOUT high_low ===")
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test AUC-ROC:  {auc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Let's count predictions in test set
print("\nPrediction distribution in test set:")
print(pd.Series(y_pred).value_counts())
