import mysql.connector
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

# Connect to database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# Load model
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

# Retrieve feature columns
feature_cols = [
    'price_vs_sma50', 'volatility_20', 'volume_ratio_20',
    'return_3d', 'return_5d', 'return_10d', 'return_20d',
    'sma_50_LogReturn', 'volume_LogReturn',
    'PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns',
    'atr_14', 'high_low', 'market_return', 'foreign_net'
]

# Fetch latest features (2026-06-01)
df = pd.read_sql("""
    SELECT * FROM daily_normalized_data 
    WHERE date = '2026-06-01'
""", conn)

print(f"Loaded {len(df)} tickers for 2026-06-01.")

if not df.empty:
    X = df[feature_cols]
    
    # Run predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    print("\n--- Predictions summary ---")
    print(pd.Series(preds).value_counts())
    print(f"Average probability up: {probs.mean():.4f}")
    
    # Explain predictions using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Calculate mean absolute SHAP value for each feature
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    # Calculate mean signed SHAP value (to see which way it pushes)
    mean_signed_shap = np.mean(shap_values, axis=0)
    
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_shap,
        'mean_signed_shap': mean_signed_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n--- SHAP feature contributions for 2026-06-01 ---")
    print(shap_df.to_string(index=False))
    
    # Let's inspect values of top features
    print("\n--- Mean values of features on 2026-06-01 ---")
    print(X.mean())

conn.close()
