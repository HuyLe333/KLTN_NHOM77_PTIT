import mysql.connector
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os

# Connect to database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="170804",
    database="kltn_stock_db"
)

# 1. Fetch VNINDEX raw data
df_vnindex = pd.read_sql("""
    SELECT date, open, high, low, close, volume 
    FROM daily_raw_data 
    WHERE ticker = 'VNINDEX' 
    ORDER BY date DESC LIMIT 5
""", conn)
print("=========================================")
# Print VNINDEX recent values
print("Recent VN-Index values:")
print(df_vnindex.to_string(index=False))

# Calculate index returns
df_vnindex['return'] = df_vnindex['close'].pct_change(-1) # since it's ordered desc
print("\nVN-Index Daily Returns:")
for i, row in df_vnindex.iterrows():
    if i < len(df_vnindex) - 1:
        ret = (row['close'] - df_vnindex.iloc[i+1]['close']) / df_vnindex.iloc[i+1]['close']
        print(f"Date: {row['date']} | Close: {row['close']:,.2f} | Return: {ret*100:+.2f}%")

# 2. Fetch FPT features for June 1st vs June 3rd
df_fpt_feats = pd.read_sql("""
    SELECT * 
    FROM daily_normalized_data 
    WHERE ticker = 'FPT' 
    ORDER BY date DESC LIMIT 2
""", conn)

print("\n=========================================")
print("FPT Features comparison (June 3rd vs June 1st):")
if len(df_fpt_feats) >= 2:
    f_j3 = df_fpt_feats.iloc[0]
    f_j1 = df_fpt_feats.iloc[1]
    
    # Let's print some important features
    cols_to_compare = [
        'date', 'close_LogReturn', 'market_return', 'price_vs_sma50', 
        'volatility_20', 'rs', 'rm', 'PCA_Trend', 'PCA_Oscillators'
    ]
    for c in cols_to_compare:
        if c == 'date':
            print(f"Feature: {c:20} | June 1st: {str(f_j1[c]):12} | June 3rd: {str(f_j3[c]):12} | Change: N/A")
        else:
            val_j1 = f_j1[c] if f_j1[c] is not None else 0.0
            val_j3 = f_j3[c] if f_j3[c] is not None else 0.0
            print(f"Feature: {c:20} | June 1st: {val_j1:12.4f} | June 3rd: {val_j3:12.4f} | Change: {float(val_j3)-float(val_j1):+.4f}")
else:
    print("Not enough normalized feature data for FPT.")

# 3. Load model and explainer to get SHAP contributions for FPT on June 3rd
print("\n=========================================")
print("SHAP explanation for FPT on June 3rd:")
if os.path.exists("xgb_model.json") and os.path.exists("pca_predictor.pkl"):
    import xgboost as xgb
    import pickle
    
    # Load feature names
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
        
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    
    # Recreate SHAP explainer
    import shap
    explainer = shap.TreeExplainer(model)
    
    # Get June 3rd features
    if len(df_fpt_feats) > 0:
        row_j3 = df_fpt_feats.iloc[0]
        X = np.array([[float(row_j3[f]) for f in feature_cols]])
        
        # Explain
        shap_vals = explainer.shap_values(X)[0]
        sum_shap = float(np.sum(shap_vals))
        prob_up = float(model.predict_proba(X)[0][1])
        base_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        p_base = 1.0 / (1.0 + np.exp(-base_val))
        delta_p = prob_up - p_base
        
        # Build SHAP table
        shap_list = []
        for feat, val, sv in zip(feature_cols, X[0], shap_vals):
            prob_contrib = 0.0
            if abs(sum_shap) > 1e-6:
                prob_contrib = delta_p * (sv / sum_shap)
            shap_list.append({
                'feature': feat,
                'value': val,
                'shap_value': sv,
                'prob_contribution': prob_contrib
            })
        
        shap_df = pd.DataFrame(shap_list)
        shap_df['abs_impact'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_impact', ascending=False)
        
        print(f"FPT prob_up on June 3rd: {prob_up*100:.2f}% (Base probability: {p_base*100:.2f}%)")
        print("\nTop 7 feature contributions on June 3rd:")
        print(shap_df[['feature', 'value', 'shap_value', 'prob_contribution']].head(7).to_string(index=False))

conn.close()
