"""
Fit PCA Mapper: Trains a Multi-Output Random Forest Regressor
to reconstruct PCA_Trend, PCA_Oscillators, PCA_MACD, and PCA_ShortReturns
from the 12 base technical indicators in data2.xlsx.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

def main():
    print("=" * 60)
    print("  TRAINING PCA RECONSTRUCTION MAPPER")
    print("=" * 60)
    
    print("\n[1/4] Loading data2.xlsx...")
    t0 = time.time()
    df = pd.read_excel('data2.xlsx')
    print(f"    OK Loaded in {time.time() - t0:.2f} seconds. Shape: {df.shape}")
    
    non_pca_cols = [
        'price_vs_sma50', 'volatility_20', 'volume_ratio_20', 
        'return_3d', 'return_5d', 'return_10d', 'return_20d', 
        'sma_50_LogReturn', 'volume_LogReturn', 'atr_14', 
        'high_low', 'market_return'
    ]
    pca_cols = ['PCA_Trend', 'PCA_Oscillators', 'PCA_MACD', 'PCA_ShortReturns']
    
    # Drop rows containing NaNs in either inputs or targets
    df_clean = df[non_pca_cols + pca_cols].dropna()
    print(f"    OK Cleaned dataset shape: {df_clean.shape}")
    
    X = df_clean[non_pca_cols]
    y = df_clean[pca_cols]
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n[2/4] Fitting Multi-Output RandomForestRegressor...")
    t1 = time.time()
    # Using ExtraTrees / RandomForest with n_estimators=50, max_depth=15
    mapper = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    mapper.fit(X_train, y_train)
    print(f"    OK Model fit in {time.time() - t1:.2f} seconds.")
    
    print("\n[3/4] Evaluating R^2 scores on Test set...")
    y_pred = mapper.predict(X_test)
    scores = {}
    for idx, col in enumerate(pca_cols):
        r2 = r2_score(y_test[col], y_pred[:, idx])
        scores[col] = r2
        print(f"      - {col:18s}: R^2 = {r2:.4f}")
        
    print("\n[4/4] Saving model to pca_predictor.pkl...")
    model_data = {
        'model': mapper,
        'features': non_pca_cols,
        'targets': pca_cols,
        'r2_scores': scores
    }
    with open('pca_predictor.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("    OK Saved: pca_predictor.pkl")
    
    print("=" * 60)
    print("  PCA RECONSTRUCTION MAPPER TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
