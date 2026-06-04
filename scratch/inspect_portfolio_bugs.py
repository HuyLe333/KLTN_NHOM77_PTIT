import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

def inspect_bugs():
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    
    # Let's test with a set of 5 tickers
    test_tickers = ['FPT', 'HPG', 'VCB', 'VNM', 'SSI']
    tickers_str = ", ".join([f"'{t}'" for t in test_tickers])
    
    # 1. Check dates in model_training_data vs daily_raw_data
    with engine.connect() as conn:
        res_train = conn.execute(text(f"SELECT MIN(date), MAX(date), COUNT(*) FROM model_training_data WHERE ticker IN ({tickers_str})")).fetchone()
        res_raw = conn.execute(text(f"SELECT MIN(date), MAX(date), COUNT(*) FROM daily_raw_data WHERE ticker IN ({tickers_str})")).fetchone()
        
    print("model_training_data date range:", res_train[0], "to", res_train[1], "Total rows:", res_train[2])
    print("daily_raw_data date range:", res_raw[0], "to", res_raw[1], "Total rows:", res_raw[2])
    
    # 2. Simulate rounding with different capital levels
    # Let's assume weights are: FPT=0.3, HPG=0.25, VCB=0.2, VNM=0.15, SSI=0.1
    weights = {'FPT': 0.3, 'HPG': 0.25, 'VCB': 0.2, 'VNM': 0.15, 'SSI': 0.1}
    prices = {'FPT': 130000.0, 'HPG': 28000.0, 'VCB': 92000.0, 'VNM': 68000.0, 'SSI': 35000.0}
    
    for capital in [10000000, 30000000, 100000000]:
        print(f"\n--- Simulation for Capital: {capital:,.0f} VND ---")
        total_allocated = 0.0
        holdings = []
        for ticker, w in weights.items():
            price = prices[ticker]
            target_value = capital * w
            qty = round(target_value / price / 100) * 100
            actual_value = qty * price
            holdings.append((ticker, w, qty, actual_value))
            total_allocated += actual_value
        
        cash_left = capital - total_allocated
        print(f"Total Allocated: {total_allocated:,.0f} VND")
        print(f"Cash Left: {cash_left:,.0f} VND")
        for ticker, w, qty, actual_value in holdings:
            act_w = (actual_value / total_allocated * 100) if total_allocated > 0 else 0
            print(f"  {ticker}: Target={w*100:.1f}%, Actual={act_w:.1f}%, Qty={qty}, Value={actual_value:,.0f} VND")

if __name__ == '__main__':
    inspect_bugs()
