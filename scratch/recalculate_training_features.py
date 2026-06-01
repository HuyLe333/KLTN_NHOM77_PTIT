import pandas as pd
import numpy as np
import mysql.connector

def main():
    # Connect to DB to get daily_raw_data
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="170804",
        database="kltn_stock_db"
    )

    print("Loading daily raw prices...")
    df_raw = pd.read_sql("SELECT ticker, date, open, high, low, close, volume FROM daily_raw_data ORDER BY ticker, date", conn)
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    print("Calculating normalized ATR-14 and high-low spread ratio...")
    df_list = []
    for ticker, group in df_raw.groupby('ticker'):
        group = group.sort_values('date').copy()
        close = group['close']
        high = group['high']
        low = group['low']
        
        # TR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # ATR 14
        atr_14_raw = tr.rolling(14).mean()
        # Normalized ATR 14: ATR / close
        group['atr_14_new'] = atr_14_raw / close
        # High-low spread ratio: (high - low) / close
        group['high_low_new'] = (high - low) / close
        
        df_list.append(group[['ticker', 'date', 'atr_14_new', 'high_low_new']])

    df_calc = pd.concat(df_list, ignore_index=True)

    print("Loading data2.xlsx...")
    df_excel = pd.read_excel("data2.xlsx")
    df_excel['date'] = pd.to_datetime(df_excel['date'])

    print("Merging recalculated features...")
    df_excel = pd.merge(df_excel, df_calc, on=['ticker', 'date'], how='left')

    # Replace old columns with new calculated values
    df_excel['atr_14'] = df_excel['atr_14_new']
    df_excel['high_low'] = df_excel['high_low_new']

    # Drop the temp columns
    df_excel = df_excel.drop(columns=['atr_14_new', 'high_low_new'])

    print("Saving updated data2.xlsx...")
    df_excel.to_excel("data2.xlsx", index=False)
    print("Successfully updated data2.xlsx!")
    conn.close()

if __name__ == '__main__':
    main()
