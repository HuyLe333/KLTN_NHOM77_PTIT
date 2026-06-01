import pandas as pd
df = pd.read_excel("data2.xlsx")
high_rows = df[df['high_low'] > 1530]
print(f"Number of rows with high_low > 1530: {len(high_rows)}")
if len(high_rows) > 0:
    print(high_rows[['ticker', 'date', 'high_low', 'market_return']].head(20).to_string())
