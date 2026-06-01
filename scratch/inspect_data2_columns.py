import pandas as pd

# Load a small part of data2.xlsx
print("Loading data2.xlsx...")
df = pd.read_excel("data2.xlsx", nrows=10)
print("\nColumn names in data2.xlsx:")
print(list(df.columns))

# Let's inspect some rows
cols_of_interest = ['ticker', 'date', 'high_low', 'market_return']
cols_present = [c for c in cols_of_interest if c in df.columns]
print("\nFirst 10 rows:")
print(df[cols_present])

# Check min/max/mean of high_low in data2.xlsx
df_all = pd.read_excel("data2.xlsx", usecols=['high_low'])
print("\nStats of high_low in data2.xlsx:")
print(df_all.describe())
