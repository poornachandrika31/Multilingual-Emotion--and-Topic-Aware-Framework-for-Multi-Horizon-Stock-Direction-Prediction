import pandas as pd

print("Merging headlines with stock labels...")

# -----------------------------
# LOAD DATASETS
# -----------------------------

news = pd.read_csv("data/processed/headlines_clean.csv")
prices = pd.read_csv("data/processed/stock_prices_with_labels.csv")

# ensure datetime format
news["date"] = pd.to_datetime(news["date"])
prices["date"] = pd.to_datetime(prices["date"])

# -----------------------------
# MERGE DATA
# -----------------------------

merged = pd.merge(
    news,
    prices,
    on=["ticker", "date"],
    how="inner"
)

# -----------------------------
# SORT DATA
# -----------------------------

merged = merged.sort_values(["ticker", "date"])

merged = merged.reset_index(drop=True)

# -----------------------------
# SAVE FILE
# -----------------------------

merged.to_csv("data/processed/news_with_labels.csv", index=False)

print("\nSaved: data/processed/news_with_labels.csv")

print("\nTotal rows:", len(merged))

print("\nTicker distribution:")
print(merged["ticker"].value_counts())

print("\nExample rows:")
print(merged.head())