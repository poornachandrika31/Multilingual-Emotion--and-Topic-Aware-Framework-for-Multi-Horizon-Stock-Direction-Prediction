import pandas as pd
import os

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

news_path = os.path.join(BASE_DIR, "..", "data", "processed", "headlines_clean.csv")
price_path = os.path.join(BASE_DIR, "..", "data", "processed", "stock_prices_with_labels.csv")

# =========================
# LOAD DATA
# =========================
news_df = pd.read_csv(news_path)
price_df = pd.read_csv(price_path)

# clean
news_df = news_df.dropna(subset=["headline", "date", "ticker"])
news_df["date"] = pd.to_datetime(news_df["date"])

price_df["date"] = pd.to_datetime(price_df["date"])
price_df.columns = [c.lower() for c in price_df.columns]

# =========================
# CREATE BASELINE FEATURE
# =========================
baseline_df = (
    news_df.groupby(["ticker", "date"])
    .size()
    .reset_index(name="headline_count")
)

# =========================
# MERGE WITH LABELS
# =========================
baseline_df = baseline_df.merge(
    price_df[["ticker", "date", "label_1d", "label_3d", "label_5d"]],
    on=["ticker", "date"],
    how="inner"
)

# =========================
# SORT (important for time-series)
# =========================
baseline_df = baseline_df.sort_values(["ticker", "date"])

# =========================
# SAVE FILE
# =========================
save_path = os.path.join(BASE_DIR, "..", "data", "processed", "baseline_dataset.csv")

baseline_df.to_csv(save_path, index=False)

print("✅ Baseline dataset saved at:", save_path)
print("Shape:", baseline_df.shape)
print(baseline_df.head())