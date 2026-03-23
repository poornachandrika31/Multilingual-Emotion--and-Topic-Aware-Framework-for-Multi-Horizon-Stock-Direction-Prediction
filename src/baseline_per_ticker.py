import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

news_path = os.path.join(BASE_DIR, "..", "data", "processed", "headlines_clean.csv")
price_path = os.path.join(BASE_DIR, "..", "data", "processed", "stock_prices_with_labels.csv")

# =========================
# LOAD DATA
# =========================
news_df = pd.read_csv(news_path)
price_df = pd.read_csv(price_path)

# =========================
# CLEAN
# =========================
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")

price_df.columns = [c.lower() for c in price_df.columns]

# =========================
# BASELINE FEATURE
# =========================
baseline_df = (
    news_df.groupby(["ticker", "date"])
    .size()
    .reset_index(name="headline_count")
)

# =========================
# MERGE
# =========================
df = baseline_df.merge(price_df, on=["ticker", "date"], how="inner")

# drop missing labels
df = df.dropna(subset=["label_1d", "label_3d", "label_5d"])

print("Merged shape:", df.shape)

# =========================
# MODELS
# =========================
MODELS = {
    "LR": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42)
}

targets = ["label_1d", "label_3d", "label_5d"]

# =========================
# PER-TICKER TRAINING
# =========================
results = []

tickers = df["ticker"].unique()   # ✅ FIXED

for ticker in tickers:
    sub = df[df["ticker"] == ticker]

    if len(sub) < 50:
        continue  # skip small data

    for target in targets:
        X = sub[["headline_count"]]
        y = sub[target].astype(int)

        split = int(len(sub) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        for name, model in MODELS.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            results.append({
                "ticker": ticker,
                "target": target,
                "model": name,
                "accuracy": round(accuracy_score(y_test, preds), 4)
            })

# =========================
# RESULTS
# =========================
res_df = pd.DataFrame(results)

print("\n===== PER TICKER RESULTS =====")
print(res_df)

# =========================
# SAVE
# =========================
results_path = os.path.join(BASE_DIR, "..", "results")
os.makedirs(results_path, exist_ok=True)

res_df.to_csv(os.path.join(results_path, "baseline_per_ticker.csv"), index=False)

# =========================
# PLOTS
# =========================

# 🔹 Plot 1 — Per ticker (1D only)
plt.figure(figsize=(12,6))
sns.barplot(data=res_df[res_df["target"] == "label_1d"],
            x="ticker", y="accuracy", hue="model")
plt.title("Baseline Performance per Ticker (1-Day)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "baseline_ticker_1d.png"))
plt.show()

sns.barplot(
    data=res_df[res_df["target"] == "label_3d"],
    x="ticker", y="accuracy", hue="model"
)
plt.title("Baseline Performance per Ticker (3-Day)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "baseline_ticker_3d.png"))
plt.show()

# 🔹 5-DAY
plt.figure(figsize=(12,6))
sns.barplot(
    data=res_df[res_df["target"] == "label_5d"],
    x="ticker", y="accuracy", hue="model"
)
plt.title("Baseline Performance per Ticker (5-Day)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "baseline_ticker_5d.png"))
plt.show()
# 🔹 Plot 2 — Horizon comparison
plt.figure(figsize=(10,6))
sns.barplot(data=res_df, x="target", y="accuracy", hue="model")
plt.title("Baseline Accuracy Across Horizons")
plt.tight_layout()
plt.savefig(os.path.join(results_path, "baseline_horizon.png"))
plt.show()