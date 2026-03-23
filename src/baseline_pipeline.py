import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 0. PATH SETUP (VERY IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

news_path = os.path.join(BASE_DIR, "..", "data", "processed", "headlines_clean.csv")
price_path = os.path.join(BASE_DIR, "..", "data", "processed", "stock_prices_with_labels.csv")

# =========================
# 1. LOAD DATA
# =========================
news_df = pd.read_csv(news_path)
price_df = pd.read_csv(price_path)

# =========================
# 2. CLEAN DATA
# =========================
news_df = news_df.dropna(subset=["headline", "date", "ticker"])
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")

price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")

# normalize column names (IMPORTANT if Close vs close issue)
price_df.columns = [c.lower() for c in price_df.columns]

# =========================
# 3. BASELINE FEATURE (headline count)
# =========================
baseline_df = (
    news_df.groupby(["ticker", "date"])
    .size()
    .reset_index(name="headline_count")
)

# =========================
# 4. MERGE WITH LABELS
# =========================
df = baseline_df.merge(price_df, on=["ticker", "date"], how="inner")

# drop rows where labels missing
df = df.dropna(subset=["label_1d", "label_3d", "label_5d"])

print("Merged shape:", df.shape)

# =========================
# 5. MODELS
# =========================
MODELS = {
    "LR": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42)
}

targets = ["label_1d", "label_3d", "label_5d"]

all_results = []

# =========================
# 6. TRAIN + EVAL
# =========================
for target in targets:
    X = df[["headline_count"]]
    y = df[target].astype(int)

    # time-based split
    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        all_results.append({
            "type": "Baseline",
            "target": target,
            "model": name,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "balanced_acc": round(balanced_accuracy_score(y_test, preds), 4)
        })

# =========================
# 7. SAVE RESULTS
# =========================
results_df = pd.DataFrame(all_results)

print("\n===== BASELINE RESULTS =====")
print(results_df)

# create results folder if not exists
results_path = os.path.join(BASE_DIR, "..", "results")
os.makedirs(results_path, exist_ok=True)

results_df.to_csv(os.path.join(results_path, "baseline_results.csv"), index=False)

# =========================
# 8. PLOT
# =========================
plt.figure(figsize=(10,6))
sns.barplot(data=results_df, x="target", y="accuracy", hue="model")
plt.title("Baseline Model Performance")
plt.tight_layout()

plt.savefig(os.path.join(results_path, "baseline_plot.png"))
plt.show()