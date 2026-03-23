import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print("Loading dataset...")

df = pd.read_csv("data/processed/daily_features.csv")

df = df.dropna()

# ------------------------------------------------
# FEATURE SETS
# ------------------------------------------------

baseline_features = [
    "headline_count"
]

topic_features = [
    "topic_score"
]

emotion_features = [

    "weighted_joy",
    "weighted_fear",
    "weighted_anger",
    "weighted_sadness",
    "weighted_surprise",
    "weighted_polarity",

    "emotion_std",

    "weighted_joy_prev",
    "weighted_fear_prev",
    "weighted_anger_prev",
    "weighted_sadness_prev",
    "weighted_surprise_prev",

    "dominant_emotion"
]

combined_features = baseline_features + topic_features + emotion_features

feature_sets = {
    "Baseline": baseline_features,
    "Topic Only": topic_features,
    "Emotion Only": emotion_features,
    "Topic + Emotion": combined_features
}

targets = [
    "label_1d",
    "label_3d",
    "label_5d"
]

# ------------------------------------------------
# MODELS
# ------------------------------------------------

models = {

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

results = []

# ------------------------------------------------
# LOOP THROUGH TICKERS
# ------------------------------------------------

tickers = df["ticker"].unique()

for ticker in tickers:

    print("\n====================================")
    print("Ticker:", ticker)
    print("====================================")

    ticker_df = df[df["ticker"] == ticker]

    if len(ticker_df) < 50:
        print("Skipping ticker (not enough data)")
        continue

    for target in targets:

        print("\nPredicting:", target)

        train_df, test_df = train_test_split(
            ticker_df,
            test_size=0.2,
            shuffle=False
        )

        for exp_name, features in feature_sets.items():

            print("\nExperiment:", exp_name)

            X_train = train_df[features]
            y_train = train_df[target]

            X_test = test_df[features]
            y_test = test_df[target]

            for model_name, model in models.items():

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds)
                bal = balanced_accuracy_score(y_test, preds)

                results.append({

                    "Ticker": ticker,
                    "Target": target,
                    "Experiment": exp_name,
                    "Model": model_name,
                    "Accuracy": round(acc,4),
                    "F1": round(f1,4),
                    "Balanced Accuracy": round(bal,4)

                })

                print(
                    f"{model_name} | Acc: {acc:.4f} | F1: {f1:.4f} | Bal: {bal:.4f}"
                )

# ------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_results_per_ticker.csv",
    index=False
)

print("\nSaved results/model_results_per_ticker.csv")

# ------------------------------------------------
# CREATE SUMMARY TABLE
# ------------------------------------------------

summary = results_df.pivot_table(
    index=["Ticker","Experiment"],
    columns="Model",
    values="Accuracy"
)

summary.to_csv(
    "results/per_ticker_accuracy_table.csv"
)

print("\nSaved results/per_ticker_accuracy_table.csv")