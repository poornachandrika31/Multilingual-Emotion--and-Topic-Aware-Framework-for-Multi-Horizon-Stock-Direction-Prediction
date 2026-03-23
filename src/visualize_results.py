import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv("results/model_results.csv")

# -----------------------------
# MODEL COMPARISON GRAPH
# -----------------------------

plt.figure(figsize=(10,6))

sns.barplot(
    data=results,
    x="Experiment",
    y="Accuracy",
    hue="Model"
)

plt.title("Model Performance Comparison")
plt.xticks(rotation=20)

plt.savefig("results/model_comparison.png")

plt.show()


# -----------------------------
# EMOTION CORRELATION GRAPH
# -----------------------------

df = pd.read_csv("data/processed/daily_features.csv")

emotion_cols = [
    "weighted_joy",
    "weighted_fear",
    "weighted_anger",
    "weighted_sadness",
    "weighted_surprise",
    "weighted_polarity"
]

corr = df[emotion_cols + ["label_1d"]].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Emotion vs Stock Movement Correlation")

plt.savefig("results/emotion_correlation.png")

plt.show()


# -----------------------------
# HEADLINE VOLUME IMPACT
# -----------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["label_1d"],
    y=df["headline_count"]
)

plt.title("Headline Volume vs Stock Movement")

plt.savefig("results/headline_volume.png")

plt.show()