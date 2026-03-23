import pandas as pd
from transformers import pipeline
from tqdm import tqdm

print("Loading dataset...")

df = pd.read_csv("data/processed/news_with_labels.csv")

# -----------------------------
# LOAD MULTILINGUAL TOPIC MODEL
# -----------------------------

print("Loading XLM-RoBERTa topic classifier...")

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

candidate_labels = [
    "stock market news",
    "company earnings",
    "financial markets",
    "business news",
    "technology news",
    "sports news",
    "entertainment news"
]

tqdm.pandas()

# -----------------------------
# TOPIC SCORING
# -----------------------------

def topic_score(text):

    result = classifier(text, candidate_labels)

    return max(result["scores"])

print("Computing topic scores...")

df["topic_score"] = df["headline"].progress_apply(topic_score)

# -----------------------------
# FILTER LOW RELEVANCE
# -----------------------------

threshold = 0.60

df = df[df["topic_score"] > threshold]

# -----------------------------
# SAVE
# -----------------------------

df.to_csv("data/processed/news_with_topics.csv", index=False)

print("\nSaved: data/processed/news_with_topics.csv")

print("\nRemaining rows:", len(df))