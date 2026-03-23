import pandas as pd
from transformers import pipeline
from tqdm import tqdm

print("Loading dataset...")

df = pd.read_csv("data/processed/news_with_topics.csv")

# -----------------------------
# LOAD XLM-R EMOTION MODEL
# -----------------------------

print("Loading XLM-RoBERTa emotion model...")

emotion_model = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

tqdm.pandas()

# Candidate emotion labels
emotion_labels = ["joy", "fear", "anger", "sadness", "surprise", "neutral"]

# -----------------------------
# EMOTION EXTRACTION FUNCTION
# -----------------------------

def extract_emotions(text):

    result = emotion_model(
        text,
        candidate_labels=emotion_labels,
        multi_label=True
    )

    emotions = dict(zip(result["labels"], result["scores"]))

    joy = emotions.get("joy", 0)
    fear = emotions.get("fear", 0)
    anger = emotions.get("anger", 0)
    sadness = emotions.get("sadness", 0)
    surprise = emotions.get("surprise", 0)
    neutral = emotions.get("neutral", 0)

    # -----------------------------
    # POLARITY CALCULATION
    # -----------------------------

    positive_score = joy + surprise
    negative_score = fear + anger + sadness

    polarity = positive_score - negative_score

    # -----------------------------
    # SENTIMENT LABEL
    # -----------------------------

    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return pd.Series([
        joy,
        fear,
        anger,
        sadness,
        surprise,
        neutral,
        polarity,
        sentiment
    ])


print("Extracting emotions from headlines...")

df[[
    "joy",
    "fear",
    "anger",
    "sadness",
    "surprise",
    "neutral",
    "polarity",
    "sentiment"
]] = df["headline"].progress_apply(extract_emotions)

# -----------------------------
# SAVE DATA
# -----------------------------

df.to_csv("data/processed/news_with_emotions_xlm.csv", index=False)

print("\nSaved: data/processed/news_with_emotions_xlm.csv")

print("\nExample rows:")
print(df[[
    "headline",
    "joy",
    "fear",
    "polarity",
    "sentiment"
]].head())