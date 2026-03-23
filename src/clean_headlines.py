import pandas as pd

print("Loading headlines...")

df = pd.read_csv("data/raw/headlines_raw.csv")

# -----------------------------
# TIMESTAMP PROCESSING
# -----------------------------

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

# remove rows with missing values
df = df.dropna(subset=["headline", "published_at"])

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

df = df.drop_duplicates(subset=["headline", "link"])

# -----------------------------
# HEADLINE CLEANING
# -----------------------------

df["headline"] = df["headline"].str.strip()

# remove very short headlines
df = df[df["headline"].str.len() > 20]

# -----------------------------
# CREATE DATE COLUMN
# -----------------------------

df["date"] = df["published_at"].dt.normalize()

# -----------------------------
# OPTIONAL: REMOVE VERY OLD NEWS
# -----------------------------

df = df[df["published_at"].dt.year >= 2010]

# -----------------------------
# SORT DATA
# -----------------------------

df = df.sort_values("published_at")

# reset index
df = df.reset_index(drop=True)

# -----------------------------
# SAVE CLEAN DATA
# -----------------------------

df.to_csv("data/processed/headlines_clean.csv", index=False)

print("\nSaved: data/processed/headlines_clean.csv")
print("Total cleaned headlines:", len(df))

print("\nTicker distribution:")
print(df["ticker"].value_counts())

print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nDate range:")
print(df["published_at"].min(), "→", df["published_at"].max())