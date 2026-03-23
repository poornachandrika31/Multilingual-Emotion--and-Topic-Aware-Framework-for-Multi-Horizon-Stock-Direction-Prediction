# Multilingual-Emotion--and-Topic-Aware-Framework-for-Multi-Horizon-Stock-Direction-Prediction

# 📊 PolySent-X: Multilingual Emotion & Topic-Aware Stock Prediction

## 🔍 Overview
PolySent-X is a novel framework for predicting stock price direction using multilingual financial news.  
It combines **topic relevance filtering** and **emotion-aware feature engineering** to improve prediction accuracy across multiple time horizons.

---

## 🎯 Problem Statement
Traditional sentiment-based stock prediction models:
- Ignore **news relevance**
- Use only **positive/negative sentiment**
- Fail to capture **real emotional signals**

👉 This project addresses these limitations.

---

## 🚀 Proposed Solution
We introduce **PolySent-X**, which includes:

- ✅ Topic Stability Filtering (removes irrelevant news)
- ✅ Multilingual Emotion Analysis (joy, fear, anger, etc.)
- ✅ Temporal Sentiment Features (momentum)
- ✅ Multi-Horizon Prediction (1D, 3D, 5D)

---

## 📂 Dataset

| Component | Details |
|----------|--------|
| Source | Google News RSS |
| Headlines | ~8000 → 3183 (after filtering) |
| Companies | AAPL, TSLA, MSFT, NVDA, META, AMZN, GOOGL |
| Languages | English, German, French, Spanish, Hindi |
| Stock Data | Yahoo Finance |

---

## ⚙️ Features Used

### 🔹 Baseline
- Headline Count

### 🔹 Topic Features
- Topic Relevance Score

### 🔹 Emotion Features
- Joy, Fear, Anger, Sadness, Surprise
- Polarity
- Emotion Intensity

### 🔹 Combined Features
- Topic + Emotion + Temporal Signals

---

## 🤖 Models Used

- Logistic Regression
- Random Forest
- Gradient Boosting

---

## 📊 Results Summary

| Horizon | Best Model | Feature | Accuracy |
|--------|-----------|--------|---------|
| 1-Day | LR | Topic + Emotion | 0.5217 |
| 3-Day | GB | Topic | 0.5275 |
| 5-Day | RF | Emotion | 0.5333 |

---

## 🔥 Key Insights

- Baseline ≈ Random (~0.5 accuracy)
- Topic features → Best for short-term
- Emotion features → Best for mid/long-term
- Market reacts with **time delay**
- Different stocks respond differently

---

## 📈 Visualizations

- Model comparison plots
- Accuracy across horizons
- Per-ticker analysis
- Emotion correlation heatmaps
- Feature importance graphs

---

📌 Contributions
Topic filtering for noise reduction
Fine-grained emotion modeling
Temporal sentiment analysis
Multi-horizon prediction framework

🔮 Future Work
Use full articles instead of headlines
Apply deep learning models (LSTM, Transformers)
Include macroeconomic data
Expand to more industries

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline
python src/baseline_pipeline.py

# Run comparison
python src/final_comparison.py
