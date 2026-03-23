import pandas as pd
import yfinance as yf

print("Downloading stock prices...")

TICKERS = [
    "AAPL",
    "TSLA",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL"
]

START_DATE = "2010-01-01"

all_data = []

for ticker in TICKERS:

    print(f"Downloading {ticker}")

    df = yf.download(
        ticker,
        start=START_DATE,
        progress=False
    )

    # reset index so date becomes column
    df = df.reset_index()

    # flatten columns if multi-index appears
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # add ticker column
    df["ticker"] = ticker

    # keep only available columns
    df = df[[
        "ticker",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]]

    all_data.append(df)

# combine vertically
prices = pd.concat(all_data, ignore_index=True)

# rename columns
prices = prices.rename(columns={
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

# convert date
prices["date"] = pd.to_datetime(prices["date"])

# sort
prices = prices.sort_values(["ticker", "date"])

prices.to_csv("data/raw/stock_prices.csv", index=False)

print("\nSaved: data/raw/stock_prices.csv")
print("Total rows:", len(prices))

print("\nTicker distribution:")
print(prices["ticker"].value_counts())

print("\nDate range:")
print(prices["date"].min(), "→", prices["date"].max())