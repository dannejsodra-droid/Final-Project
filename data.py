# --- STEP 1: Data Collection and Exploratory Data Analysis ---

import yfinance as yf
import pandas as pd


def load_data():
    # Download Volvo B stock data
    data = yf.download("VOLV-B.ST", start="2020-01-01", auto_adjust=True)

    # Fix column format if yfinance returns multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


# --- STEP 2: Feature Selection and Cleaning ---

def prepare_data():
    data = load_data()

    # Create explanatory variables
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["Volatility"] = data["Return"].rolling(10).std()

    # Create target variable: 1 = price goes up tomorrow, 0 = price goes down
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # Remove missing values
    data = data.dropna()

    return data


if __name__ == "__main__":

    data = prepare_data()

    # --- STEP 1: Exploratory Data Analysis output ---

    print("\nFirst rows:")
    print(data.head())

    print("\nLast rows:")
    print(data.tail())

    print("\nDataset information:")
    print(data.info())

    print("\nSummary statistics:")
    print(data.describe())

    print("\nTarget distribution:")
    print(data["Target"].value_counts(normalize=True))