"""Test script for S&P 500 data fetching dependencies."""

try:
    import pandas as pd
    import pandas_datareader
    import yfinance as yf
    print("All dependencies are successfully imported!")
    print("pandas:", pd.__version__)
    print("pandas_datareader:", pandas_datareader.__version__)
    print("yfinance:", yf.__version__)
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install the required packages with: pip install -r requirements.txt") 