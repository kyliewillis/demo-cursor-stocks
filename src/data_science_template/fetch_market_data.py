"""Script to fetch market index data (S&P 500, Dow Jones, Nasdaq) and save it to CSV.

This script downloads historical data for specified market indices using yfinance
and saves each index to a separate CSV file in the data/raw directory.
"""

import pandas as pd
# import pandas_datareader.data as web # No longer needed
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import os

# Define the indices to fetch
INDICES = {
    "sp500": "^GSPC",
    "dow": "^DJI",
    "nasdaq": "^IXIC"
}

def get_index_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical data for a given ticker symbol using yfinance.
    
    Args:
        ticker: The ticker symbol to fetch (e.g., "^GSPC").
        start_date: Start date for data (string YYYY-MM-DD).
        end_date: End date for data (string YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: DataFrame containing the index's historical data.
    """
    try:
        # Download data directly using yfinance
        # yf.pdr_override() # No longer needed
        # df = web.get_data_yahoo(ticker, start=start_date, end=end_date) # Old method
        
        index_Ticker = yf.Ticker(ticker)
        df = index_Ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to lowercase and remove spaces
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        # Ensure standard columns are present
        if 'adj_close' not in df.columns and 'close' in df.columns:
             df['adj_close'] = df['close'] # Simple fallback
        if 'volume' not in df.columns:
            df['volume'] = 0 # Add volume column if missing (e.g., for ^DJI)

        # Keep only standard columns
        standard_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[[col for col in standard_cols if col in df.columns]]

        # Convert date to date object (without time)
        if 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date']).dt.date

        return df
    except Exception as e:
        print(f"Error fetching data for {ticker} using yfinance: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])

def main():
    """Main function to fetch and save data for multiple indices."""
    # Set date range - default to last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching market index data from {start_date_str} to {end_date_str}...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    os.makedirs(data_dir, exist_ok=True)
    
    all_data_fetched = True
    for name, ticker in INDICES.items():
        print(f"\nFetching {name.upper()} ({ticker})...")
        index_data = get_index_data(ticker, start_date_str, end_date_str)
        
        if index_data.empty:
            print(f"Failed to fetch data for {name.upper()}. Skipping save.")
            all_data_fetched = False
            continue
        
        # Save to CSV
        output_file = data_dir / f"{name}_data.csv"
        index_data.to_csv(output_file, index=False)
        
        print(f"Successfully saved {len(index_data)} records for {name.upper()} to {output_file}")
        if not index_data.empty and 'date' in index_data.columns:
            print(f"Data spans from {index_data['date'].min()} to {index_data['date'].max()}")

    if all_data_fetched:
        print("\nAll index data fetched and saved successfully.")
    else:
        print("\nSome index data failed to fetch.")

if __name__ == "__main__":
    main() 