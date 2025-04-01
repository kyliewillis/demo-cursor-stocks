"""Script to fetch S&P 500 data and save it to CSV.

This script downloads historical S&P 500 data using yfinance
and saves it to the data/raw directory as sp500_data.csv.
"""

import pandas as pd
# import pandas_datareader.data as web # No longer needed
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import os

def get_sp500_data(start_date, end_date):
    """Fetch S&P 500 data for the specified date range using yfinance.
    
    Args:
        start_date: Start date for data (datetime or string YYYY-MM-DD)
        end_date: End date for data (datetime or string YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: DataFrame containing S&P 500 historical data
    """
    # Yahoo Finance uses ^GSPC as the ticker for S&P 500
    ticker = "^GSPC"
    try:
        # Download data directly using yfinance
        # yf.pdr_override() # No longer needed
        # df = web.get_data_yahoo(ticker, start=start_date, end=end_date) # Old method
        
        sp500 = yf.Ticker(ticker)
        df = sp500.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to lowercase and remove spaces
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        # Ensure 'adj close' column exists, calculate if necessary (yfinance might name it differently or not include it)
        if 'adj_close' not in df.columns and 'close' in df.columns:
             df['adj_close'] = df['close'] # Or implement proper calculation if needed

        # Keep only standard columns if needed
        standard_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[[col for col in standard_cols if col in df.columns]]

        return df
    except Exception as e:
        print(f"Error fetching S&P 500 data using yfinance: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])

def main():
    """Main function to fetch and save S&P 500 data."""
    # Set date range - default to last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    # Format dates as strings for yfinance history function
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching S&P 500 data from {start_date_str} to {end_date_str} using yfinance...")
    
    # Fetch S&P 500 data
    sp500_data = get_sp500_data(start_date_str, end_date_str)
    
    if sp500_data.empty:
        print("Failed to fetch S&P 500 data.")
        return
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_file = data_dir / "sp500_data.csv"
    # Ensure 'date' column is in the correct format before saving
    if 'date' in sp500_data.columns:
        sp500_data['date'] = pd.to_datetime(sp500_data['date']).dt.date

    sp500_data.to_csv(output_file, index=False)
    
    print(f"Successfully saved {len(sp500_data)} records to {output_file}")
    if not sp500_data.empty and 'date' in sp500_data.columns:
        print(f"Data spans from {sp500_data['date'].min()} to {sp500_data['date'].max()}")

if __name__ == "__main__":
    main() 