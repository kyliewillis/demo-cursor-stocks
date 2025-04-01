"""Module for fetching market index data using yfinance."""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Dict, Optional

class DataFetcher:
    """Fetches historical data for market indices."""

    DEFAULT_INDICES = {
        "sp500": {"name": "S&P 500", "ticker": "^GSPC"},
        "dow": {"name": "Dow Jones", "ticker": "^DJI"},
        "nasdaq": {"name": "Nasdaq", "ticker": "^IXIC"}
    }
    RAW_DATA_DIR = Path("data/raw")

    def __init__(self, indices: Optional[Dict] = None, data_dir: Optional[Path] = None):
        """Initialize the DataFetcher.

        Args:
            indices (Optional[Dict]): Dictionary defining indices to fetch. 
                                       Defaults to DEFAULT_INDICES.
            data_dir (Optional[Path]): Directory to save raw data. Defaults to RAW_DATA_DIR.
        """
        self.indices = indices if indices is not None else self.DEFAULT_INDICES
        self.data_dir = data_dir if data_dir is not None else self.RAW_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_index_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a given ticker symbol using yfinance.
        
        Args:
            ticker: The ticker symbol to fetch (e.g., "^GSPC").
            start_date: Start date for data (string YYYY-MM-DD).
            end_date: End date for data (string YYYY-MM-DD).
            
        Returns:
            pd.DataFrame: DataFrame containing the index's historical data.
        """
        print(f"Fetching {ticker} from {start_date} to {end_date}...")
        try:
            index_ticker = yf.Ticker(ticker)
            df = index_ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"  No data found for {ticker}.")
                return pd.DataFrame()

            df = df.reset_index()
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df['adj_close'] = df['close']
            if 'volume' not in df.columns:
                df['volume'] = 0

            standard_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[[col for col in standard_cols if col in df.columns]]

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.date # Ensure date only, no timezone

            print(f"  Successfully fetched {len(df)} records for {ticker}.")
            return df
        except Exception as e:
            print(f"  Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: str) -> bool:
        """Save DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): Data to save.
            filename (str): Base filename (without extension).
            
        Returns:
            bool: True if save was successful, False otherwise.
        """
        if df.empty:
            print(f"  Skipping save for {filename}: DataFrame is empty.")
            return False
            
        output_file = self.data_dir / f"{filename}.csv"
        try:
            df.to_csv(output_file, index=False)
            print(f"  Data saved to {output_file}")
            return True
        except Exception as e:
            print(f"  Error saving data to {output_file}: {e}")
            return False

    def fetch_and_save_all(self, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch and save data for all configured indices.

        Args:
            years (int): Number of years of historical data to fetch.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping index keys to fetched dataframes.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        print(f"--- Starting Data Fetch ({start_date_str} to {end_date_str}) ---")
        fetched_data = {}
        all_successful = True
        for key, info in self.indices.items():
            ticker = info['ticker']
            df = self.fetch_index_data(ticker, start_date_str, end_date_str)
            if not df.empty:
                if self.save_data(df, f"{key}_data"):
                    fetched_data[key] = df
                else:
                    all_successful = False # Failed to save
            else:
                 all_successful = False # Failed to fetch
        
        status = "successfully" if all_successful else "with some errors"
        print(f"--- Data Fetch Completed ({status}) ---")
        return fetched_data

# Allow running this module directly to fetch data
if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.fetch_and_save_all() 