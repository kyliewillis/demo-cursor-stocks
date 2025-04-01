"""Tests for the DataFetcher class."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import shutil
from unittest.mock import patch, MagicMock
from data_science_template.data_fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    """Test cases for DataFetcher class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary test directory
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.test_df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 10, len(dates)),
            'High': np.random.normal(105, 10, len(dates)),
            'Low': np.random.normal(95, 10, len(dates)),
            'Close': np.random.normal(100, 10, len(dates)),
            'Adj Close': np.random.normal(100, 10, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Initialize DataFetcher with test directory
        self.fetcher = DataFetcher(data_dir=self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test initialization of DataFetcher."""
        # Test default initialization
        self.assertEqual(self.fetcher.indices, DataFetcher.DEFAULT_INDICES)
        self.assertEqual(self.fetcher.data_dir, self.test_dir)
        
        # Test custom indices
        custom_indices = {"test": {"name": "Test Index", "ticker": "TEST"}}
        fetcher = DataFetcher(indices=custom_indices, data_dir=self.test_dir)
        self.assertEqual(fetcher.indices, custom_indices)

    @patch('yfinance.Ticker')
    def test_fetch_index_data(self, mock_yf):
        """Test fetching index data."""
        # Mock yfinance response
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self.test_df
        mock_yf.return_value = mock_ticker
        
        # Test successful fetch
        df = self.fetcher.fetch_index_data(
            "^TEST",
            "2023-01-01",
            "2023-12-31"
        )
        
        # Check DataFrame structure
        self.assertFalse(df.empty)
        self.assertTrue(all(col in df.columns for col in [
            'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'
        ]))
        self.assertTrue(isinstance(df['date'].iloc[0], datetime))
        
        # Test empty response
        mock_ticker.history.return_value = pd.DataFrame()
        df = self.fetcher.fetch_index_data(
            "^TEST",
            "2023-01-01",
            "2023-12-31"
        )
        self.assertTrue(df.empty)

    def test_save_data(self):
        """Test saving data to CSV."""
        # Test successful save
        success = self.fetcher.save_data(self.test_df, "test_data")
        self.assertTrue(success)
        self.assertTrue((self.test_dir / "test_data.csv").exists())
        
        # Test empty DataFrame
        success = self.fetcher.save_data(pd.DataFrame(), "empty_data")
        self.assertFalse(success)
        self.assertFalse((self.test_dir / "empty_data.csv").exists())

    @patch('yfinance.Ticker')
    def test_fetch_and_save_all(self, mock_yf):
        """Test fetching and saving all indices."""
        # Mock yfinance response
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self.test_df
        mock_yf.return_value = mock_ticker
        
        # Test successful fetch and save
        fetched_data = self.fetcher.fetch_and_save_all(years=1)
        
        # Check results
        self.assertEqual(len(fetched_data), len(self.fetcher.indices))
        for key in self.fetcher.indices:
            self.assertTrue(key in fetched_data)
            self.assertTrue((self.test_dir / f"{key}_data.csv").exists())
        
        # Test with empty response
        mock_ticker.history.return_value = pd.DataFrame()
        fetched_data = self.fetcher.fetch_and_save_all(years=1)
        self.assertEqual(len(fetched_data), 0)

    def test_data_standardization(self):
        """Test data standardization in fetch_index_data."""
        # Create test data with different column names
        test_df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=5),
            'Open': [100] * 5,
            'High': [105] * 5,
            'Low': [95] * 5,
            'Close': [100] * 5,
            'Volume': [1000000] * 5
        })
        
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = test_df
            mock_yf.return_value = mock_ticker
            
            df = self.fetcher.fetch_index_data(
                "^TEST",
                "2023-01-01",
                "2023-01-05"
            )
            
            # Check column standardization
            self.assertTrue(all(col in df.columns for col in [
                'date', 'open', 'high', 'low', 'close', 'volume'
            ]))
            self.assertTrue(isinstance(df['date'].iloc[0], datetime))

if __name__ == '__main__':
    unittest.main() 