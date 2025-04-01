"""Tests for the main module."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import shutil
from unittest.mock import patch, MagicMock
from data_science_template.main import load_data, process_index, main

class TestMain(unittest.TestCase):
    """Test cases for main module functions."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary test directory
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.test_df = pd.DataFrame({
            'date': dates,
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'adj_close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Create raw data directory
        self.raw_dir = self.test_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        
        # Save test data
        self.test_df.to_csv(self.raw_dir / "sp500_data.csv", index=False)

    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_data(self):
        """Test loading data from CSV file."""
        # Test successful load
        df = load_data("SP500")
        self.assertIsNotNone(df)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue('date' in df.columns)
        self.assertTrue('close' in df.columns)
        
        # Test file not found
        df = load_data("NONEXISTENT")
        self.assertIsNone(df)

    def test_process_index(self):
        """Test processing market data for a single index."""
        # Test successful processing
        result = process_index(self.test_df, "SP500")
        self.assertIsNotNone(result)
        insights, df = result
        
        # Check insights
        self.assertTrue('name' in insights)
        self.assertTrue('latest_close' in insights)
        self.assertTrue('predictions' in insights)
        
        # Check DataFrame
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue('ma20' in df.columns)
        self.assertTrue('ma50' in df.columns)
        self.assertTrue('rsi' in df.columns)
        
        # Test with invalid data
        invalid_df = pd.DataFrame({'wrong_column': [1]})
        result = process_index(invalid_df, "SP500")
        self.assertIsNone(result)

    @patch('data_science_template.main.DataFetcher')
    @patch('data_science_template.main.ReportGenerator')
    def test_main(self, mock_report_generator, mock_data_fetcher):
        """Test main function execution."""
        # Mock DataFetcher
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_and_save_all.return_value = None
        mock_data_fetcher.return_value = mock_fetcher
        
        # Mock ReportGenerator
        mock_reporter = MagicMock()
        mock_reporter.output_dir = str(self.test_dir)
        mock_reporter.create_visualizations.return_value = {
            'price_ma': 'test',
            'rsi': 'test',
            'volatility': 'test'
        }
        mock_report_generator.return_value = mock_reporter
        
        # Create test data files
        for index in ['SP500', 'DOW', 'NASDAQ']:
            self.test_df.to_csv(self.raw_dir / f"{index.lower()}_data.csv", index=False)
        
        # Run main function
        main()
        
        # Verify DataFetcher was called
        mock_fetcher.fetch_and_save_all.assert_called_once()
        
        # Verify ReportGenerator methods were called
        self.assertTrue(mock_reporter.create_visualizations.called)
        mock_reporter.generate_html_report.assert_called_once()

    def test_main_no_data(self):
        """Test main function with no data available."""
        # Remove test data files
        for file in self.raw_dir.glob('*.csv'):
            file.unlink()
        
        # Run main function
        main()
        
        # Verify no report was generated
        self.assertFalse((self.test_dir / "market_indices_report.html").exists())

if __name__ == '__main__':
    unittest.main() 