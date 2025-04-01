"""Tests for market index analysis functionality."""

import pytest
import pandas as pd
import numpy as np
import os
from src.data_science_template.main import (
    load_data,
    process_index,
    main
)
from src.data_science_template.analyzer import MarketIndexAnalyzer
from src.data_science_template.reporter import ReportGenerator

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'date': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }
    return pd.DataFrame(data)

def test_load_data(tmp_path, sample_market_data):
    """Test loading market data from CSV file."""
    # Create test data directory
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    
    # Save sample data
    sample_market_data.to_csv(data_dir / "sp500_data.csv", index=False)
    
    # Test loading data with custom base directory
    df = load_data("SP500", base_dir=str(data_dir))
    assert df is not None
    assert len(df) == len(sample_market_data)
    assert all(col in df.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume'])

def test_load_data_missing_file():
    """Test loading data when file doesn't exist."""
    df = load_data("NONEXISTENT")
    assert df is None

def test_process_index(sample_market_data):
    """Test processing market index data."""
    result = process_index(sample_market_data, "SP500")
    assert result is not None
    insights, df = result
    assert isinstance(insights, dict)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_market_data)

def test_process_index_with_invalid_data():
    """Test processing with invalid data."""
    invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
    result = process_index(invalid_df, "SP500")
    assert result is None

def test_main(tmp_path, monkeypatch):
    """Test main function execution."""
    # Mock the data fetcher and report generator
    class MockDataFetcher:
        def fetch_and_save_all(self):
            pass
    
    class MockReportGenerator:
        def create_visualizations(self, df, insights, index_name):
            return {}
        
        def generate_html_report(self, all_data, all_insights, all_visualizations):
            return "test.html", "test.pdf"
    
    # Patch the imports
    monkeypatch.setattr("src.data_science_template.main.DataFetcher", MockDataFetcher)
    monkeypatch.setattr("src.data_science_template.main.ReportGenerator", MockReportGenerator)
    
    # Run main function
    main() 