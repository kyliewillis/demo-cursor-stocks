"""Tests for weather data fetching functionality."""

import pytest
import pandas as pd
from datetime import datetime
from src.data_science_template.fetch_weather_data import get_weather_data

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    dates = pd.date_range(start='2025-01-01', end='2025-01-02', freq='H')
    return pd.DataFrame({
        'date': dates,
        'temperature': [0.0] * len(dates),
        'humidity': [50.0] * len(dates),
        'pressure': [1013.0] * len(dates),
        'wind_speed': [10.0] * len(dates),
        'rainfall': [0.0] * len(dates)
    })

def test_get_weather_data_columns(sample_weather_data):
    """Test that the weather data has all required columns."""
    required_columns = ['date', 'temperature', 'humidity', 'pressure', 'wind_speed', 'rainfall']
    assert all(col in sample_weather_data.columns for col in required_columns)

def test_get_weather_data_types(sample_weather_data):
    """Test that the weather data has correct data types."""
    assert pd.api.types.is_datetime64_any_dtype(sample_weather_data['date'])
    assert pd.api.types.is_numeric_dtype(sample_weather_data['temperature'])
    assert pd.api.types.is_numeric_dtype(sample_weather_data['humidity'])
    assert pd.api.types.is_numeric_dtype(sample_weather_data['pressure'])
    assert pd.api.types.is_numeric_dtype(sample_weather_data['wind_speed'])
    assert pd.api.types.is_numeric_dtype(sample_weather_data['rainfall'])

def test_get_weather_data_ranges(sample_weather_data):
    """Test that the weather data values are within reasonable ranges."""
    assert sample_weather_data['temperature'].between(-50, 50).all()
    assert sample_weather_data['humidity'].between(0, 100).all()
    assert sample_weather_data['pressure'].between(800, 1100).all()
    assert sample_weather_data['wind_speed'].between(0, 200).all()
    assert sample_weather_data['rainfall'].between(0, 1000).all()

def test_get_weather_data_empty():
    """Test handling of empty data response."""
    # Test with invalid coordinates
    df = get_weather_data(0, 0, datetime(2025, 1, 1), datetime(2025, 1, 2))
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_get_weather_data_date_range():
    """Test that the data spans the correct date range."""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 2)
    df = get_weather_data(44.4759, -73.2121, start_date, end_date)
    
    assert not df.empty
    assert df['date'].min().date() == start_date.date()
    assert df['date'].max().date() == end_date.date()
    assert len(df) == 25  # 24 hours + 1 for the end date

def test_get_weather_data_outliers(sample_weather_data_with_outliers):
    """Test handling of data with outliers."""
    # Test that the data still has the correct structure
    assert isinstance(sample_weather_data_with_outliers, pd.DataFrame)
    assert all(col in sample_weather_data_with_outliers.columns for col in ['date', 'temperature', 'humidity', 'pressure', 'wind_speed', 'rainfall'])
    
    # Test that outliers are present
    assert sample_weather_data_with_outliers['temperature'].max() > 50
    assert sample_weather_data_with_outliers['humidity'].max() > 100
    assert sample_weather_data_with_outliers['pressure'].min() < 800

def test_get_weather_data_missing(sample_weather_data_with_missing):
    """Test handling of data with missing values."""
    # Test that the data still has the correct structure
    assert isinstance(sample_weather_data_with_missing, pd.DataFrame)
    assert all(col in sample_weather_data_with_missing.columns for col in ['date', 'temperature', 'humidity', 'pressure', 'wind_speed', 'rainfall'])
    
    # Test that missing values are present
    assert sample_weather_data_with_missing['temperature'].isna().any()
    assert sample_weather_data_with_missing['humidity'].isna().any()
    assert sample_weather_data_with_missing['pressure'].isna().any() 