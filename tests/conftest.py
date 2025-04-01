"""Shared test fixtures for weather analysis project."""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    dates = pd.date_range(start='2025-01-01', end='2025-01-02', freq='H')
    return pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(0, 5, len(dates)),
        'humidity': np.random.normal(50, 10, len(dates)),
        'pressure': np.random.normal(1013, 5, len(dates)),
        'wind_speed': np.random.normal(10, 2, len(dates)),
        'rainfall': np.random.exponential(1, len(dates))
    })

@pytest.fixture
def sample_weather_data_with_outliers(sample_weather_data):
    """Create sample weather data with outliers for testing."""
    df = sample_weather_data.copy()
    # Add some outliers
    df.loc[5, 'temperature'] = 100  # Unrealistic temperature
    df.loc[10, 'humidity'] = 150    # Unrealistic humidity
    df.loc[15, 'pressure'] = 500    # Unrealistic pressure
    return df

@pytest.fixture
def sample_weather_data_with_missing(sample_weather_data):
    """Create sample weather data with missing values for testing."""
    df = sample_weather_data.copy()
    # Add some missing values
    df.loc[5, 'temperature'] = np.nan
    df.loc[10, 'humidity'] = np.nan
    df.loc[15, 'pressure'] = np.nan
    return df 