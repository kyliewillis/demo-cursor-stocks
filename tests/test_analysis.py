"""Tests for weather data analysis functionality."""

import pytest
import pandas as pd
import numpy as np
from src.data_science_template.main import (
    load_data,
    calculate_insights,
    create_visualizations,
    generate_html_report
)

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

def test_load_data(tmp_path, sample_weather_data):
    """Test data loading functionality."""
    # Save sample data to temporary CSV
    data_path = tmp_path / "weather_data.csv"
    sample_weather_data.to_csv(data_path, index=False)
    
    # Test loading
    df = load_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_weather_data)
    assert all(col in df.columns for col in sample_weather_data.columns)

def test_load_data_missing_file():
    """Test handling of missing data file."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")

def test_calculate_insights(sample_weather_data):
    """Test insight calculation functionality."""
    insights = calculate_insights(sample_weather_data)
    
    # Check required insights
    required_insights = [
        'temperature_stats',
        'humidity_stats',
        'pressure_stats',
        'wind_speed_stats',
        'rainfall_stats',
        'correlations'
    ]
    assert all(key in insights for key in required_insights)
    
    # Check statistics format
    for stat in ['temperature_stats', 'humidity_stats', 'pressure_stats', 
                 'wind_speed_stats', 'rainfall_stats']:
        assert isinstance(insights[stat], dict)
        assert all(key in insights[stat] for key in ['mean', 'std', 'min', 'max'])
    
    # Check correlations
    assert isinstance(insights['correlations'], pd.DataFrame)
    assert insights['correlations'].shape == (5, 5)  # 5 weather parameters

def test_calculate_insights_with_outliers(sample_weather_data_with_outliers):
    """Test insight calculation with outliers."""
    insights = calculate_insights(sample_weather_data_with_outliers)
    
    # Check that insights are still calculated
    assert all(key in insights for key in ['temperature_stats', 'humidity_stats', 'pressure_stats'])
    
    # Check that outliers are reflected in statistics
    assert insights['temperature_stats']['max'] > 50
    assert insights['humidity_stats']['max'] > 100
    assert insights['pressure_stats']['min'] < 800

def test_calculate_insights_with_missing(sample_weather_data_with_missing):
    """Test insight calculation with missing values."""
    insights = calculate_insights(sample_weather_data_with_missing)
    
    # Check that insights are still calculated
    assert all(key in insights for key in ['temperature_stats', 'humidity_stats', 'pressure_stats'])
    
    # Check that statistics are calculated correctly with missing values
    assert not np.isnan(insights['temperature_stats']['mean'])
    assert not np.isnan(insights['humidity_stats']['mean'])
    assert not np.isnan(insights['pressure_stats']['mean'])

def test_create_visualizations(sample_weather_data):
    """Test visualization creation functionality."""
    insights = calculate_insights(sample_weather_data)
    visualizations = create_visualizations(sample_weather_data, insights)
    
    # Check required visualizations
    required_viz = ['temperature_plot', 'correlation_heatmap']
    assert all(key in visualizations for key in required_viz)
    
    # Check visualization types
    assert isinstance(visualizations['temperature_plot'], str)  # Base64 encoded image
    assert isinstance(visualizations['correlation_heatmap'], str)  # Base64 encoded image

def test_create_visualizations_with_outliers(sample_weather_data_with_outliers):
    """Test visualization creation with outliers."""
    insights = calculate_insights(sample_weather_data_with_outliers)
    visualizations = create_visualizations(sample_weather_data_with_outliers, insights)
    
    # Check that visualizations are still created
    assert all(key in visualizations for key in ['temperature_plot', 'correlation_heatmap'])
    assert isinstance(visualizations['temperature_plot'], str)
    assert isinstance(visualizations['correlation_heatmap'], str)

def test_generate_html_report(tmp_path, sample_weather_data):
    """Test HTML report generation."""
    insights = calculate_insights(sample_weather_data)
    visualizations = create_visualizations(sample_weather_data, insights)
    
    # Generate report
    report_path = tmp_path / "weather_report.html"
    generate_html_report(sample_weather_data, insights, visualizations, report_path)
    
    # Check report file
    assert report_path.exists()
    assert report_path.stat().st_size > 0
    
    # Check report content
    report_content = report_path.read_text()
    assert "Weather Analysis Report" in report_content
    assert "Temperature" in report_content
    assert "Humidity" in report_content
    assert "Pressure" in report_content
    assert "Wind Speed" in report_content
    assert "Rainfall" in report_content

def test_generate_html_report_with_outliers(tmp_path, sample_weather_data_with_outliers):
    """Test HTML report generation with outliers."""
    insights = calculate_insights(sample_weather_data_with_outliers)
    visualizations = create_visualizations(sample_weather_data_with_outliers, insights)
    
    # Generate report
    report_path = tmp_path / "weather_report.html"
    generate_html_report(sample_weather_data_with_outliers, insights, visualizations, report_path)
    
    # Check report file
    assert report_path.exists()
    assert report_path.stat().st_size > 0
    
    # Check report content
    report_content = report_path.read_text()
    assert "Weather Analysis Report" in report_content
    assert "Temperature" in report_content
    assert "Humidity" in report_content
    assert "Pressure" in report_content 