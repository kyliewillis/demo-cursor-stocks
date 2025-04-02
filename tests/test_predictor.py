"""Tests for the market predictor module."""
import pytest
import pandas as pd
import numpy as np
from data_science_template.predictor import (
    calculate_rsi,
    prepare_features,
    MarketPredictor
)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Generate dates
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generate prices with both upward and downward trends
    np.random.seed(42)
    t = np.linspace(0, 8*np.pi, n)  # Multiple cycles
    trend = 100 + 30 * np.sin(t) + t/10  # Sine wave with slight upward trend
    noise = np.random.normal(0, 5, n)
    prices = trend + noise
    
    # Create OHLCV data
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.01, n)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    })
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.01, n)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.01, n)
    
    return df


def test_calculate_rsi():
    """Test RSI calculation."""
    # Create a simple price series
    prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105])
    rsi = calculate_rsi(prices)
    
    # RSI should be between 0 and 100
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()
    
    # RSI should be 100 when price is always increasing
    increasing_prices = pd.Series([100, 101, 102, 103, 104, 105])
    rsi_increasing = calculate_rsi(increasing_prices)
    assert (rsi_increasing == 100).all()


def test_prepare_features(sample_market_data):
    """Test feature preparation."""
    features, target = prepare_features(sample_market_data)
    
    # Check if all expected features are present
    expected_features = {
        'SMA_20', 'SMA_50', 'SMA_200',
        'Returns_5d', 'Returns_20d', 'Returns_60d',
        'RSI', 'MACD', 'Volatility', 'Volume_Ratio'
    }
    assert set(features.columns) == expected_features
    
    # Check if target is binary
    assert target.isin([0, 1]).all()
    
    # Check if there are no NaN values
    assert not features.isna().any().any()
    assert not target.isna().any()
    
    # Test with different forward returns period
    features_60d, target_60d = prepare_features(sample_market_data, forward_returns_days=60)
    assert len(features_60d) == len(target_60d)


def test_market_predictor_initialization():
    """Test MarketPredictor initialization."""
    predictor = MarketPredictor()
    assert predictor.model is not None
    assert predictor.feature_columns is None
    assert predictor.auc_score is None


def test_market_predictor_training(sample_market_data):
    """Test MarketPredictor training."""
    predictor = MarketPredictor()
    
    # Train the model
    predictor.train(sample_market_data)
    
    # Check if feature columns are set
    assert predictor.feature_columns is not None
    assert len(predictor.feature_columns) > 0
    
    # Check if AUC score is calculated
    assert predictor.auc_score is not None
    assert 0 <= predictor.auc_score <= 1
    
    # Test with insufficient data
    small_data = sample_market_data.iloc[:50]
    with pytest.raises(ValueError):
        predictor.train(small_data)


def test_market_predictor_prediction(sample_market_data):
    """Test MarketPredictor prediction."""
    predictor = MarketPredictor()
    
    # Train the model first
    predictor.train(sample_market_data)
    
    # Test prediction with valid data
    prediction = predictor.predict(sample_market_data)
    assert 'buy_probability' in prediction
    assert 'model_confidence' in prediction
    assert 0 <= prediction['buy_probability'] <= 100
    assert 0 <= prediction['model_confidence'] <= 100
    
    # Test prediction with empty data
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        predictor.predict(empty_data)
    
    # Test prediction with missing features
    invalid_data = pd.DataFrame({'close': [100], 'volume': [1000]})
    with pytest.raises(ValueError, match="Missing required features"):
        predictor.predict(invalid_data)


def test_feature_engineering_consistency(sample_market_data):
    """Test consistency of feature engineering."""
    # Prepare features twice and compare
    features1, target1 = prepare_features(sample_market_data)
    features2, target2 = prepare_features(sample_market_data)
    
    # Check if features are identical
    pd.testing.assert_frame_equal(features1, features2)
    
    # Check if targets are identical
    pd.testing.assert_series_equal(target1, target2)
    
    # Check if technical indicators are calculated correctly
    sma_20 = sample_market_data['close'].rolling(window=20).mean()
    pd.testing.assert_series_equal(
        features1['SMA_20'],
        sma_20[features1.index],
        check_names=False
    ) 