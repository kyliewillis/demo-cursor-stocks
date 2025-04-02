"""Tests for the market predictor module."""
import pytest
import pandas as pd
import numpy as np
from data_science_template.predictor import (
    calculate_rsi,
    MarketPredictor
)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Generate dates
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate prices with both upward and downward trends
    # Using sine wave with slight upward trend and noise
    t = np.arange(len(dates))
    base_price = 100
    trend = 0.0001 * t  # Slight upward trend
    noise = np.random.normal(0, 0.5, len(dates))
    prices = base_price + trend + np.sin(2 * np.pi * t / 252) + noise
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.normal(0, 0.1, len(dates)),
        'high': prices + np.abs(np.random.normal(0, 0.2, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 0.2, len(dates))),
        'close': prices + np.random.normal(0, 0.1, len(dates)),
        'volume': np.random.normal(10000, 1000, len(dates))
    })
    
    # Ensure OHLC relationships are maintained
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, 0.1, len(dates)))
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, 0.1, len(dates)))
    
    return df


def test_calculate_rsi():
    """Test RSI calculation."""
    # Create sample price data
    prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105])
    
    # Calculate RSI
    rsi = calculate_rsi(prices, periods=2)
    
    # Check basic properties
    assert len(rsi) == len(prices)
    assert all(0 <= x <= 100 for x in rsi)
    assert not rsi.isna().any()


def test_prepare_features(sample_market_data):
    """Test feature preparation."""
    # Create predictor instance
    predictor = MarketPredictor()
    
    # Prepare features
    features, target = predictor.prepare_features(sample_market_data)
    
    # Check if all expected features are present
    expected_features = [
        'ma20', 'ma50', 'ma200',
        'Returns_5d', 'Returns_20d', 'Returns_60d',
        'RSI', 'MACD', 'Volatility',
        'Volume_Ratio'
    ]
    assert all(feature in features.columns for feature in expected_features)
    
    # Check if target is binary
    assert target.isin([0, 1]).all()
    
    # Check if there are no NaN values
    assert not features.isna().any().any()
    assert not target.isna().any()


def test_market_predictor_initialization():
    """Test MarketPredictor initialization."""
    predictor = MarketPredictor()
    
    # Check if model is initialized with correct parameters
    assert predictor.model is not None
    assert predictor.model.get_params()['objective'] == 'binary:logistic'
    assert predictor.model.get_params()['eval_metric'] == 'auc'
    assert predictor.model.get_params()['learning_rate'] == 0.01
    assert predictor.model.get_params()['max_depth'] == 3
    assert predictor.model.get_params()['subsample'] == 0.8
    assert predictor.model.get_params()['colsample_bytree'] == 0.8


def test_market_predictor_training(sample_market_data):
    """Test MarketPredictor training."""
    predictor = MarketPredictor()
    
    # Train the model
    predictor.train(sample_market_data)
    
    # Check if model is trained
    assert predictor.feature_columns is not None
    assert predictor.auc_score is not None
    assert predictor.feature_importance is not None
    assert predictor.cv_scores is not None
    
    # Check feature importance
    assert len(predictor.feature_importance) == len(predictor.feature_columns)
    assert all(0 <= importance <= 1 for importance in predictor.feature_importance['importance'])
    
    # Check cross-validation scores
    assert len(predictor.cv_scores) == 5  # 5-fold cross-validation
    assert all(0 <= score <= 1 for score in predictor.cv_scores)


def test_market_predictor_prediction(sample_market_data):
    """Test MarketPredictor prediction."""
    predictor = MarketPredictor()
    
    # Train the model first
    predictor.train(sample_market_data)
    
    # Test prediction with valid data
    prediction = predictor.predict(sample_market_data)
    assert 'buy_probability' in prediction
    assert 'confidence' in prediction
    assert 0 <= prediction['buy_probability'] <= 100
    assert 0 <= prediction['confidence'] <= 100
    
    # Test prediction with invalid data
    with pytest.raises(ValueError):
        predictor.predict(pd.DataFrame())
    
    # Test prediction with missing columns
    invalid_data = pd.DataFrame({'date': [1, 2, 3]})
    with pytest.raises(ValueError):
        predictor.predict(invalid_data)


def test_feature_engineering_consistency(sample_market_data):
    """Test consistency of feature engineering between training and prediction."""
    predictor = MarketPredictor()
    
    # Train the model
    predictor.train(sample_market_data)
    
    # Get features from training
    train_features, _ = predictor.prepare_features(sample_market_data)
    
    # Get features from prediction
    pred_features, _ = predictor.prepare_features(sample_market_data)
    
    # Check if features are consistent
    assert list(train_features.columns) == list(pred_features.columns)
    assert train_features.shape == pred_features.shape
    assert np.allclose(train_features, pred_features) 