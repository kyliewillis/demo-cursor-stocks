"""Market prediction module using XGBoost."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb


def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI).
    
    Args:
        prices: Series of price data
        periods: Number of periods for RSI calculation
        
    Returns:
        Series containing RSI values
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Special case: all increasing prices should have RSI = 100
    if (delta.fillna(0) >= 0).all():
        return pd.Series(100, index=prices.index)
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.inf)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge cases
    rsi = rsi.replace([np.inf, -np.inf], [100, 0])
    rsi = rsi.fillna(50)  # Fill NaN with neutral value
    
    return rsi


def prepare_features(df: pd.DataFrame, forward_returns_days: int = 252) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features for the model.
    
    Args:
        df: DataFrame with OHLCV data
        forward_returns_days: Number of days to look forward for returns calculation
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Technical indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Price momentum
    df['Returns_5d'] = df['close'].pct_change(5)
    df['Returns_20d'] = df['close'].pct_change(20)
    df['Returns_60d'] = df['close'].pct_change(60)
    
    # Technical indicators
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    
    # Volatility
    df['Volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # Volume indicators
    df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
    
    # Target variable - now using absolute returns threshold
    df['Forward_Returns'] = df['close'].shift(-forward_returns_days) / df['close'] - 1
    
    # Create binary target (1 for positive returns, 0 for negative)
    # Using a threshold of 0 to ensure consistent signals across indices
    df['Target'] = (df['Forward_Returns'] > 0).astype(int)
    
    # Select features
    feature_columns = [
        'SMA_20', 'SMA_50', 'SMA_200',
        'Returns_5d', 'Returns_20d', 'Returns_60d',
        'RSI', 'MACD', 'Volatility',
        'Volume_Ratio'
    ]
    
    # Drop rows with NaN values in features or target
    df = df.dropna(subset=feature_columns + ['Target'])
    
    if df.empty:
        raise ValueError("No valid data points after feature preparation")
    
    return df[feature_columns], df['Target']


class MarketPredictor:
    """Market prediction model using XGBoost."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            base_score=0.5,  # Set base_score between 0 and 1
            random_state=42
        )
        self.feature_columns = None
        self.auc_score = None
    
    def train(self, df: pd.DataFrame) -> None:
        """Train the model on historical data.
        
        Args:
            df: DataFrame with OHLCV data
        """
        if df.empty:
            raise ValueError("Training data is empty")
            
        X, y = prepare_features(df)
        self.feature_columns = X.columns
        
        if len(X) < 100:  # Need minimum amount of data
            raise ValueError("Insufficient data for training")
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            raise ValueError("Training data must contain both positive and negative examples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate AUC score
        y_pred = self.model.predict_proba(X_test)[:, 1]
        self.auc_score = roc_auc_score(y_test, y_pred)
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Make predictions on new data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with buy probability and model confidence
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        if not all(col in df.columns for col in ['close', 'volume']):
            raise ValueError("Missing required features: close and volume columns are required")
            
        try:
            X, _ = prepare_features(df)
        except ValueError as e:
            if str(e) == "No valid data points after feature preparation":
                raise ValueError("Missing required features: insufficient data for feature calculation")
            raise
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Make predictions
        probabilities = self.model.predict_proba(X)
        buy_probability = probabilities[:, 1].mean() * 100
        
        return {
            'buy_probability': buy_probability,
            'model_confidence': self.auc_score * 100 if self.auc_score is not None else 0.0
        } 