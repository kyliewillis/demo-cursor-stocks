"""Market prediction module using XGBoost."""
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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


class MarketPredictor:
    """Market prediction model using XGBoost."""
    
    def __init__(self):
        """Initialize the predictor with default parameters."""
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.feature_columns = None
        self.feature_importance = None
        self.cv_scores = None
        self.auc_score = 0.0
    
    def prepare_features(self, df: pd.DataFrame, forward_returns_days: int = 252) -> Tuple[pd.DataFrame, pd.Series]:
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
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
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
            'ma20', 'ma50', 'ma200',
            'Returns_5d', 'Returns_20d', 'Returns_60d',
            'RSI', 'MACD', 'Volatility',
            'Volume_Ratio'
        ]
        
        # Drop rows with NaN values in features or target
        df = df.dropna(subset=feature_columns + ['Target'])
        
        if len(df) == 0:
            raise ValueError("No valid data points after feature preparation")
            
        return df[feature_columns], df['Target']

    def train(self, df: pd.DataFrame) -> None:
        """Train the model on historical data with cross-validation.
        
        Args:
            df: DataFrame with OHLCV data
        """
        if df.empty:
            raise ValueError("Training data is empty")
            
        X, y = self.prepare_features(df)
        self.feature_columns = X.columns
        
        if len(X) < 100:  # Need minimum amount of data
            raise ValueError("Insufficient data for training")
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            raise ValueError("Training data must contain both positive and negative examples")
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Perform cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=kfold, scoring='roc_auc'
        )
        
        # Train model with early stopping
        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate AUC score on test set
        y_pred = self.model.predict_proba(X_test)[:, 1]
        self.auc_score = roc_auc_score(y_test, y_pred)
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Make predictions on new data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with buy probability and AUC-based confidence
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        if not all(col in df.columns for col in ['close', 'volume']):
            raise ValueError("Missing required features: close and volume columns are required")
            
        try:
            X, _ = self.prepare_features(df)
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
        
        # Use cross-validation AUC as confidence
        confidence = np.mean(self.cv_scores) * 100 if self.cv_scores is not None else 0.0
        
        return {
            'buy_probability': buy_probability,
            'confidence': confidence
        } 