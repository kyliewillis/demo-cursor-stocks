"""Module for analyzing market index data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class MarketIndexAnalyzer:
    """Analyzes a DataFrame of market index data to calculate insights."""

    # Class constants
    MOVING_AVERAGE_PERIODS = [20, 50, 200]
    RSI_PERIOD = 14
    VOLATILITY_WINDOW = 20
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, df: pd.DataFrame, index_name: str):
        """Initialize the analyzer.
        
        Args:
            df: DataFrame containing market data with required columns ('date', 'close')
            index_name: Name of the market index being analyzed
        """
        self._validate_input(df)
        self.df = df.copy()
        self.index_name = index_name
        self.insights = {"name": index_name}
        self.df['date'] = pd.to_datetime(self.df['date'])

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df.empty or 'date' not in df.columns or 'close' not in df.columns:
            raise ValueError("Input DataFrame is empty or missing required columns ('date', 'close').")

    def calculate_all_insights(self) -> Dict[str, Any]:
        """Calculate all insights for the index.
        
        Returns:
            Dictionary containing all calculated insights
        """
        try:
            if len(self.df) < 2:
                print(f"Warning: Insufficient data points ({len(self.df)}) for full analysis of {self.index_name}.")
                self._populate_default_metrics()
                return self.insights
                
            self._calculate_all_metrics()
            return self.insights
            
        except Exception as e:
            print(f"Error calculating insights for {self.index_name}: {str(e)}")
            self._populate_default_metrics()
            return self.insights

    def _calculate_all_metrics(self) -> None:
        """Calculate all metrics in the correct order."""
        # Basic info
        self._populate_basic_info()
        
        # Returns and volatility
        self._calculate_returns()
        self._calculate_volatility()
        
        # Technical indicators
        self._calculate_moving_averages()
        self._calculate_rsi()
        
        # Predictions and returns
        self._calculate_predictions()
        self._calculate_period_returns()
        
        # Final statistics
        self._calculate_final_stats()

    def _populate_basic_info(self) -> None:
        """Populate basic info like latest price, date, high, low, volume."""
        self.insights.update({
            "latest_close": float(self.df['close'].iloc[-1]),
            "latest_date": self.df['date'].iloc[-1].strftime('%Y-%m-%d'),
            "current_price": float(self.df['close'].iloc[-1]),
            "year_high": float(self.df['high'].max() if 'high' in self.df.columns else self.df['close'].iloc[-1]),
            "year_low": float(self.df['low'].min() if 'low' in self.df.columns else self.df['close'].iloc[-1]),
            "latest_volume": float(self.df['volume'].iloc[-1] if 'volume' in self.df.columns else 0),
            "avg_volume": float(self.df['volume'].mean() if 'volume' in self.df.columns else 0)
        })

    def _calculate_returns(self) -> None:
        """Calculate daily percentage returns."""
        try:
            self.df['daily_return'] = self.df['close'].pct_change() * 100
            self.df['daily_return'] = self.df['daily_return'].fillna(0)
            self.insights.update({
                "max_daily_gain": float(self.df['daily_return'].max()),
                "max_daily_loss": float(self.df['daily_return'].min()),
                "avg_daily_return": float(self.df['daily_return'].mean())
            })
        except Exception as e:
            print(f"Error calculating returns: {str(e)}")
            self._set_default_returns()

    def _calculate_volatility(self) -> None:
        """Calculate rolling volatility."""
        try:
            if 'daily_return' not in self.df.columns:
                self._calculate_returns()
            
            self.df['volatility'] = self.df['daily_return'].rolling(
                window=self.VOLATILITY_WINDOW, 
                min_periods=1
            ).std()
            self.df['volatility'] = self.df['volatility'].fillna(0)
            
            # Annualize volatility
            self.df['volatility'] = self.df['volatility'] * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            
            self.insights.update({
                "current_volatility": float(self.df['volatility'].iloc[-1]),
                "avg_volatility": float(self.df['volatility'].mean())
            })
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            self._set_default_volatility()

    def _calculate_moving_averages(self) -> None:
        """Calculate moving averages."""
        try:
            for period in self.MOVING_AVERAGE_PERIODS:
                col_name = f'ma{period}'
                self.df[col_name] = self.df['close'].rolling(
                    window=period, 
                    min_periods=1
                ).mean()
                self.df[col_name] = self.df[col_name].ffill().bfill()
                
            self.insights["moving_averages"] = {
                f"ma{period}": float(self.df[f'ma{period}'].iloc[-1])
                for period in self.MOVING_AVERAGE_PERIODS
            }
        except Exception as e:
            print(f"Error calculating moving averages: {str(e)}")
            self._set_default_moving_averages()

    def _calculate_rsi(self) -> None:
        """Calculate Relative Strength Index (RSI)."""
        try:
            delta = self.df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(
                window=self.RSI_PERIOD, 
                min_periods=1
            ).mean()
            loss = -delta.where(delta < 0, 0).rolling(
                window=self.RSI_PERIOD, 
                min_periods=1
            ).mean()
            rs = gain / loss.replace(0, 1e-6)
            self.df['rsi'] = 100 - (100 / (1 + rs))
            self.df['rsi'] = self.df['rsi'].fillna(50)
            self.insights["current_rsi"] = float(self.df['rsi'].iloc[-1])
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            self.insights["current_rsi"] = 50.0

    def _calculate_predictions(self) -> None:
        """Calculate prediction signals based on technical indicators."""
        try:
            if not all(col in self.df.columns for col in ['ma20', 'ma50', 'ma200']):
                self._calculate_moving_averages()
                
            ma_signal, ma_description = self._calculate_ma_signal()
            rsi_signal, rsi_description = self._calculate_rsi_signal()
            combined_signal = (ma_signal * 0.6 + rsi_signal * 0.4)
            
            recommendation = self._get_recommendation(combined_signal)
            signal_strength = self._get_signal_strength(combined_signal)
            
            self.insights["predictions"] = {
                "ma_signal": ma_signal,
                "ma_description": ma_description,
                "rsi_signal": rsi_signal,
                "rsi_description": rsi_description,
                "combined_signal": combined_signal,
                "recommendation": recommendation,
                "signal_strength": signal_strength
            }
        except Exception as e:
            print(f"Error calculating predictions: {str(e)}")
            self._set_default_predictions()

    def _calculate_ma_signal(self) -> tuple[float, str]:
        """Calculate moving average signal and description."""
        if (self.df['ma20'].iloc[-1] > self.df['ma50'].iloc[-1] and 
            self.df['ma50'].iloc[-1] > self.df['ma200'].iloc[-1]):
            return 1.0, "Strong Bullish (Golden Cross)"
        elif (self.df['ma20'].iloc[-1] < self.df['ma50'].iloc[-1] and 
              self.df['ma50'].iloc[-1] < self.df['ma200'].iloc[-1]):
            return -1.0, "Strong Bearish (Death Cross)"
        elif self.df['ma20'].iloc[-1] > self.df['ma50'].iloc[-1]:
            return 0.5, "Moderate Bullish"
        elif self.df['ma20'].iloc[-1] < self.df['ma50'].iloc[-1]:
            return -0.5, "Moderate Bearish"
        return 0.0, "Neutral"

    def _calculate_rsi_signal(self) -> tuple[float, str]:
        """Calculate RSI signal and description."""
        current_rsi = self.df['rsi'].iloc[-1]
        if current_rsi < 30:
            return 1.0, "Strong Oversold"
        elif current_rsi > 70:
            return -1.0, "Strong Overbought"
        elif current_rsi < 40:
            return 0.5, "Moderate Oversold"
        elif current_rsi > 60:
            return -0.5, "Moderate Overbought"
        return 0.0, "Neutral"

    def _get_recommendation(self, combined_signal: float) -> str:
        """Get trading recommendation based on combined signal."""
        if combined_signal > 0.5:
            return "Buy"
        elif combined_signal < -0.5:
            return "Sell"
        return "Hold"

    def _get_signal_strength(self, combined_signal: float) -> str:
        """Get signal strength description."""
        if abs(combined_signal) > 0.7:
            return "Strong"
        elif abs(combined_signal) > 0.3:
            return "Moderate"
        return "Weak"

    def _calculate_period_returns(self) -> None:
        """Calculate total and annualized returns for the period."""
        try:
            start_close = self.df['close'].iloc[0]
            end_close = self.df['close'].iloc[-1]
            days_diff = (self.df['date'].iloc[-1] - self.df['date'].iloc[0]).days
            
            total_return = ((end_close / start_close) - 1) * 100 if start_close != 0 else 0
            annual_return = ((end_close / start_close) ** (365.0 / days_diff) - 1) * 100 if days_diff > 0 and start_close > 0 else 0
            
            self.insights.update({
                "total_return": float(total_return),
                "annual_return": float(annual_return)
            })
        except Exception as e:
            print(f"Error calculating period returns: {str(e)}")
            self._set_default_returns()

    def _calculate_final_stats(self) -> None:
        """Calculate summary stats and correlations."""
        try:
            numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 
                          'daily_return', 'volatility', 'ma20', 'ma50', 'ma200', 'rsi']
            valid_cols = [col for col in numeric_cols if col in self.df.columns]
            
            if valid_cols:
                self.insights["summary_stats"] = self.df[valid_cols].describe()
                
                corr_cols = ['close', 'volume', 'daily_return', 'volatility']
                valid_corr_cols = [col for col in corr_cols if col in valid_cols]
                if len(valid_corr_cols) > 1:
                    self.insights["correlations"] = self.df[valid_corr_cols].fillna(0).corr()
                else:
                    self.insights["correlations"] = pd.DataFrame()
        except Exception as e:
            print(f"Error calculating final stats: {str(e)}")
            self._set_default_stats()

    def _populate_default_metrics(self) -> None:
        """Populate insights with default values."""
        self._set_default_returns()
        self._set_default_volatility()
        self._set_default_moving_averages()
        self._set_default_predictions()
        self._set_default_stats()

    def _set_default_returns(self) -> None:
        """Set default return values."""
        self.insights.update({
            "max_daily_gain": 0.0,
            "max_daily_loss": 0.0,
            "avg_daily_return": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0
        })

    def _set_default_volatility(self) -> None:
        """Set default volatility values."""
        self.insights.update({
            "current_volatility": 0.0,
            "avg_volatility": 0.0
        })

    def _set_default_moving_averages(self) -> None:
        """Set default moving average values."""
        current_price = float(self.df['close'].iloc[-1])
        self.insights["moving_averages"] = {
            f"ma{period}": current_price
            for period in self.MOVING_AVERAGE_PERIODS
        }

    def _set_default_predictions(self) -> None:
        """Set default prediction values."""
        self.insights["predictions"] = {
            "ma_signal": 0.0,
            "ma_description": "Neutral",
            "rsi_signal": 0.0,
            "rsi_description": "Neutral",
            "combined_signal": 0.0,
            "recommendation": "Hold",
            "signal_strength": "Weak"
        }

    def _set_default_stats(self) -> None:
        """Set default statistics values."""
        self.insights.update({
            "summary_stats": pd.DataFrame(),
            "correlations": pd.DataFrame()
        }) 