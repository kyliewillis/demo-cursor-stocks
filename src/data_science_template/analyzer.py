"""Module for analyzing market index data."""

import pandas as pd
import numpy as np
from typing import Dict, Any

class MarketIndexAnalyzer:
    """Analyzes a DataFrame of market index data to calculate insights."""

    def __init__(self, df: pd.DataFrame, index_name: str):
        """Initialize the analyzer."""
        if df.empty or 'date' not in df.columns or 'close' not in df.columns:
            raise ValueError("Input DataFrame is empty or missing required columns ('date', 'close').")
        self.df = df.copy()
        self.index_name = index_name
        self.insights = {"name": index_name}
        self.df['date'] = pd.to_datetime(self.df['date'])

    def calculate_all_insights(self) -> Dict[str, Any]:
        """Calculate all insights for the index."""
        if len(self.df) < 2:
            print(f"Warning: Insufficient data points ({len(self.df)}) for full analysis of {self.index_name}.")
            self._populate_default_metrics()
            return self.insights
            
        self._populate_basic_info()
        self._calculate_returns()
        self._calculate_volatility()
        self._calculate_moving_averages()
        self._calculate_rsi()
        self._calculate_predictions()
        self._calculate_period_returns()
        self._calculate_final_stats()
        
        return self.insights

    def _populate_basic_info(self):
        """Populate basic info like latest price, date, high, low, volume."""
        self.insights.update({
            "latest_close": self.df['close'].iloc[-1],
            "latest_date": self.df['date'].iloc[-1].strftime('%Y-%m-%d'),
            "current_price": self.df['close'].iloc[-1],
            "year_high": self.df['high'].max() if 'high' in self.df.columns else self.df['close'].iloc[-1],
            "year_low": self.df['low'].min() if 'low' in self.df.columns else self.df['close'].iloc[-1],
            "latest_volume": self.df['volume'].iloc[-1] if 'volume' in self.df.columns else 0,
            "avg_volume": self.df['volume'].mean() if 'volume' in self.df.columns else 0
        })

    def _calculate_returns(self):
        """Calculate daily percentage returns."""
        self.df['daily_return'] = self.df['close'].pct_change() * 100
        self.df['daily_return'] = self.df['daily_return'].fillna(0)
        self.insights.update({
            "max_daily_gain": self.df['daily_return'].max(),
            "max_daily_loss": self.df['daily_return'].min(),
            "avg_daily_return": self.df['daily_return'].mean()
        })

    def _calculate_volatility(self, window: int = 20):
        """Calculate rolling volatility."""
        self.df['volatility'] = self.df['daily_return'].rolling(window=window).std().fillna(0)
        self.insights.update({
            "current_volatility": self.df['volatility'].iloc[-1],
            "avg_volatility": self.df['volatility'].mean()
        })

    def _calculate_moving_averages(self):
        """Calculate moving averages."""
        # Calculate and store moving averages in the DataFrame
        for period in [20, 50, 200]:
            col_name = f'ma{period}'
            # Calculate rolling mean and store in DataFrame
            self.df[col_name] = self.df['close'].rolling(window=period).mean()
            # Forward fill NaN values
            self.df[col_name] = self.df[col_name].ffill()
            # Backward fill any remaining NaN values
            self.df[col_name] = self.df[col_name].bfill()
            
        # Store current values in insights
        self.insights["moving_averages"] = {
            f"ma{period}": float(self.df[f'ma{period}'].iloc[-1])
            for period in [20, 50, 200]
        }
        
        # Ensure the columns are in the DataFrame
        assert all(col in self.df.columns for col in ['ma20', 'ma50', 'ma200']), "Moving average columns not properly created"

    def _calculate_rsi(self, window: int = 14):
        """Calculate Relative Strength Index (RSI)."""
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss.replace(0, 1e-6)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi'] = self.df['rsi'].fillna(50)
        self.insights["current_rsi"] = self.df['rsi'].iloc[-1]

    def _calculate_predictions(self):
        """Calculate prediction signals based on technical indicators."""
        # Ensure moving averages exist
        if not all(col in self.df.columns for col in ['ma20', 'ma50', 'ma200']):
            self._calculate_moving_averages()
            
        # Calculate MA signal based on moving average relationships
        ma_signal = 0.0
        ma_description = "Neutral"
        if (self.df['ma20'].iloc[-1] > self.df['ma50'].iloc[-1] and 
            self.df['ma50'].iloc[-1] > self.df['ma200'].iloc[-1]):
            ma_signal = 1.0
            ma_description = "Strong Bullish (Golden Cross)"
        elif (self.df['ma20'].iloc[-1] < self.df['ma50'].iloc[-1] and 
              self.df['ma50'].iloc[-1] < self.df['ma200'].iloc[-1]):
            ma_signal = -1.0
            ma_description = "Strong Bearish (Death Cross)"
        elif self.df['ma20'].iloc[-1] > self.df['ma50'].iloc[-1]:
            ma_signal = 0.5
            ma_description = "Moderate Bullish"
        elif self.df['ma20'].iloc[-1] < self.df['ma50'].iloc[-1]:
            ma_signal = -0.5
            ma_description = "Moderate Bearish"
        
        # Calculate RSI signal
        current_rsi = self.df['rsi'].iloc[-1]
        rsi_signal = 0.0
        rsi_description = "Neutral"
        if current_rsi < 30:
            rsi_signal = 1.0
            rsi_description = "Strong Oversold"
        elif current_rsi > 70:
            rsi_signal = -1.0
            rsi_description = "Strong Overbought"
        elif current_rsi < 40:
            rsi_signal = 0.5
            rsi_description = "Moderate Oversold"
        elif current_rsi > 60:
            rsi_signal = -0.5
            rsi_description = "Moderate Overbought"
        
        # Calculate combined signal (weighted average)
        combined_signal = (ma_signal * 0.6 + rsi_signal * 0.4)  # MA has more weight
        
        # Determine overall recommendation
        recommendation = "Hold"
        if combined_signal > 0.5:
            recommendation = "Buy"
        elif combined_signal < -0.5:
            recommendation = "Sell"
            
        # Calculate signal strength
        signal_strength = "Strong" if abs(combined_signal) > 0.7 else "Moderate" if abs(combined_signal) > 0.3 else "Weak"
        
        self.insights["predictions"] = {
            "ma_signal": ma_signal,
            "ma_description": ma_description,
            "rsi_signal": rsi_signal,
            "rsi_description": rsi_description,
            "combined_signal": combined_signal,
            "recommendation": recommendation,
            "signal_strength": signal_strength
        }

    def _calculate_period_returns(self):
        """Calculate total and annualized returns for the period."""
        start_close = self.df['close'].iloc[0]
        end_close = self.df['close'].iloc[-1]
        days_diff = (self.df['date'].iloc[-1] - self.df['date'].iloc[0]).days
        
        total_return = ((end_close / start_close) - 1) * 100 if start_close != 0 else 0
        annual_return = ((end_close / start_close) ** (365.0 / days_diff) - 1) * 100 if days_diff > 0 and start_close > 0 else 0
        
        self.insights.update({
            "total_return": total_return,
            "annual_return": annual_return
        })

    def _calculate_final_stats(self):
        """Calculate summary stats and correlations."""
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

    def _populate_default_metrics(self):
        """Populate insights with default values."""
        defaults = {
            'max_daily_gain': 0.0, 'max_daily_loss': 0.0, 'avg_daily_return': 0.0,
            'current_volatility': 0.0, 'avg_volatility': 0.0, 'current_rsi': 50.0,
            'total_return': 0.0, 'annual_return': 0.0,
            'ma_signal': "N/A", 'rsi_signal': "N/A", 'combined_signal': "N/A",
            'summary_stats': pd.DataFrame(), 'correlations': pd.DataFrame()
        }
        self.insights.update(defaults) 