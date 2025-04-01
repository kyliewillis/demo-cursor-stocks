"""Tests for the MarketIndexAnalyzer class."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_science_template.analyzer import MarketIndexAnalyzer

class TestMarketIndexAnalyzer(unittest.TestCase):
    """Test cases for MarketIndexAnalyzer class."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        self.analyzer = MarketIndexAnalyzer(self.test_data, "TEST_INDEX")

    def test_initialization(self):
        """Test initialization of MarketIndexAnalyzer."""
        # Test valid initialization
        self.assertEqual(self.analyzer.index_name, "TEST_INDEX")
        self.assertTrue(isinstance(self.analyzer.df, pd.DataFrame))
        
        # Test invalid initialization
        with self.assertRaises(ValueError):
            MarketIndexAnalyzer(pd.DataFrame(), "TEST_INDEX")
        
        with self.assertRaises(ValueError):
            MarketIndexAnalyzer(pd.DataFrame({'wrong_column': [1]}), "TEST_INDEX")

    def test_populate_basic_info(self):
        """Test basic info population."""
        self.analyzer._populate_basic_info()
        insights = self.analyzer.insights
        
        self.assertEqual(insights['latest_close'], self.test_data['close'].iloc[-1])
        self.assertEqual(insights['latest_date'], self.test_data['date'].iloc[-1].strftime('%Y-%m-%d'))
        self.assertEqual(insights['year_high'], self.test_data['high'].max())
        self.assertEqual(insights['year_low'], self.test_data['low'].min())
        self.assertEqual(insights['latest_volume'], self.test_data['volume'].iloc[-1])

    def test_calculate_returns(self):
        """Test returns calculation."""
        self.analyzer._calculate_returns()
        
        # Check if daily_return column exists
        self.assertTrue('daily_return' in self.analyzer.df.columns)
        
        # Check if insights are populated
        self.assertTrue('max_daily_gain' in self.analyzer.insights)
        self.assertTrue('max_daily_loss' in self.analyzer.insights)
        self.assertTrue('avg_daily_return' in self.analyzer.insights)

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        self.analyzer._calculate_returns()  # Required for volatility calculation
        self.analyzer._calculate_volatility()
        
        # Check if volatility column exists
        self.assertTrue('volatility' in self.analyzer.df.columns)
        
        # Check if insights are populated
        self.assertTrue('current_volatility' in self.analyzer.insights)
        self.assertTrue('avg_volatility' in self.analyzer.insights)

    def test_calculate_moving_averages(self):
        """Test moving averages calculation."""
        self.analyzer._calculate_moving_averages()
        
        # Check if all MA columns exist
        for period in [20, 50, 200]:
            self.assertTrue(f'ma{period}' in self.analyzer.df.columns)
        
        # Check if insights are populated
        self.assertTrue('moving_averages' in self.analyzer.insights)
        for period in [20, 50, 200]:
            self.assertTrue(f'ma{period}' in self.analyzer.insights['moving_averages'])

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        self.analyzer._calculate_rsi()
        
        # Check if RSI column exists
        self.assertTrue('rsi' in self.analyzer.df.columns)
        
        # Check if insights are populated
        self.assertTrue('current_rsi' in self.analyzer.insights)
        
        # Check RSI range
        self.assertTrue(0 <= self.analyzer.insights['current_rsi'] <= 100)

    def test_calculate_predictions(self):
        """Test predictions calculation."""
        self.analyzer._calculate_moving_averages()
        self.analyzer._calculate_rsi()
        self.analyzer._calculate_predictions()
        
        # Check if predictions are populated
        self.assertTrue('predictions' in self.analyzer.insights)
        predictions = self.analyzer.insights['predictions']
        
        # Check required fields
        required_fields = ['ma_signal', 'ma_description', 'rsi_signal', 
                         'rsi_description', 'combined_signal', 'recommendation', 
                         'signal_strength']
        for field in required_fields:
            self.assertTrue(field in predictions)

    def test_calculate_period_returns(self):
        """Test period returns calculation."""
        self.analyzer._calculate_period_returns()
        
        # Check if insights are populated
        self.assertTrue('total_return' in self.analyzer.insights)
        self.assertTrue('annual_return' in self.analyzer.insights)

    def test_calculate_final_stats(self):
        """Test final statistics calculation."""
        self.analyzer._calculate_returns()
        self.analyzer._calculate_volatility()
        self.analyzer._calculate_moving_averages()
        self.analyzer._calculate_rsi()
        self.analyzer._calculate_final_stats()
        
        # Check if insights are populated
        self.assertTrue('summary_stats' in self.analyzer.insights)
        self.assertTrue('correlations' in self.analyzer.insights)

    def test_populate_default_metrics(self):
        """Test default metrics population."""
        self.analyzer._populate_default_metrics()
        
        # Check if all default metrics are populated
        default_fields = ['max_daily_gain', 'max_daily_loss', 'avg_daily_return',
                         'current_volatility', 'avg_volatility', 'current_rsi',
                         'total_return', 'annual_return']
        for field in default_fields:
            self.assertTrue(field in self.analyzer.insights)

    def test_calculate_all_insights(self):
        """Test complete analysis workflow."""
        insights = self.analyzer.calculate_all_insights()
        
        # Check if all major components are present
        self.assertTrue('name' in insights)
        self.assertTrue('latest_close' in insights)
        self.assertTrue('moving_averages' in insights)
        self.assertTrue('predictions' in insights)
        self.assertTrue('summary_stats' in insights)
        self.assertTrue('correlations' in insights)

if __name__ == '__main__':
    unittest.main() 