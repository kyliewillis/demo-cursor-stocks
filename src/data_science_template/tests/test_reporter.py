"""Tests for the ReportGenerator class."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import shutil
from data_science_template.reporter import ReportGenerator

class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary test directory
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.test_df = pd.DataFrame({
            'date': dates,
            'close': np.random.normal(100, 10, len(dates)),
            'ma20': np.random.normal(100, 10, len(dates)),
            'ma50': np.random.normal(100, 10, len(dates)),
            'rsi': np.random.uniform(0, 100, len(dates)),
            'volatility': np.random.uniform(0, 5, len(dates))
        })
        
        # Create test insights
        self.test_insights = {
            'name': 'TEST_INDEX',
            'current_rsi': 50.0,
            'current_volatility': 2.5,
            'total_return': 10.0,
            'predictions': {
                'recommendation': 'Buy',
                'signal_strength': 'Strong',
                'ma_description': 'Bullish',
                'rsi_description': 'Neutral',
                'combined_signal': 0.8
            }
        }
        
        # Initialize ReportGenerator with test directory
        self.reporter = ReportGenerator(output_dir=str(self.test_dir))

    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test initialization of ReportGenerator."""
        self.assertEqual(self.reporter.output_dir, str(self.test_dir))
        self.assertTrue(os.path.exists(self.test_dir))

    def test_create_visualizations(self):
        """Test creation of visualizations."""
        visualizations = self.reporter.create_visualizations(
            self.test_df,
            self.test_insights,
            "TEST_INDEX"
        )
        
        # Check if all expected visualizations are created
        self.assertTrue('price_ma' in visualizations)
        self.assertTrue('rsi' in visualizations)
        self.assertTrue('volatility' in visualizations)
        
        # Check if visualizations are base64 encoded
        for viz in visualizations.values():
            self.assertTrue(isinstance(viz, str))
            self.assertTrue(len(viz) > 0)

    def test_create_overlay_visualization(self):
        """Test creation of overlay visualization."""
        all_data = {
            'TEST_INDEX': self.test_df,
            'OTHER_INDEX': self.test_df.copy()
        }
        
        overlay = self.reporter.create_overlay_visualization(all_data)
        
        # Check if overlay is base64 encoded
        self.assertTrue(isinstance(overlay, str))
        self.assertTrue(len(overlay) > 0)

    def test_generate_html_report(self):
        """Test HTML and PDF report generation."""
        all_data = {
            'TEST_INDEX': self.test_df,
            'OTHER_INDEX': self.test_df.copy()
        }
        all_insights = {
            'TEST_INDEX': self.test_insights,
            'OTHER_INDEX': self.test_insights.copy()
        }
        all_visualizations = {
            'TEST_INDEX': self.reporter.create_visualizations(
                self.test_df,
                self.test_insights,
                "TEST_INDEX"
            ),
            'OTHER_INDEX': self.reporter.create_visualizations(
                self.test_df,
                self.test_insights,
                "OTHER_INDEX"
            )
        }
        
        html_path, pdf_path = self.reporter.generate_html_report(
            all_data,
            all_insights,
            all_visualizations
        )
        
        # Check if both reports were generated
        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(os.path.exists(pdf_path))
        self.assertTrue(html_path.endswith('market_indices_report.html'))
        self.assertTrue(pdf_path.endswith('market_indices_report.pdf'))

    def test_get_signal_class(self):
        """Test signal class determination."""
        # Test bullish signal
        bullish_insights = self.test_insights.copy()
        bullish_insights['predictions']['combined_signal'] = 0.5
        self.assertEqual(self.reporter._get_signal_class(bullish_insights), 'bullish')
        
        # Test bearish signal
        bearish_insights = self.test_insights.copy()
        bearish_insights['predictions']['combined_signal'] = -0.5
        self.assertEqual(self.reporter._get_signal_class(bearish_insights), 'bearish')
        
        # Test neutral signal
        neutral_insights = self.test_insights.copy()
        neutral_insights['predictions']['combined_signal'] = 0.1
        self.assertEqual(self.reporter._get_signal_class(neutral_insights), 'neutral')

    def test_get_combined_signal(self):
        """Test combined signal calculation."""
        all_insights = {
            'INDEX1': {
                'predictions': {'combined_signal': 0.8}
            },
            'INDEX2': {
                'predictions': {'combined_signal': 0.6}
            }
        }
        
        combined = self.reporter._get_combined_signal(all_insights)
        
        # Check combined signal properties
        self.assertTrue('signal' in combined)
        self.assertTrue('signal_class' in combined)
        self.assertTrue('bullish' in combined['signal'].lower())
        self.assertEqual(combined['signal_class'], 'bullish')

    def test_generate_index_section(self):
        """Test index section generation."""
        visualizations = self.reporter.create_visualizations(
            self.test_df,
            self.test_insights,
            "TEST_INDEX"
        )
        
        section = self.reporter._generate_index_section(
            "TEST_INDEX",
            self.test_insights,
            visualizations
        )
        
        # Check if section contains expected elements
        self.assertTrue('TEST_INDEX Analysis' in section)
        self.assertTrue('Overall Recommendation' in section)
        self.assertTrue('Technical Signals' in section)
        self.assertTrue('Volatility' in section)
        self.assertTrue('Total Return' in section)
        self.assertTrue('Price and Moving Averages' in section)
        self.assertTrue('RSI' in section)

if __name__ == '__main__':
    unittest.main() 