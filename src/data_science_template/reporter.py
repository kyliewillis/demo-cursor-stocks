"""Module for generating HTML reports with market analysis visualizations."""

import os
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from jinja2 import Environment, FileSystemLoader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch


class ReportGenerator:
    """Class for generating HTML reports with market analysis visualizations."""

    def __init__(self, output_dir: str = "out"):
        """Initialize the ReportGenerator.

        Args:
            output_dir: Directory to save generated reports.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_visualizations(
        self, df: pd.DataFrame, insights: Dict, index_name: str
    ) -> Dict[str, str]:
        """Create visualizations for a market index.

        Args:
            df: DataFrame containing market data.
            insights: Dictionary containing calculated insights.
            index_name: Name of the market index.

        Returns:
            Dictionary mapping visualization names to base64-encoded images.
        """
        visualizations = {}

        # Price and Moving Averages
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["date"], df["close"], label="Close Price", alpha=0.7)
        ax.plot(df["date"], df["ma20"], label="20-day MA", alpha=0.7)
        ax.plot(df["date"], df["ma50"], label="50-day MA", alpha=0.7)
        ax.set_title(f"{index_name} Price and Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        visualizations["price_ma"] = self._fig_to_base64(fig)
        plt.close(fig)

        # RSI
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["date"], df["RSI"], label="RSI", color="purple")
        ax.axhline(y=70, color="r", linestyle="--", alpha=0.5)
        ax.axhline(y=30, color="g", linestyle="--", alpha=0.5)
        ax.set_title(f"{index_name} RSI")
        ax.set_xlabel("Date")
        ax.set_ylabel("RSI")
        ax.legend()
        ax.grid(True, alpha=0.3)
        visualizations["rsi"] = self._fig_to_base64(fig)
        plt.close(fig)

        # Volatility
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["date"], df["Volatility"], label="Volatility", color="orange")
        ax.set_title(f"{index_name} Volatility")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.legend()
        ax.grid(True, alpha=0.3)
        visualizations["volatility"] = self._fig_to_base64(fig)
        plt.close(fig)

        return visualizations

    def create_overlay_visualization(
        self, all_data: Dict[str, pd.DataFrame]
    ) -> str:
        """Create an overlay visualization of normalized prices for all indices.

        Args:
            all_data: Dictionary mapping index names to their DataFrames.

        Returns:
            Base64-encoded image of the overlay plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, df in all_data.items():
            # Normalize prices to start at 100
            normalized_price = df["close"] / df["close"].iloc[0] * 100
            ax.plot(df["date"], normalized_price, label=name, alpha=0.7)

        ax.set_title("Normalized Market Index Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price (100 = Start)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._fig_to_base64(fig)

    def generate_html_report(
        self,
        all_data: Dict[str, pd.DataFrame],
        all_insights: Dict[str, Dict],
        all_visualizations: Dict[str, Dict[str, str]],
        all_predictions: Dict[str, Dict[str, float]],
    ) -> Tuple[str, str]:
        """Generate HTML and PDF reports with market analysis.

        Args:
            all_data: Dictionary mapping index names to their DataFrames.
            all_insights: Dictionary mapping index names to their insights.
            all_visualizations: Dictionary mapping index names to their visualizations.
            all_predictions: Dictionary mapping index names to their ML predictions.

        Returns:
            Tuple containing paths to the generated HTML and PDF reports.
        """
        # Get combined market signal
        combined_signal = self._get_combined_signal(all_insights)
        
        # Load and render template
        template = self.env.get_template('report_template.html')
        html_content = template.render(
            all_insights=all_insights,
            all_visualizations=all_visualizations,
            all_predictions=all_predictions,
            signal=combined_signal["signal"],
            signal_class=combined_signal["signal_class"],
            overlay_plot=self.create_overlay_visualization(all_data)
        )
        
        # Save HTML report
        html_path = os.path.join(self.output_dir, "market_indices_report.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        # Generate PDF from HTML
        pdf_path = os.path.join(self.output_dir, "market_indices_report.pdf")
        self._generate_pdf(html_content, pdf_path)
            
        return html_path, pdf_path

    def _generate_pdf(self, html_content: str, output_path: str) -> None:
        """Generate PDF from HTML content.

        Args:
            html_content: HTML string to convert to PDF.
            output_path: Path where to save the PDF file.
        """
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create content
        story = []
        
        # Add title
        story.append(Paragraph("Market Indices Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Add market summary
        story.append(Paragraph("Overall Market Summary", heading_style))
        story.append(Spacer(1, 12))
        
        # Add signal information
        signal_style = ParagraphStyle(
            'SignalStyle',
            parent=normal_style,
            fontSize=12,
            spaceAfter=30
        )
        
        # Extract signal from HTML content
        import re
        signal_match = re.search(r'<div class="market-signal signal .*?">\s*(.*?)\s*</div>', html_content, re.DOTALL)
        signal_text = signal_match.group(1).strip() if signal_match else "No Signal Available"
        story.append(Paragraph(f"Market Analysis: {signal_text}", signal_style))
        
        # Extract and add overlay plot from HTML content
        overlay_match = re.search(r'<img src="data:image/png;base64,(.*?)" alt="Overlay Plot"', html_content, re.DOTALL)
        if overlay_match:
            img_data = base64.b64decode(overlay_match.group(1))
            img = Image(BytesIO(img_data))
            img.drawHeight = 4*inch
            img.drawWidth = 7*inch
            story.append(img)
        story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)

    def _generate_index_section(
        self,
        index_name: str,
        insights: Dict,
        visualizations: Dict[str, str],
    ) -> str:
        """Generate HTML section for a single index.

        Args:
            index_name: Name of the market index.
            insights: Dictionary containing calculated insights.
            visualizations: Dictionary containing base64-encoded visualizations.

        Returns:
            HTML string for the index section.
        """
        predictions = insights['predictions']
        signal_class = self._get_signal_class(insights)
        
        return f"""
            <div class="section">
                <h2>{index_name} Analysis</h2>
                <div class="summary">
                    <div class="metric">
                        <strong>Overall Recommendation:</strong>
                        <div class="signal {signal_class}">
                            <div style="font-size: 1.2em; font-weight: bold;">
                                {predictions['recommendation']} ({predictions['signal_strength']} Signal)
                            </div>
                        </div>
                    </div>
                    <div class="metric">
                        <strong>Technical Signals:</strong>
                        <div class="signal {signal_class}">
                            <div>Moving Averages: {predictions['ma_description']}</div>
                            <div>RSI: {predictions['rsi_description']} (Current: {insights['current_rsi']:.1f})</div>
                            <div>Combined Signal Strength: {predictions['signal_strength']}</div>
                        </div>
                    </div>
                    <div class="metric">
                        <strong>Volatility:</strong> {insights['current_volatility']:.2f}%
                    </div>
                    <div class="metric">
                        <strong>Total Return:</strong> {insights['total_return']:.2f}%
                    </div>
                </div>
                <div class="chart">
                    <h3>Price and Moving Averages</h3>
                    <img src="data:image/png;base64,{visualizations['price_ma']}" alt="Price and MA">
                </div>
                <div class="chart">
                    <h3>RSI</h3>
                    <img src="data:image/png;base64,{visualizations['rsi']}" alt="RSI">
                </div>
                <div class="chart">
                    <h3>Volatility</h3>
                    <img src="data:image/png;base64,{visualizations['volatility']}" alt="Volatility">
                </div>
            </div>
        """

    def _get_signal_class(self, insights: Dict) -> str:
        """Get CSS class for the prediction signal.

        Args:
            insights: Dictionary containing calculated insights.

        Returns:
            CSS class name for the signal.
        """
        if 'predictions' not in insights:
            return "neutral"
            
        combined_signal = insights['predictions']['combined_signal']
        if combined_signal > 0.3:
            return "bullish"
        elif combined_signal < -0.3:
            return "bearish"
        return "neutral"

    def _get_combined_signal(self, all_insights: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate combined market signal from all indices.
        
        Args:
            all_insights: Dictionary of insights for each index.
            
        Returns:
            Dictionary containing combined signal information.
        """
        if not all_insights:
            return {
                "signal": "No Data Available",
                "signal_class": "neutral",
                "strength": "N/A",
                "value": 0
            }
            
        signals = []
        for insights in all_insights.values():
            if 'predictions' in insights and 'combined_signal' in insights['predictions']:
                signals.append(insights['predictions']['combined_signal'])
            
        if not signals:
            return {
                "signal": "No Signals Available",
                "signal_class": "neutral",
                "strength": "N/A",
                "value": 0
            }
            
        avg_signal = sum(signals) / len(signals)
        
        # Determine signal strength and class
        if abs(avg_signal) >= 0.7:
            strength = "Strong"
        elif abs(avg_signal) >= 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"
            
        if avg_signal > 0:
            signal_class = "bullish"
            signal = f"Strong Bullish ({strength} Signal)"
        elif avg_signal < 0:
            signal_class = "bearish"
            signal = f"Strong Bearish ({strength} Signal)"
        else:
            signal_class = "neutral"
            signal = f"Neutral ({strength} Signal)"
            
        return {
            "signal": signal,
            "signal_class": signal_class,
            "strength": strength,
            "value": avg_signal
        }

    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert a matplotlib figure to base64 string.

        Args:
            fig: Matplotlib figure to convert.

        Returns:
            Base64-encoded string of the figure.
        """
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode() 