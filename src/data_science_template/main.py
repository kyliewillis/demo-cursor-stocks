"""Main script for S&P 500 data analysis and report generation."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import base64
from io import BytesIO

def load_data() -> pd.DataFrame:
    """Load S&P 500 data from CSV file.
    
    Returns:
        pd.DataFrame: Loaded S&P 500 data
    """
    data_path = Path("data/raw/sp500_data.csv")
    return pd.read_csv(data_path)

def calculate_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key insights from the S&P 500 data.
    
    Args:
        df: S&P 500 data DataFrame
        
    Returns:
        Dict containing various insights
    """
    # Calculate daily returns
    df['daily_return'] = df['adj_close'].pct_change() * 100
    
    # Calculate moving averages
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Calculate Relative Strength Index (RSI) - simplified version
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- Simple Prediction Logic ---
    current_rsi = df['rsi'].iloc[-1]
    ma50_above_ma200 = df['ma50'].iloc[-1] > df['ma200'].iloc[-1]
    
    prediction_signal = "Hold" # Default
    if current_rsi < 30 and ma50_above_ma200:
        prediction_signal = "Strong Buy"
    elif current_rsi < 40 and ma50_above_ma200:
         prediction_signal = "Buy"
    elif current_rsi > 70 and not ma50_above_ma200:
        prediction_signal = "Strong Sell"
    elif current_rsi > 60 and not ma50_above_ma200:
        prediction_signal = "Sell"
    elif ma50_above_ma200 and current_rsi <= 55: # General bullish trend, not oversold
        prediction_signal = "Consider Buy/Hold"
    elif not ma50_above_ma200 and current_rsi >= 45: # General bearish trend, not overbought
        prediction_signal = "Consider Sell/Hold"
    # --- End Prediction Logic ---

    insights = {
        # Summary statistics
        "summary_stats": df.describe(),
        
        # Latest data
        "latest_close": df['close'].iloc[-1],
        "latest_date": df['date'].iloc[-1],
        
        # Performance metrics
        "total_return": ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
        # Ensure date conversion for calculation
        "annual_return": ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (365 / (pd.to_datetime(df['date'].iloc[-1]) - pd.to_datetime(df['date'].iloc[0])).days) - 1) * 100 if (pd.to_datetime(df['date'].iloc[-1]) - pd.to_datetime(df['date'].iloc[0])).days > 0 else 0,
        "max_daily_gain": df['daily_return'].max(),
        "max_daily_loss": df['daily_return'].min(),
        "avg_daily_return": df['daily_return'].mean(),
        
        # Volatility metrics
        "avg_volatility": df['volatility'].mean(),
        "current_volatility": df['volatility'].iloc[-1],
        
        # Technical indicators
        "current_rsi": current_rsi, # Use calculated variable
        "ma_signal": "Bullish" if ma50_above_ma200 else "Bearish", # Use calculated variable
        
        # Price trends
        "current_price": df['close'].iloc[-1],
        "year_high": df['high'].max(),
        "year_low": df['low'].min(),
        
        # Trading volume
        "avg_volume": df['volume'].mean(),
        "latest_volume": df['volume'].iloc[-1],
        
        # Correlations
        "correlations": df[['close', 'volume', 'daily_return', 'volatility']].corr(),

        # Prediction Signal
        "prediction_signal": prediction_signal 
    }
    return insights

def create_visualizations(df: pd.DataFrame, insights: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for the report.
    
    Args:
        df: S&P 500 data DataFrame
        insights: Dictionary of calculated insights
        
    Returns:
        Dict containing base64 encoded matplotlib figures
    """
    figures = {}
    
    # Set style
    plt.style.use('default')
    
    # Price chart with moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df['date']), df['close'], label='S&P 500', alpha=0.7)
    plt.plot(pd.to_datetime(df['date']), df['ma50'], label='50-day MA', alpha=0.8)
    plt.plot(pd.to_datetime(df['date']), df['ma200'], label='200-day MA', alpha=0.8)
    plt.title('S&P 500 Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    figures['price_chart'] = fig_to_base64()
    
    # Daily returns
    plt.figure(figsize=(12, 5))
    plt.bar(pd.to_datetime(df['date']), df['daily_return'], alpha=0.7, color='darkblue')
    plt.title('S&P 500 Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    figures['returns_chart'] = fig_to_base64()
    
    # Volatility
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(df['date']), df['volatility'], color='red', alpha=0.7)
    plt.title('S&P 500 Volatility (20-day rolling std dev)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True, alpha=0.3)
    figures['volatility_chart'] = fig_to_base64()
    
    # Volume
    plt.figure(figsize=(12, 5))
    plt.bar(pd.to_datetime(df['date']), df['volume'], alpha=0.7, color='green')
    plt.title('S&P 500 Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    figures['volume_chart'] = fig_to_base64()
    
    # RSI
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(df['date']), df['rsi'], color='purple', alpha=0.7)
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.grid(True, alpha=0.3)
    figures['rsi_chart'] = fig_to_base64()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(insights['correlations'], annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    figures['correlation'] = fig_to_base64()
    
    return figures

def fig_to_base64() -> str:
    """Convert matplotlib figure to base64 string.
    
    Returns:
        str: Base64 encoded figure
    """
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_html_report(df: pd.DataFrame, insights: Dict[str, Any], 
                        visualizations: Dict[str, Any]) -> str:
    """Generate HTML report with insights and visualizations.
    
    Args:
        df: S&P 500 data DataFrame
        insights: Dictionary of calculated insights
        visualizations: Dictionary of base64 encoded matplotlib figures
        
    Returns:
        str: Generated HTML report
    """
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>S&P 500 Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; }}
            .metrics {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; width: 30%; }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .metric-name {{ font-size: 14px; color: #666; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .neutral {{ color: #444; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>S&P 500 Analysis Report</h1>
            <p>Generated on: {report_date} | Data from {pd.to_datetime(df['date'].iloc[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d')}</p>
            
            <div class="section">
                <h2>Prediction Signal</h2>
                <div class="prediction-card" style="background-color: #e9ecef; padding: 20px; border-radius: 5px; text-align: center;">
                    <p style="font-size: 20px; margin: 0;">Current Signal: 
                        <span class="{
                            'positive' if 'Buy' in insights['prediction_signal'] else 
                            ('negative' if 'Sell' in insights['prediction_signal'] else 'neutral')
                        }" style="font-weight: bold;">{insights['prediction_signal']}</span>
                    </p>
                    <p style="font-size: 12px; color: #6c757d; margin-top: 5px;">
                        Based on RSI ({insights['current_rsi']:.1f}) and Moving Average ({insights['ma_signal']}) analysis.
                        <br><i>This is not financial advice.</i>
                    </p>
                </div>
            </div>

            <div class="section">
                <h2>Key Metrics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-name">Latest Close</div>
                        <div class="metric-value neutral">${insights['latest_close']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Total Return</div>
                        <div class="metric-value {'positive' if insights['total_return'] > 0 else 'negative'}">{insights['total_return']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Annual Return</div>
                        <div class="metric-value {'positive' if insights['annual_return'] > 0 else 'negative'}">{insights['annual_return']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Average Daily Return</div>
                        <div class="metric-value {'positive' if insights['avg_daily_return'] > 0 else 'negative'}">{insights['avg_daily_return']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Current Volatility</div>
                        <div class="metric-value neutral">{insights['current_volatility']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">MA Signal</div>
                        <div class="metric-value {'positive' if insights['ma_signal'] == 'Bullish' else 'negative'}">{insights['ma_signal']}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Price Performance</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{visualizations['price_chart']}" alt="S&P 500 Price Chart">
                </div>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Current Price</td>
                        <td>${insights['current_price']:.2f}</td>
                    </tr>
                    <tr>
                        <td>52-Week High</td>
                        <td>${insights['year_high']:.2f}</td>
                    </tr>
                    <tr>
                        <td>52-Week Low</td>
                        <td>${insights['year_low']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Daily Gain</td>
                        <td>{insights['max_daily_gain']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Max Daily Loss</td>
                        <td>{insights['max_daily_loss']:.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Returns Analysis</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{visualizations['returns_chart']}" alt="S&P 500 Returns">
                </div>
            </div>
            
            <div class="section">
                <h2>Volatility Analysis</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{visualizations['volatility_chart']}" alt="S&P 500 Volatility">
                </div>
            </div>
            
            <div class="section">
                <h2>Volume Analysis</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{visualizations['volume_chart']}" alt="S&P 500 Volume">
                </div>
                <p>Average Daily Volume: {insights['avg_volume']:.0f} | Latest Volume: {insights['latest_volume']:.0f}</p>
            </div>
            
            <div class="section">
                <h2>Technical Indicators</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{visualizations['rsi_chart']}" alt="RSI Chart">
                </div>
                <p>Current RSI: {insights['current_rsi']:.2f} (Overbought > 70, Oversold < 30)</p>
                <p>Moving Average Signal: {insights['ma_signal']} (50-day MA {'above' if insights['ma_signal'] == 'Bullish' else 'below'} 200-day MA)</p>
            </div>
            
            <div class="section">
                <h2>Correlation Analysis</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{visualizations['correlation']}" alt="Correlation Heatmap">
                </div>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                {insights['summary_stats'].to_html()}
            </div>
        </div>
    </body>
    </html>
    """
    return html

def main():
    """Main function to run the analysis and generate the report."""
    # Load data
    df = load_data()
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate insights
    insights = calculate_insights(df)
    
    # Create visualizations
    visualizations = create_visualizations(df, insights)
    
    # Generate HTML report
    html_report = generate_html_report(df, insights, visualizations)
    
    # Save report
    output_path = Path("out/sp500_report.html")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html_report)
    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    main() 