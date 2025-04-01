"""Main script for Market Index (S&P 500, Dow, Nasdaq) data analysis and report generation."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, Any, Tuple, Optional
import base64
from io import BytesIO

# Define the indices to analyze
INDICES = {
    "sp500": {"name": "S&P 500", "ticker": "^GSPC", "file": "sp500_data.csv"},
    "dow": {"name": "Dow Jones", "ticker": "^DJI", "file": "dow_data.csv"},
    "nasdaq": {"name": "Nasdaq", "ticker": "^IXIC", "file": "nasdaq_data.csv"}
}

# --- Data Loading ---

def load_index_data(index_key: str) -> pd.DataFrame:
    """Load market index data from CSV file."""
    if index_key not in INDICES:
        raise ValueError(f"Unknown index key: {index_key}")
    
    file_name = INDICES[index_key]["file"]
    data_path = Path("data/raw") / file_name
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Please run fetch_market_data.py first.")
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# --- Calculation Helper Functions ---

def _calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate daily percentage returns."""
    if 'adj_close' in df.columns and not df['adj_close'].isnull().all():
        returns = df['adj_close'].pct_change() * 100
    elif 'close' in df.columns and not df['close'].isnull().all():
        returns = df['close'].pct_change() * 100
    else:
        returns = pd.Series(0.0, index=df.index)
    return returns.fillna(0)

def _calculate_volatility(daily_returns: pd.Series, window: int = 20) -> Optional[pd.Series]:
    """Calculate rolling volatility."""
    if len(daily_returns) >= window:
        volatility = daily_returns.rolling(window=window).std()
        return volatility.fillna(0)
    return None

def _calculate_moving_averages(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Calculate 50-day and 200-day moving averages."""
    ma50 = None
    ma200 = None
    if len(df) >= 50:
        ma50 = df['close'].rolling(window=50).mean()
    if len(df) >= 200:
        ma200 = df['close'].rolling(window=200).mean()
    return ma50, ma200

def _calculate_rsi(df: pd.DataFrame, window: int = 14) -> Optional[pd.Series]:
    """Calculate Relative Strength Index (RSI)."""
    if len(df) >= window + 1: # Need window + 1 for diff()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss.replace(0, 1e-6) # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) # Fill initial NaNs with neutral 50
    return None

def _generate_prediction(rsi: Optional[pd.Series], ma50: Optional[pd.Series], ma200: Optional[pd.Series]) -> Tuple[str, str]:
    """Generate MA signal and prediction based on indicators."""
    # Defaults
    ma_signal = "N/A"
    prediction = "N/A"
    
    # Check if all necessary data is available (not None and has recent values)
    if rsi is not None and ma50 is not None and ma200 is not None and not rsi.empty and not ma50.empty and not ma200.empty:
        current_rsi = rsi.iloc[-1]
        ma50_last = ma50.iloc[-1]
        ma200_last = ma200.iloc[-1]
        
        # Check if recent values are valid numbers
        if not pd.isna(ma50_last) and not pd.isna(ma200_last):
            ma50_above_ma200 = ma50_last > ma200_last
            ma_signal = "Bullish" if ma50_above_ma200 else "Bearish"
            
            if not pd.isna(current_rsi):
                prediction = "Hold" # Reset default
                if current_rsi < 30 and ma50_above_ma200:
                    prediction = "Strong Buy"
                elif current_rsi < 40 and ma50_above_ma200:
                    prediction = "Buy"
                elif current_rsi > 70 and not ma50_above_ma200:
                    prediction = "Strong Sell"
                elif current_rsi > 60 and not ma50_above_ma200:
                    prediction = "Sell"
                elif ma50_above_ma200 and current_rsi <= 55:
                    prediction = "Consider Buy/Hold"
                elif not ma50_above_ma200 and current_rsi >= 45:
                    prediction = "Consider Sell/Hold"
            else:
                prediction = "Hold (RSI N/A)" # RSI is NaN
        else:
             prediction = "Hold (MA N/A)" # MAs are NaN
    
    return ma_signal, prediction

# --- Main Calculation Function ---

def calculate_insights(df: pd.DataFrame, index_name: str) -> Dict[str, Any]:
    """Calculate key insights by calling helper functions."""
    insights = {"name": index_name}
    
    # Basic checks
    if df.empty or len(df) < 2:
        print(f"Warning: Insufficient data for {index_name}. Returning basic info.")
        insights["latest_date"] = "N/A"
        insights["latest_close"] = 0
        # ... add other defaults as needed ...
        return insights

    # Add basic info
    insights["latest_date"] = pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d')
    insights["latest_close"] = df['close'].iloc[-1]
    insights["current_price"] = insights["latest_close"]
    insights["year_high"] = df['high'].max()
    insights["year_low"] = df['low'].min()
    insights["latest_volume"] = df['volume'].iloc[-1]
    insights["avg_volume"] = df['volume'].mean(skipna=True)

    # Calculate indicators using helpers
    df['daily_return'] = _calculate_returns(df)
    volatility_series = _calculate_volatility(df['daily_return'])
    df['volatility'] = volatility_series if volatility_series is not None else np.nan
    
    ma50_series, ma200_series = _calculate_moving_averages(df)
    df['ma50'] = ma50_series if ma50_series is not None else np.nan
    df['ma200'] = ma200_series if ma200_series is not None else np.nan
    
    rsi_series = _calculate_rsi(df)
    df['rsi'] = rsi_series if rsi_series is not None else np.nan
    
    ma_signal, prediction = _generate_prediction(rsi_series, ma50_series, ma200_series)
    insights["ma_signal"] = ma_signal
    insights["prediction_signal"] = prediction
    insights["current_rsi"] = rsi_series.iloc[-1] if rsi_series is not None and not rsi_series.empty else np.nan
    insights["current_volatility"] = volatility_series.iloc[-1] if volatility_series is not None and not volatility_series.empty else np.nan

    # Calculate returns and other stats
    insights["max_daily_gain"] = df['daily_return'].max()
    insights["max_daily_loss"] = df['daily_return'].min()
    insights["avg_daily_return"] = df['daily_return'].mean()
    insights["avg_volatility"] = df['volatility'].mean(skipna=True)

    start_close = df['close'].iloc[0]
    end_close = df['close'].iloc[-1]
    insights["total_return"] = ((end_close / start_close) - 1) * 100 if start_close != 0 else 0
    
    start_date = pd.to_datetime(df['date'].iloc[0])
    end_date = pd.to_datetime(df['date'].iloc[-1])
    days_diff = (end_date - start_date).days
    annual_return = 0
    if days_diff > 0 and start_close != 0 and not pd.isna(start_close) and not pd.isna(end_close):
        total_return_ratio = end_close / start_close
        if total_return_ratio > 0: 
            annual_return = ((total_return_ratio ** (365.0 / days_diff)) - 1) * 100
        else: 
             annual_return = -100.0
    insights["annual_return"] = annual_return

    # Final stats and correlations
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'daily_return', 'volatility', 'ma50', 'ma200', 'rsi']
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    insights["summary_stats"] = df[valid_numeric_cols].describe()
    insights["correlations"] = df[['close', 'volume', 'daily_return', 'volatility']].corr()

    # Final cleanup of NaN floats in insights dict
    for key, value in insights.items():
        if isinstance(value, (float, np.floating)) and pd.isna(value):
            insights[key] = 0 
            
    return insights

# --- Visualization & Reporting --- (Largely unchanged, but use insights dict correctly)

def create_overlay_visualization(all_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]]) -> str:
    """Create an overlay plot of normalized index prices."""
    plt.style.use('default')
    plt.figure(figsize=(12, 6))
    
    all_plot_data_available = True
    for index_key, (df, insights, _) in all_data.items():
        if 'close' not in df.columns or df['close'].isnull().all() or len(df) == 0:
             print(f"Warning: Skipping {insights.get('name', index_key)} in overlay plot due to missing/invalid close data.")
             all_plot_data_available = False
             continue
        # Normalize price: (price / first_price) * 100
        first_price = df['close'].iloc[0]
        if first_price == 0: # Avoid division by zero
             print(f"Warning: Skipping {insights.get('name', index_key)} in overlay plot due to zero start price.")
             all_plot_data_available = False
             continue
        normalized_price = (df['close'] / first_price) * 100
        plt.plot(df['date'], normalized_price, label=insights.get("name", index_key), alpha=0.8)

    if not all_plot_data_available and plt.gca().lines == []: # Check if any lines were actually plotted
        plt.close() # Close the empty figure
        return "" # Return empty string if no data could be plotted
        
    plt.title('Normalized Market Index Performance (Base 100)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Start = 100)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    return fig_to_base64()

def create_visualizations(df: pd.DataFrame, index_name: str) -> Dict[str, Any]:
    """Create visualizations for a given market index."""
    figures = {}
    plt.style.use('default')
    
    # Price chart
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['close'], label=f'{index_name} Close', alpha=0.7)
    if 'ma50' in df.columns: plt.plot(df['date'], df['ma50'], label='50-day MA', alpha=0.8)
    if 'ma200' in df.columns: plt.plot(df['date'], df['ma200'], label='200-day MA', alpha=0.8)
    plt.title(f'{index_name} Price & Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    figures['price_chart'] = fig_to_base64()
    
    # Daily returns
    if 'daily_return' in df.columns:
        plt.figure(figsize=(10, 4))
        plt.bar(df['date'], df['daily_return'], alpha=0.7)
        plt.title(f'{index_name} Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
        figures['returns_chart'] = fig_to_base64()

    # Volatility
    if 'volatility' in df.columns and not df['volatility'].isnull().all():
        plt.figure(figsize=(10, 4))
        plt.plot(df['date'], df['volatility'], color='red', alpha=0.7)
        plt.title(f'{index_name} Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True, alpha=0.3)
        figures['volatility_chart'] = fig_to_base64()

    # RSI
    if 'rsi' in df.columns and not df['rsi'].isnull().all():
        plt.figure(figsize=(10, 4))
        plt.plot(df['date'], df['rsi'], color='purple', alpha=0.7)
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        plt.title(f'{index_name} RSI (14-day)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.ylim(0, 100) # RSI ranges from 0 to 100
        plt.legend()
        plt.grid(True, alpha=0.3)
        figures['rsi_chart'] = fig_to_base64()
    
    return figures

def fig_to_base64() -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=75) 
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_combined_prediction(all_insights: Dict[str, Dict[str, Any]]) -> str:
    """Generate a combined prediction signal."""
    signals = [insights.get("prediction_signal", "N/A") for insights in all_insights.values()]
    # Count signals, excluding N/A
    valid_signals = [s for s in signals if s != "N/A"]
    if not valid_signals:
        return "Market Signal N/A (Insufficient Data)"
        
    buy_signals = sum(1 for s in valid_signals if "Buy" in s) 
    sell_signals = sum(1 for s in valid_signals if "Sell" in s)
    hold_signals = valid_signals.count("Hold")
    
    total_valid = len(valid_signals)
    if buy_signals == total_valid: 
        return "Strong Market Buy Signal"
    elif sell_signals == total_valid:
        return "Strong Market Sell Signal"
    elif buy_signals > sell_signals:
        return "Overall Market Leaning Bullish (Buy/Hold)"
    elif sell_signals > buy_signals:
        return "Overall Market Leaning Bearish (Sell/Hold)"
    else: # Equal buy/sell or mostly holds
        return "Mixed Market Signals (Hold)"

def generate_html_report(all_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]]) -> str:
    """Generate a consolidated HTML report."""
    report_date = datetime.now().strftime("%Y-%m-%d")
    combined_prediction = get_combined_prediction({k: v[1] for k, v in all_data.items()})
    overlay_plot_base64 = create_overlay_visualization(all_data)
    
    # Start HTML (simplified style for brevity)
    html = f"""
    <!DOCTYPE html><html><head><title>Market Index Analysis</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }} 
        .container {{ max-width: 1200px; margin: auto; }} 
        .header, .disclaimer {{ text-align: center; margin-bottom: 20px; }}
        .summary-section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; background-color: #f8f9fa; }}
        .index-section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }} 
        h1, h2, h3 {{ color: #333; }} 
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 10px; font-size: 0.9em; }} 
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }} 
        th {{ background-color: #f0f0f0; }} 
        .plot img {{ max-width: 100%; height: auto; margin-top: 10px; }} 
        .prediction-card {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; border-radius: 5px; margin-bottom: 15px; }} 
        .prediction-signal {{ font-size: 1.3em; font-weight: bold; }}
        .metric-card {{ display: inline-block; margin: 5px; padding: 8px; background-color: #eee; border-radius: 4px; text-align: center; min-width: 120px; }}
        .metric-value {{ font-weight: bold; }}
        .positive {{ color: green; }} .negative {{ color: red; }} .neutral {{ color: #555; }}
    </style></head><body><div class="container">
    <div class="header"><h1>Market Index Analysis Report</h1><p>Generated: {report_date}</p></div>
    
    <div class="summary-section">
        <h2>Overall Market Summary</h2>
        <div class="prediction-card"><p class="prediction-signal">Overall Signal: {combined_prediction}</p></div>
        <div class="plot"><img src="data:image/png;base64,{overlay_plot_base64}" alt="Normalized Index Performance"></div>
    </div>
    
    """

    # Loop through each index
    for index_key, (df, insights, visualizations) in all_data.items():
        index_name = insights.get("name", index_key)
        signal = insights.get("prediction_signal", "N/A")
        signal_class = 'positive' if 'Buy' in signal else ('negative' if 'Sell' in signal else 'neutral')
        latest_date = insights.get("latest_date", "N/A")
        
        html += f'<div class="index-section"><h2>{index_name} ({latest_date})</h2>'
        html += f'<div style="text-align: center; margin-bottom: 15px;">' # Metrics container
        # Key Metrics Mini-Cards
        html += f'<span class="metric-card">Close: <span class="metric-value">${insights.get("latest_close", 0):.2f}</span></span>'
        html += f'<span class="metric-card">Annual Rtn: <span class="metric-value {('positive' if insights.get("annual_return", 0) > 0 else 'negative') if insights.get("annual_return", 0) != 0 else 'neutral'}">{insights.get("annual_return", 0):.1f}%</span></span>'
        html += f'<span class="metric-card">Volatility: <span class="metric-value">{insights.get("current_volatility", 0):.2f}</span></span>'
        html += f'<span class="metric-card">Signal: <span class="metric-value {signal_class}">{signal}</span></span>'
        html += '</div>' # End metrics container

        # Simple Table for other stats
        html += '<table>'
        html += f'<tr><th>Total Return (Period)</th><td>{insights.get("total_return", 0):.2f}%</td><th>Avg Daily Rtn</th><td>{insights.get("avg_daily_return", 0):.3f}%</td></tr>'
        html += f'<tr><th>Max Daily Gain</th><td>{insights.get("max_daily_gain", 0):.2f}%</td><th>Max Daily Loss</th><td>{insights.get("max_daily_loss", 0):.2f}%</td></tr>'
        html += f'<tr><th>RSI</th><td>{insights.get("current_rsi", 0):.1f}</td><th>MA Signal</th><td>{insights.get("ma_signal", "N/A")}</td></tr>'
        html += '</table>'

        # Visualizations
        if 'price_chart' in visualizations: html += f'<div class="plot"><img src="data:image/png;base64,{visualizations["price_chart"]}" alt="Price"></div>'
        if 'returns_chart' in visualizations: html += f'<div class="plot"><img src="data:image/png;base64,{visualizations["returns_chart"]}" alt="Returns"></div>'
        if 'volatility_chart' in visualizations: html += f'<div class="plot"><img src="data:image/png;base64,{visualizations["volatility_chart"]}" alt="Volatility"></div>'
        if 'rsi_chart' in visualizations: html += f'<div class="plot"><img src="data:image/png;base64,{visualizations["rsi_chart"]}" alt="RSI"></div>'

        html += '</div>' # Close index-section

    # Close HTML
    html += '<p class="disclaimer">Disclaimer: For informational purposes only. Not financial advice.</p></div></body></html>'
    return html

# --- Main Execution --- 

def main():
    """Main function to load data, run analysis, and generate report."""
    all_data = {}
    successful_indices = []

    for index_key in INDICES:
        try:
            print(f"Processing {INDICES[index_key]['name']}...")
            df = load_index_data(index_key)
            # Pass a fresh copy to avoid side effects between index calculations if df is modified
            insights = calculate_insights(df.copy(), INDICES[index_key]['name']) 
            # Pass the original df (or the copy if insights didn't modify it) to visualizations
            visualizations = create_visualizations(df, INDICES[index_key]['name'])
            all_data[index_key] = (df, insights, visualizations)
            successful_indices.append(INDICES[index_key]['name'])
            print(f"Successfully processed {INDICES[index_key]['name']}.")
        except FileNotFoundError as e:
            print(f"Error processing {INDICES[index_key]['name']}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred processing {INDICES[index_key]['name']}: {e}")
            # Consider adding traceback print here for better debugging
            # import traceback
            # traceback.print_exc()

    if not all_data:
        print("No index data could be processed. Exiting.")
        return

    print("\nGenerating consolidated report...")
    html_report = generate_html_report(all_data)
    
    output_path = Path("out/market_indices_report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_report)
    print(f"Consolidated report: {output_path} (Indices: {', '.join(successful_indices)})")

if __name__ == "__main__":
    main() 