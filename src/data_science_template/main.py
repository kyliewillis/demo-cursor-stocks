"""Main script for market indices analysis and report generation."""

import os
from typing import Dict, Optional, Tuple

import pandas as pd
from .data_fetcher import DataFetcher
from .analyzer import MarketIndexAnalyzer
from .reporter import ReportGenerator
from .predictor import MarketPredictor


def load_data(index_name: str, base_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load market data from CSV file.

    Args:
        index_name: Name of the market index.
        base_dir: Optional base directory path. If not provided, uses default data directory.

    Returns:
        DataFrame containing market data or None if file not found.
    """
    if base_dir is None:
        base_dir = "data/raw"
    file_path = os.path.join(base_dir, f"{index_name.lower()}_data.csv")
    try:
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        print(f"Data file not found for {index_name}")
        return None


def process_index(df: pd.DataFrame, index_name: str) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Process market data for a single index.
    
    Args:
        df: DataFrame with market data
        index_name: Name of the index
        
    Returns:
        Tuple of (insights, modified DataFrame, prediction insights)
    """
    try:
        # Initialize components
        analyzer = MarketIndexAnalyzer(df, index_name)
        predictor = MarketPredictor()
        
        # Train predictor
        predictor.train(df)
        
        # Get predictions
        predictions = predictor.predict(df)
        print(f"\nPredictions for {index_name}:")
        print(f"Buy Probability: {predictions['buy_probability']:.2f}%")
        print(f"Model Confidence: {predictions['model_confidence']:.2f}%")
        
        # Get insights
        insights = analyzer.calculate_all_insights()
        
        return insights, analyzer.df, predictions
    except Exception as e:
        print(f"Error processing {index_name}: {str(e)}")
        return {}, df, None


def main() -> None:
    """Main function to orchestrate market data analysis and report generation."""
    # Initialize components
    data_fetcher = DataFetcher()
    report_generator = ReportGenerator()

    # Fetch latest data
    print("Fetching latest market data...")
    data_fetcher.fetch_and_save_all()

    # Process each index
    indices = ["SP500", "DOW", "NASDAQ"]
    all_data = {}
    all_insights = {}
    all_visualizations = {}
    all_predictions = {}

    for index_name in indices:
        print(f"\nProcessing {index_name}...")
        df = load_data(index_name)
        if df is None:
            continue

        result = process_index(df, index_name)
        if result is None:
            continue
            
        insights, df, predictions = result  # Unpack the tuple

        # Create visualizations
        visualizations = report_generator.create_visualizations(df, insights, index_name)

        # Store results
        all_data[index_name] = df
        all_insights[index_name] = insights
        all_visualizations[index_name] = visualizations
        all_predictions[index_name] = predictions

    if not all_data:
        print("No index data could be processed. Exiting.")
        return

    # Generate report
    print("\nGenerating HTML report...")
    html_path, pdf_path = report_generator.generate_html_report(
        all_data, 
        all_insights, 
        all_visualizations,
        all_predictions
    )
    print(f"Reports generated successfully:")
    print(f"- HTML Report: {html_path}")
    print(f"- PDF Report: {pdf_path}")


if __name__ == "__main__":
    main() 