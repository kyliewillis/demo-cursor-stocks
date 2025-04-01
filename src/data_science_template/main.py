"""Main script for market indices analysis and report generation."""

import os
from typing import Dict, Optional

import pandas as pd
from data_fetcher import DataFetcher
from analyzer import MarketIndexAnalyzer
from reporter import ReportGenerator


def load_data(index_name: str) -> Optional[pd.DataFrame]:
    """Load market data from CSV file.

    Args:
        index_name: Name of the market index.

    Returns:
        DataFrame containing market data or None if file not found.
    """
    file_path = f"data/raw/{index_name.lower()}_data.csv"
    try:
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        print(f"Data file not found for {index_name}")
        return None


def process_index(df: pd.DataFrame, index_name: str) -> Optional[tuple[Dict, pd.DataFrame]]:
    """Process market data for a single index.

    Args:
        df: DataFrame containing market data.
        index_name: Name of the market index.

    Returns:
        Tuple containing (insights dict, modified DataFrame) or None if processing fails.
    """
    try:
        analyzer = MarketIndexAnalyzer(df, index_name)
        insights = analyzer.calculate_all_insights()
        return insights, analyzer.df
    except Exception as e:
        print(f"An unexpected error occurred processing {index_name}: {str(e)}")
        return None


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

    for index_name in indices:
        print(f"\nProcessing {index_name}...")
        df = load_data(index_name)
        if df is None:
            continue

        result = process_index(df, index_name)
        if result is None:
            continue
            
        insights, df = result  # Unpack the tuple

        # Create visualizations
        visualizations = report_generator.create_visualizations(df, insights, index_name)

        # Store results
        all_data[index_name] = df
        all_insights[index_name] = insights
        all_visualizations[index_name] = visualizations

    if not all_data:
        print("No index data could be processed. Exiting.")
        return

    # Generate report
    print("\nGenerating HTML report...")
    report_generator.generate_html_report(all_data, all_insights, all_visualizations)
    print(f"Report generated successfully: {report_generator.output_dir}/market_indices_report.html")


if __name__ == "__main__":
    main() 