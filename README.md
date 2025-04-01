# Market Index Analysis (S&P 500, Dow Jones, Nasdaq)

This project analyzes historical data for major market indices (S&P 500, Dow Jones, Nasdaq), generating insights on market performance, volatility, and technical indicators. The analysis includes price trends, returns, volume patterns, and predictive technical indicators like RSI and moving averages for each index, culminating in a consolidated report with an overall market signal.

## Project Structure

- `src/data_science_template/`
  - `main.py`: Main script for generating the consolidated market analysis report
  - `fetch_market_data.py`: Script to fetch S&P 500, Dow Jones, and Nasdaq data from Yahoo Finance
- `data/raw/`: Contains the raw index data CSV files (e.g., `sp500_data.csv`, `dow_data.csv`)
- `out/`: Contains the generated HTML report (`market_indices_report.html`)
- `tests/`: Contains test files for data loading and processing
- `docs/`: Contains project documentation
  - `design/`: Design specifications
  - `requirements/`: Requirements specifications

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Fetch Market Index Data:
```bash
python src/data_science_template/fetch_market_data.py
```
This will download 5 years of historical data for S&P 500, Dow Jones, and Nasdaq, saving each to a separate CSV file in `data/raw/`.

2. Generate the Analysis Report:
```bash
python src/data_science_template/main.py
```

The consolidated report will be generated in the `out/` directory as `market_indices_report.html`. The report includes:
- An overall market signal based on combined index analysis.
- Individual sections for S&P 500, Dow Jones, and Nasdaq, each containing:
  - Price performance with moving averages
  - Key statistics (returns, volatility, highs/lows)
  - Technical indicators (RSI, Moving Average signals)
  - Individual Buy/Sell/Hold signal
  - Returns and Volatility charts

## Analysis Features

The project calculates and visualizes several key metrics for each index:

- **Price Performance**: Historical price with 50-day and 200-day moving averages
- **Returns Analysis**: Daily returns and cumulative performance
- **Volatility Analysis**: 20-day rolling standard deviation of returns
- **Technical Indicators**: Relative Strength Index (RSI) and Moving Average crossovers
- **Trading Volume**: Volume patterns and anomalies (where available)
- **Combined Signal**: An overall market sentiment signal derived from individual index signals.

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Documentation

Project documentation can be found in the `docs/` directory, including:
- Design specifications in `docs/design/`
- Requirements specifications in `docs/requirements/` 