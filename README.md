# S&P 500 Market Analysis

This project analyzes historical S&P 500 data, generating insights on market performance, volatility, and technical indicators. The analysis includes price trends, returns, volume patterns, and predictive technical indicators like RSI and moving averages.

## Project Structure

- `src/data_science_template/`
  - `main.py`: Main script for generating the S&P 500 analysis report
  - `fetch_sp500_data.py`: Script to fetch S&P 500 data from Yahoo Finance
- `data/raw/`: Contains the raw S&P 500 data CSV file
- `out/`: Contains the generated HTML report
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
pip install -e .
```

## Usage

1. Fetch S&P 500 data:
```bash
python src/data_science_template/fetch_sp500_data.py
```
This will download 5 years of S&P 500 historical data and save it to `data/raw/sp500_data.csv`.

2. Generate the analysis report:
```bash
python src/data_science_template/main.py
```

The report will be generated in the `out/` directory as `sp500_report.html`. The report includes:
- Price performance with moving averages
- Returns analysis
- Volatility metrics
- Volume trends
- Technical indicators (RSI, Moving Average signals)
- Correlation analysis

## Analysis Features

The project calculates and visualizes several key metrics:

- **Price Performance**: Historical price with 50-day and 200-day moving averages
- **Returns Analysis**: Daily returns and cumulative performance
- **Volatility Analysis**: 20-day rolling standard deviation of returns
- **Technical Indicators**: Relative Strength Index (RSI) and Moving Average crossovers
- **Trading Volume**: Volume patterns and anomalies
- **Correlation Analysis**: Relationships between price, volume, returns, and volatility

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Documentation

Project documentation can be found in the `docs/` directory, including:
- Design specifications in `docs/design/`
- Requirements specifications in `docs/requirements/` 