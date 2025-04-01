# Market Index Analysis Project Architecture

## Overview
This document outlines the architectural design of the Market Index Analysis project, which provides comprehensive analysis and visualization of major market indices using technical indicators and modern visualization techniques.

## Directory Structure
```
project_root/
├── src/data_science_template/    # Core analysis code
│   ├── main.py                  # Main orchestration script
│   ├── analyzer.py              # Market analysis logic
│   ├── data_fetcher.py          # Data collection from Yahoo Finance
│   ├── reporter.py              # Report generation
│   ├── templates/               # HTML templates
│   │   └── report_template.html # Main report template
│   └── tests/                   # Unit tests
├── data/                        # Data storage
│   └── raw/                     # Raw market data CSV files
├── out/                         # Generated reports
├── docs/                        # Documentation
│   ├── requirements/            # Project requirements
│   └── design/                  # Design documentation
└── tests/                       # Integration tests
```

## Component Architecture

### 1. Data Management (`data_fetcher.py`)
- Responsibilities:
  - Fetch market data from Yahoo Finance
  - Data validation and standardization
  - CSV storage management
- Key Components:
  - `DataFetcher` class:
    - Index configuration
    - Data fetching logic
    - Error handling
    - Data storage management

### 2. Market Analysis (`analyzer.py`)
- Responsibilities:
  - Technical analysis calculations
  - Signal generation
  - Performance metrics
- Key Components:
  - `MarketIndexAnalyzer` class:
    - Moving average calculations
    - RSI computation
    - Volatility analysis
    - Return calculations
    - Signal generation
    - Performance metrics

### 3. Report Generation (`reporter.py`)
- Responsibilities:
  - Visualization creation
  - HTML report generation
  - Template management
- Key Components:
  - `ReportGenerator` class:
    - Chart generation
    - HTML templating
    - Signal aggregation
    - Report styling

### 4. Main Orchestration (`main.py`)
- Responsibilities:
  - Component coordination
  - Error handling
  - Process flow management
- Key Functions:
  - Data loading
  - Analysis orchestration
  - Report generation
  - Error management

## Data Flow

1. Data Collection:
   - Yahoo Finance API → Raw CSV files
   - Data validation and standardization
   - Storage in `data/raw/`

2. Analysis Pipeline:
   - Raw data → MarketIndexAnalyzer
   - Technical calculations
   - Signal generation
   - Performance metrics

3. Report Generation:
   - Analysis results → ReportGenerator
   - Visualization creation
   - HTML report assembly
   - Output to `out/` directory

## Technical Implementation

### 1. Data Structures
- Market Data DataFrame:
  ```python
  {
      'date': datetime64,
      'open': float64,
      'high': float64,
      'low': float64,
      'close': float64,
      'adj_close': float64,
      'volume': int64,
      'daily_return': float64,
      'volatility': float64,
      'ma20': float64,
      'ma50': float64,
      'ma200': float64,
      'rsi': float64
  }
  ```

- Analysis Insights:
  ```python
  {
      'name': str,
      'latest_close': float,
      'moving_averages': dict,
      'current_rsi': float,
      'volatility': float,
      'predictions': {
          'recommendation': str,
          'signal_strength': str,
          'combined_signal': float
      }
  }
  ```

### 2. Key Algorithms
- Moving Averages:
  - Simple moving averages (20, 50, 200 days)
  - Crossover detection
- RSI Calculation:
  - 14-day period
  - Smoothed averages
- Signal Generation:
  - Weighted technical indicators
  - Combined signal strength
- Volatility:
  - Rolling standard deviation
  - Annualized calculation

### 3. External Dependencies
- Core Libraries:
  - pandas: Data manipulation
  - numpy: Numerical computations
  - matplotlib/seaborn: Visualization
  - yfinance: Market data
  - jinja2: HTML templating

## Development Guidelines

### 1. Code Quality
- Style Guide:
  - Black formatting (88 chars)
  - Google docstring style
  - Type hints
- Testing:
  - Unit tests (pytest)
  - Integration tests
  - Test coverage tracking

### 2. Error Handling
- Graceful degradation
- Comprehensive logging
- User-friendly error messages
- API retry mechanisms

### 3. Performance
- Efficient data processing
- Memory optimization
- Caching where appropriate
- Responsive visualizations

## Future Considerations
- Real-time data updates
- Additional technical indicators
- Machine learning integration
- Advanced visualization options
- Performance optimizations
- API rate limiting 