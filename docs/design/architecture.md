# Weather Analysis Project Architecture

## Overview
This document outlines the architectural decisions and patterns used in the Burlington Weather Analysis project.

## Directory Structure
The project follows a modular structure designed for weather data analysis and reporting:

- `src/data_science_template/`: Core analysis code
  - `main.py`: Main script for report generation
  - `fetch_weather_data.py`: Data collection from Open-Meteo API
- `data/`: Data storage
  - `raw/`: Raw weather data CSV files
- `out/`: Generated reports and visualizations
- `docs/`: Project documentation
- `tests/`: Test files

## Code Organization

### Source Code (`src/data_science_template/`)
- `main.py`: Handles data analysis and report generation
  - Data loading and preprocessing
  - Statistical analysis
  - Visualization generation
  - HTML report creation
- `fetch_weather_data.py`: Manages data collection
  - API interaction with Open-Meteo
  - Data validation and storage
  - Error handling and retries

### Data Management
- Raw weather data stored in `data/raw/`
- Data format: CSV with columns for temperature, humidity, pressure, wind speed, and rainfall
- Data validation on loading
- Error handling for missing or invalid data

### Report Generation
- HTML reports generated in `out/`
- Static matplotlib visualizations
- Summary statistics and key insights
- Clean, responsive design

## Development Guidelines

### Code Quality
- PEP 8 compliance enforced through Black and Flake8
- Type hints required for all functions
- Unit tests for data loading and processing
- Documentation required for all public functions

### Documentation
- All code changes require documentation updates
- API documentation using docstrings (Google style)
- Design decisions documented in `docs/design/`
- Requirements documented in `docs/requirements/`

### Version Control
- Feature branch workflow
- Semantic versioning
- Conventional commits
- Pull request reviews required

## Dependencies
Core dependencies:
- pandas: Data manipulation and analysis
- matplotlib: Static visualizations
- seaborn: Statistical visualizations
- python-dotenv: Environment variable management

## Future Considerations
- Additional weather data sources
- More advanced visualizations
- Automated report scheduling
- Historical data comparison
- API rate limiting and caching 