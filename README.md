# Data Science Project Template

This repository provides a structured template for data science projects, following best practices for Python development and Jupyter notebooks.

## Project Structure

```
.
├── data/               # Data files (raw and processed)
├── docs/              # Documentation
│   ├── design/        # Design specifications
│   └── requirements/  # Requirements specifications
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
├── tests/            # Test files
├── .gitignore        # Git ignore rules
├── requirements.txt  # Project dependencies
└── setup.py         # Package setup file
```

## Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Code Quality

This project uses:
- Black for code formatting
- Flake8 for linting
- Pylint for code analysis
- nbformat for notebook formatting

## Documentation

All code changes require corresponding documentation updates in the `docs/` directory. This includes:
- Design specifications
- Requirements specifications
- API documentation
- Usage examples

## Git Workflow

1. Create a new branch for each feature
2. Follow PEP 8 style guide
3. Update documentation with code changes
4. Submit pull requests for review

## License

[Add your license here] 