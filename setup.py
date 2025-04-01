from setuptools import setup, find_packages

setup(
    name="sp500-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas-datareader>=0.10.0",
        "yfinance>=0.1.70",
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
    description="S&P 500 data analysis and reporting tool",
    author="Demo Project",
) 