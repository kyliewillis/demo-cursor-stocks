"""Setup configuration for data science template."""

from setuptools import setup, find_packages

setup(
    name="data_science_template",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "yfinance",
        "jinja2",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    python_requires=">=3.8",
) 