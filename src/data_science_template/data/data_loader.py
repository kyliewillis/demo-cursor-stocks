"""Data loading utilities for the data science template.

This module provides utilities for loading and preprocessing data in various formats.
It includes functions for handling CSV, Excel, and other common data formats.
"""

from pathlib import Path
from typing import Union, Optional

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike


def load_csv(
    filepath: Union[str, Path],
    **kwargs: dict
) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        filepath: Path to the CSV file.
        **kwargs: Additional arguments to pass to pandas.read_csv().

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return pd.read_csv(filepath, **kwargs)


def load_excel(
    filepath: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
    **kwargs: dict
) -> pd.DataFrame:
    """Load data from an Excel file.

    Args:
        filepath: Path to the Excel file.
        sheet_name: Name or index of the sheet to load.
        **kwargs: Additional arguments to pass to pandas.read_excel().

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the specified sheet is not found.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)


def load_numpy(
    filepath: Union[str, Path],
    **kwargs: dict
) -> ArrayLike:
    """Load data from a NumPy file.

    Args:
        filepath: Path to the NumPy file.
        **kwargs: Additional arguments to pass to numpy.load().

    Returns:
        ArrayLike: Loaded data as a NumPy array.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be loaded as a NumPy array.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return np.load(filepath, **kwargs) 