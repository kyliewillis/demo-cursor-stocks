"""Tests for the data loader module."""

import pytest
import pandas as pd
import numpy as np
from data_science_template.data.data_loader import load_csv, load_excel, load_numpy


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    filepath = tmp_path / "test.csv"
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_excel(tmp_path):
    """Create a sample Excel file for testing."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    return filepath


@pytest.fixture
def sample_numpy(tmp_path):
    """Create a sample NumPy file for testing."""
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    filepath = tmp_path / "test.npy"
    np.save(filepath, arr)
    return filepath


def test_load_csv(sample_csv):
    """Test loading CSV file."""
    df = load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B']


def test_load_excel(sample_excel):
    """Test loading Excel file."""
    df = load_excel(sample_excel)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B']


def test_load_numpy(sample_numpy):
    """Test loading NumPy file."""
    arr = load_numpy(sample_numpy)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)


def test_load_csv_file_not_found():
    """Test loading non-existent CSV file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent.csv")


def test_load_excel_file_not_found():
    """Test loading non-existent Excel file."""
    with pytest.raises(FileNotFoundError):
        load_excel("nonexistent.xlsx")


def test_load_numpy_file_not_found():
    """Test loading non-existent NumPy file."""
    with pytest.raises(FileNotFoundError):
        load_numpy("nonexistent.npy") 