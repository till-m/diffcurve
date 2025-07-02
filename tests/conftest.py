#!/usr/bin/env python3
"""
Pytest configuration and fixtures for curvelet transform tests.
"""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def test_image_64x64():
    """Generate a reproducible 64x64 test image."""
    np.random.seed(42)
    return np.random.randn(64, 64)


@pytest.fixture
def test_image_128x128():
    """Generate a reproducible 128x128 test image.""" 
    np.random.seed(123)
    return np.random.randn(128, 128)


@pytest.fixture
def test_data_dir():
    """Path to test reference data directory."""
    return Path(__file__).parent / "reference_data"


@pytest.fixture
def matlab_tolerance():
    """Tolerance for comparing against MATLAB results."""
    return 1e-10


@pytest.fixture
def python_tolerance():
    """Tolerance for Python-only tests."""
    return 1e-12


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "matlab: tests that require MATLAB reference data"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "memory: tests that require significant memory"
    )