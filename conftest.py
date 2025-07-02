#!/usr/bin/env python3
"""
Pytest configuration for curvelet transform tests
"""

import pytest
import matlab.engine

@pytest.fixture(scope="session")
def matlab_engine_session():
    """Session-scoped MATLAB engine to avoid repeated startup/shutdown"""
    try:
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        yield eng
        eng.quit()
    except Exception as e:
        pytest.skip(f"MATLAB engine not available: {e}")

@pytest.fixture(scope="class")
def matlab_engine_class(matlab_engine_session):
    """Class-scoped MATLAB engine fixture"""
    return matlab_engine_session

# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "matlab: mark test as requiring MATLAB engine"
    )
    config.addinivalue_line(
        "markers", "machine_precision: mark test as requiring machine precision"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as testing edge cases"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as preventing regression"
    )

def pytest_runtest_setup(item):
    """Setup for individual test items"""
    # Skip MATLAB tests if no MATLAB marker and engine not available
    if "matlab" in item.keywords:
        try:
            matlab.engine.start_matlab()
        except Exception:
            pytest.skip("MATLAB engine not available")