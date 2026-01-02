"""
Pytest configuration and fixtures.

Handles missing optional dependencies and provides test fixtures.
"""

import sys
import logging
from unittest.mock import MagicMock

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

# Mock MetaTrader5 module if not available (non-Windows platforms)
if 'MetaTrader5' not in sys.modules:
    mock_mt5 = MagicMock()

    # Set common constants
    mock_mt5.TIMEFRAME_M1 = 1
    mock_mt5.TIMEFRAME_M5 = 5
    mock_mt5.TIMEFRAME_M15 = 15
    mock_mt5.TIMEFRAME_M30 = 30
    mock_mt5.TIMEFRAME_H1 = 60
    mock_mt5.TIMEFRAME_H4 = 240
    mock_mt5.TIMEFRAME_D1 = 1440
    mock_mt5.TIMEFRAME_W1 = 10080
    mock_mt5.TIMEFRAME_MN1 = 43200

    mock_mt5.TRADE_ACTION_DEAL = 1
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.TRADE_ACTION_MODIFY = 7
    mock_mt5.TRADE_ACTION_REMOVE = 8

    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.ORDER_TYPE_BUY_STOP = 4
    mock_mt5.ORDER_TYPE_SELL_STOP = 5

    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_TIME_DAY = 1
    mock_mt5.ORDER_TIME_SPECIFIED = 2

    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_RETURN = 2

    mock_mt5.POSITION_TYPE_BUY = 0
    mock_mt5.POSITION_TYPE_SELL = 1

    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.TRADE_RETCODE_REQUOTE = 10004
    mock_mt5.TRADE_RETCODE_INVALID = 10013
    mock_mt5.TRADE_RETCODE_INVALID_VOLUME = 10014
    mock_mt5.TRADE_RETCODE_ERROR = 10006

    # Mock common functions
    mock_mt5.initialize.return_value = True
    mock_mt5.login.return_value = True
    mock_mt5.shutdown.return_value = None
    mock_mt5.last_error.return_value = (0, "No error")
    mock_mt5.version.return_value = (5, 0, 0)

    # Insert mock into sys.modules
    sys.modules['MetaTrader5'] = mock_mt5


# Add project root to path
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Check for optional dependencies
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    import statsmodels
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# Fixtures
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_aiohttp: mark test as requiring aiohttp"
    )
    config.addinivalue_line(
        "markers", "requires_tensorflow: mark test as requiring tensorflow"
    )
    config.addinivalue_line(
        "markers", "requires_arch: mark test as requiring arch"
    )
    config.addinivalue_line(
        "markers", "requires_statsmodels: mark test as requiring statsmodels"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that require missing dependencies."""

    if not HAS_AIOHTTP:
        skip_aiohttp = pytest.mark.skip(reason="aiohttp not installed")
        for item in items:
            if "requires_aiohttp" in item.keywords:
                item.add_marker(skip_aiohttp)

    if not HAS_TENSORFLOW:
        skip_tf = pytest.mark.skip(reason="tensorflow not installed")
        for item in items:
            if "requires_tensorflow" in item.keywords:
                item.add_marker(skip_tf)

    if not HAS_ARCH:
        skip_arch = pytest.mark.skip(reason="arch not installed")
        for item in items:
            if "requires_arch" in item.keywords:
                item.add_marker(skip_arch)

    if not HAS_STATSMODELS:
        skip_sm = pytest.mark.skip(reason="statsmodels not installed")
        for item in items:
            if "requires_statsmodels" in item.keywords:
                item.add_marker(skip_sm)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = 200

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.uniform(1000, 10000, n)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def tmp_state_file(tmp_path):
    """Provide temporary file for risk state persistence."""
    return str(tmp_path / "risk_state.json")


# Utility function for tests to check dependencies
def requires_optional_dep(dep_name: str):
    """Decorator to skip tests requiring optional dependencies."""
    deps = {
        'aiohttp': HAS_AIOHTTP,
        'tensorflow': HAS_TENSORFLOW,
        'arch': HAS_ARCH,
        'statsmodels': HAS_STATSMODELS,
    }

    available = deps.get(dep_name, False)
    return pytest.mark.skipif(
        not available,
        reason=f"{dep_name} not installed"
    )
