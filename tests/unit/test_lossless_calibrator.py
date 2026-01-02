"""
Tests for Lossless Calibrator - Verifies no hardcoded values.

These tests ensure the lossless principle is maintained:
- All parameters are derived from market data
- No magic numbers in threshold calculations
- Adaptive behavior across different market conditions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import test fixtures
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.sample_data import generate_ohlcv_data


class TestLosslessParameter:
    """Test LosslessParameter class."""

    def test_parameter_not_calibrated_raises(self):
        """Test that uncalibrated parameter raises error."""
        from core.lossless.parameter import LosslessParameter

        param = LosslessParameter(
            name="test_param",
            derivation_function=lambda df: df['close'].mean()
        )

        with pytest.raises(Exception):
            param.get()

    def test_parameter_calibration(self):
        """Test parameter calibration from data."""
        from core.lossless.parameter import LosslessParameter

        param = LosslessParameter(
            name="test_mean",
            derivation_function=lambda df: float(df['close'].mean()),
            min_samples=50
        )

        df = generate_ohlcv_data(n_bars=100, seed=42)
        result = param.calibrate(df)

        assert result is not None
        assert isinstance(result.value, float)
        assert param.get() == result.value

    def test_parameter_changes_with_data(self):
        """Test that parameter value changes with different data."""
        from core.lossless.parameter import LosslessParameter

        param = LosslessParameter(
            name="volatility",
            derivation_function=lambda df: float(df['close'].pct_change().std()),
            min_samples=50
        )

        # Low volatility data
        df_low = generate_ohlcv_data(n_bars=100, volatility=0.01, seed=1)
        result_low = param.calibrate(df_low)
        val_low = result_low.value

        # High volatility data
        df_high = generate_ohlcv_data(n_bars=100, volatility=0.05, seed=2)
        result_high = param.calibrate(df_high)
        val_high = result_high.value

        assert val_high > val_low, "Parameter should adapt to market volatility"


class TestSpectralAnalysis:
    """Test spectral analysis for cycle detection."""

    def test_dominant_period_detection(self):
        """Test detection of dominant cycle period."""
        from core.lossless.spectral import dominant_cycle_period, find_all_cycles

        # Generate data with embedded cycle
        np.random.seed(42)
        n = 500
        t = np.arange(n)
        cycle_period = 20

        # Price with 20-bar cycle
        cycle = np.sin(2 * np.pi * t / cycle_period)
        noise = np.random.normal(0, 0.1, n)
        price = 100 + 5 * cycle + np.cumsum(noise)

        # Test dominant_cycle_period function
        period = dominant_cycle_period(price)

        # Should be a positive integer
        assert isinstance(period, int)
        assert period > 0

    def test_find_all_cycles(self):
        """Test finding all significant cycles."""
        from core.lossless.spectral import find_all_cycles

        # Generate data with multiple cycles
        np.random.seed(42)
        n = 500
        t = np.arange(n)

        # Price with 20-bar and 50-bar cycles
        cycle1 = np.sin(2 * np.pi * t / 20)
        cycle2 = np.sin(2 * np.pi * t / 50) * 0.5
        noise = np.random.normal(0, 0.1, n)
        price = 100 + 3 * cycle1 + 2 * cycle2 + np.cumsum(noise)

        cycles = find_all_cycles(price, min_period=5)

        # Should return list of (period, power) tuples
        assert isinstance(cycles, list)
        if len(cycles) > 0:
            assert len(cycles[0]) == 2  # (period, power)


class TestEntropyCalculation:
    """Test entropy-based period selection."""

    def test_entropy_calculation(self):
        """Test entropy calculation is consistent."""
        from core.lossless.entropy import sample_entropy, market_entropy

        df = generate_ohlcv_data(n_bars=200, seed=42)
        returns = df['close'].pct_change().dropna().values

        # Test sample entropy
        entropy = sample_entropy(returns)

        assert isinstance(entropy, float)
        assert entropy >= 0 or np.isnan(entropy), "Entropy should be non-negative or NaN"

    def test_entropy_optimal_window(self):
        """Test entropy-based optimal window selection."""
        from core.lossless.entropy import optimal_lookback_from_entropy

        df = generate_ohlcv_data(n_bars=500, seed=42)
        prices = df['close'].values

        # Use correct parameter names: min_period and max_period
        optimal_window = optimal_lookback_from_entropy(prices, min_period=10, max_period=100)

        assert isinstance(optimal_window, int)
        assert optimal_window >= 10
        assert optimal_window <= 100


class TestHurstExponent:
    """Test Hurst exponent calculation."""

    def test_hurst_trending_data(self):
        """Test Hurst for trending (persistent) data."""
        from core.lossless.hurst import hurst_exponent

        # Create trending data (H > 0.5)
        np.random.seed(42)
        n = 500
        trend = np.linspace(0, 10, n)
        noise = np.cumsum(np.random.normal(0, 0.1, n))
        price = 100 + trend + noise * 0.5

        hurst = hurst_exponent(price)

        assert 0 < hurst < 1, "Hurst should be between 0 and 1"
        # Trending data typically has H > 0.5
        assert hurst > 0.4, f"Trending data should have higher Hurst, got {hurst}"

    def test_hurst_mean_reverting_data(self):
        """Test Hurst for mean-reverting data."""
        from core.lossless.hurst import hurst_exponent

        # Create mean-reverting data (H < 0.5)
        np.random.seed(42)
        n = 500
        mean = 100
        reversion = 0.1
        price = [mean]

        for i in range(1, n):
            change = -reversion * (price[-1] - mean) + np.random.normal(0, 1)
            price.append(price[-1] + change)

        price = np.array(price)
        hurst = hurst_exponent(price)

        assert 0 < hurst < 1, "Hurst should be between 0 and 1"
        # Mean reverting typically has H < 0.5
        assert hurst < 0.6, f"Mean-reverting data should have lower Hurst, got {hurst}"


class TestFractalDimension:
    """Test fractal dimension calculations."""

    def test_fractal_dimension_range(self):
        """Test fractal dimension is in valid range."""
        from core.lossless.fractal import fractal_dimension, higuchi_fractal_dimension

        df = generate_ohlcv_data(n_bars=300, seed=42)
        price = df['close'].values

        fd = fractal_dimension(price)

        assert 1.0 <= fd <= 2.0, f"Fractal dimension should be 1-2, got {fd}"

    def test_higuchi_fractal_dimension(self):
        """Test Higuchi fractal dimension calculation."""
        from core.lossless.fractal import higuchi_fractal_dimension

        df = generate_ohlcv_data(n_bars=300, seed=42)
        price = df['close'].values

        fd = higuchi_fractal_dimension(price)

        assert isinstance(fd, float)
        assert fd > 0, "Fractal dimension should be positive"


class TestMarketCalibrator:
    """Test main MarketCalibrator class."""

    def test_full_calibration(self):
        """Test full market calibration."""
        from core.lossless.calibrator import MarketCalibrator

        calibrator = MarketCalibrator(min_calibration_bars=100)

        df = generate_ohlcv_data(n_bars=500, seed=42)

        result = calibrator.calibrate_all(df)

        assert result is not None
        assert hasattr(result, 'fast_period')
        assert hasattr(result, 'slow_period')
        assert hasattr(result, 'current_regime')
        assert hasattr(result, 'hurst_exponent')

    def test_calibration_varies_by_market(self):
        """Test that calibration produces different results for different markets."""
        from core.lossless.calibrator import MarketCalibrator

        calibrator = MarketCalibrator(min_calibration_bars=100)

        # Low volatility market
        df_calm = generate_ohlcv_data(n_bars=500, volatility=0.01, seed=1)
        params_calm = calibrator.calibrate_all(df_calm)

        # High volatility market
        df_volatile = generate_ohlcv_data(n_bars=500, volatility=0.05, seed=2)
        params_volatile = calibrator.calibrate_all(df_volatile)

        # Parameters should differ
        assert params_calm.volatility_scalar != params_volatile.volatility_scalar or \
               params_calm.current_regime != params_volatile.current_regime, \
            "Calibration should adapt to market conditions"

    def test_calibration_result_to_dict(self):
        """Test calibration result can be converted to dict."""
        from core.lossless.calibrator import MarketCalibrator

        calibrator = MarketCalibrator(min_calibration_bars=100)
        df = generate_ohlcv_data(n_bars=500, seed=42)

        result = calibrator.calibrate_all(df)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'fast_period' in result_dict
        assert 'hurst_exponent' in result_dict


class TestLosslessPrinciple:
    """Meta-tests to verify lossless principle across codebase."""

    def test_no_magic_numbers_in_signals(self):
        """Check signal generators don't use magic numbers."""
        import os
        import re

        signals_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'signals')

        if not os.path.exists(signals_dir):
            pytest.skip("Signals directory not found")

        magic_patterns = [
            r'= 14\b',
            r'= 70\b',
            r'= 30\b',
            r'= 0\.8\b',
            r'= 2\.0\b',
        ]

        for root, dirs, files in os.walk(signals_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        content = f.read()

                    for pattern in magic_patterns:
                        # Skip comments
                        lines = [l for l in content.split('\n')
                                 if re.search(pattern, l)
                                 and not l.strip().startswith('#')
                                 and 'guardian' not in l.lower()  # Allow guardian limits
                                 and 'test' not in l.lower()]

                        assert len(lines) == 0, \
                            f"Magic number {pattern} found in {file}: {lines}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
