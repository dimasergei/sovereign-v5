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
            derivation_func=lambda df: df['close'].mean()
        )

        with pytest.raises(Exception):
            param.get()

    def test_parameter_calibration(self):
        """Test parameter calibration from data."""
        from core.lossless.parameter import LosslessParameter

        param = LosslessParameter(
            name="test_mean",
            derivation_func=lambda df: float(df['close'].mean())
        )

        df = generate_ohlcv_data(n_bars=100, seed=42)
        value = param.calibrate(df)

        assert value is not None
        assert isinstance(value, float)
        assert param.get() == value

    def test_parameter_changes_with_data(self):
        """Test that parameter value changes with different data."""
        from core.lossless.parameter import LosslessParameter

        param = LosslessParameter(
            name="volatility",
            derivation_func=lambda df: float(df['close'].pct_change().std())
        )

        # Low volatility data
        df_low = generate_ohlcv_data(n_bars=100, volatility=0.01, seed=1)
        val_low = param.calibrate(df_low)

        # High volatility data
        df_high = generate_ohlcv_data(n_bars=100, volatility=0.05, seed=2)
        val_high = param.calibrate(df_high)

        assert val_high > val_low, "Parameter should adapt to market volatility"


class TestSpectralAnalysis:
    """Test spectral analysis for cycle detection."""

    def test_dominant_period_detection(self):
        """Test detection of dominant cycle period."""
        from core.lossless.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer()

        # Generate data with embedded cycle
        np.random.seed(42)
        n = 500
        t = np.arange(n)
        cycle_period = 20

        # Price with 20-bar cycle
        cycle = np.sin(2 * np.pi * t / cycle_period)
        noise = np.random.normal(0, 0.1, n)
        price = 100 + 5 * cycle + np.cumsum(noise)

        df = pd.DataFrame({
            'close': price,
            'open': price * 0.999,
            'high': price * 1.001,
            'low': price * 0.998,
            'volume': np.random.uniform(100, 1000, n)
        })

        periods = analyzer.find_dominant_periods(df['close'])

        # Should detect period near 20
        assert len(periods) > 0
        assert any(15 <= p <= 25 for p in periods[:3]), \
            f"Should detect ~20 bar cycle, got {periods[:3]}"

    def test_no_hardcoded_periods(self):
        """Verify no hardcoded period values."""
        from core.lossless.spectral import SpectralAnalyzer
        import inspect

        analyzer = SpectralAnalyzer()
        source = inspect.getsource(SpectralAnalyzer)

        # Check for common hardcoded values
        hardcoded_periods = ['= 14', '= 20', '= 50', '= 200']

        for pattern in hardcoded_periods:
            # Allow if in comments or as bounds
            lines = [l for l in source.split('\n')
                     if pattern in l and not l.strip().startswith('#')]
            assert len(lines) == 0 or all('min' in l or 'max' in l or 'limit' in l for l in lines), \
                f"Found potentially hardcoded value: {pattern}"


class TestEntropyCalculation:
    """Test entropy-based period selection."""

    def test_entropy_calculation(self):
        """Test entropy calculation is consistent."""
        from core.lossless.entropy import EntropyCalculator

        calc = EntropyCalculator()

        df = generate_ohlcv_data(n_bars=200, seed=42)
        returns = df['close'].pct_change().dropna()

        entropy = calc.calculate_sample_entropy(returns)

        assert isinstance(entropy, float)
        assert entropy >= 0, "Entropy should be non-negative"

    def test_entropy_optimal_window(self):
        """Test entropy-based optimal window selection."""
        from core.lossless.entropy import EntropyCalculator

        calc = EntropyCalculator()

        df = generate_ohlcv_data(n_bars=500, seed=42)

        optimal_window = calc.find_optimal_window(df['close'])

        assert isinstance(optimal_window, int)
        assert optimal_window > 0
        assert optimal_window < len(df)


class TestHurstExponent:
    """Test Hurst exponent calculation."""

    def test_hurst_trending_data(self):
        """Test Hurst for trending (persistent) data."""
        from core.lossless.hurst import HurstCalculator

        calc = HurstCalculator()

        # Create trending data (H > 0.5)
        np.random.seed(42)
        n = 500
        trend = np.linspace(0, 10, n)
        noise = np.cumsum(np.random.normal(0, 0.1, n))
        price = 100 + trend + noise * 0.5

        hurst = calc.calculate(price)

        assert 0 < hurst < 1, "Hurst should be between 0 and 1"
        # Trending data typically has H > 0.5
        assert hurst > 0.4, f"Trending data should have higher Hurst, got {hurst}"

    def test_hurst_mean_reverting_data(self):
        """Test Hurst for mean-reverting data."""
        from core.lossless.hurst import HurstCalculator

        calc = HurstCalculator()

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
        hurst = calc.calculate(price)

        assert 0 < hurst < 1, "Hurst should be between 0 and 1"
        # Mean reverting typically has H < 0.5
        assert hurst < 0.6, f"Mean-reverting data should have lower Hurst, got {hurst}"


class TestFractalDimension:
    """Test fractal dimension calculations."""

    def test_fractal_dimension_range(self):
        """Test fractal dimension is in valid range."""
        from core.lossless.fractal import FractalCalculator

        calc = FractalCalculator()

        df = generate_ohlcv_data(n_bars=300, seed=42)
        price = df['close'].values

        fd = calc.calculate_box_dimension(price)

        assert 1.0 <= fd <= 2.0, f"Fractal dimension should be 1-2, got {fd}"

    def test_fractal_adapts_to_complexity(self):
        """Test fractal dimension adapts to price complexity."""
        from core.lossless.fractal import FractalCalculator

        calc = FractalCalculator()

        # Simple trend (low complexity)
        simple = np.linspace(100, 200, 300)
        fd_simple = calc.calculate_box_dimension(simple)

        # Complex noisy data (high complexity)
        np.random.seed(42)
        complex_data = 100 + np.cumsum(np.random.normal(0, 2, 300))
        fd_complex = calc.calculate_box_dimension(complex_data)

        assert fd_complex > fd_simple, \
            "Complex data should have higher fractal dimension"


class TestMarketCalibrator:
    """Test main MarketCalibrator class."""

    def test_full_calibration(self):
        """Test full market calibration."""
        from core.lossless.calibrator import MarketCalibrator

        calibrator = MarketCalibrator()

        df = generate_ohlcv_data(n_bars=500, seed=42)

        params = calibrator.calibrate(df)

        assert params is not None
        assert 'optimal_period' in params
        assert 'volatility_regime' in params
        assert 'hurst_exponent' in params

    def test_calibration_varies_by_market(self):
        """Test that calibration produces different results for different markets."""
        from core.lossless.calibrator import MarketCalibrator

        calibrator = MarketCalibrator()

        # Low volatility market
        df_calm = generate_ohlcv_data(n_bars=500, volatility=0.01, seed=1)
        params_calm = calibrator.calibrate(df_calm)

        # High volatility market
        df_volatile = generate_ohlcv_data(n_bars=500, volatility=0.05, seed=2)
        params_volatile = calibrator.calibrate(df_volatile)

        # Parameters should differ
        assert params_calm['volatility_regime'] != params_volatile['volatility_regime'] or \
               params_calm.get('risk_scalar', 1) != params_volatile.get('risk_scalar', 1), \
            "Calibration should adapt to market conditions"

    def test_no_hardcoded_thresholds(self):
        """Verify calibrator derives all thresholds from data."""
        from core.lossless.calibrator import MarketCalibrator
        import inspect

        source = inspect.getsource(MarketCalibrator)

        # Common hardcoded patterns to avoid
        forbidden_patterns = [
            '= 0.7', '= 0.3',  # Typical RSI thresholds
            '= 70', '= 30',    # RSI values
            '= 14',            # Common period
            '= 2.0',           # Common multiplier
        ]

        for pattern in forbidden_patterns:
            # Skip if in comments
            matches = [l for l in source.split('\n')
                       if pattern in l
                       and not l.strip().startswith('#')
                       and 'version' not in l.lower()]
            # Allow if it's derived (e.g., = value * 0.7)
            real_matches = [m for m in matches if f"'{pattern[2:]}'" not in m]
            assert len(real_matches) == 0, \
                f"Found potentially hardcoded: {pattern} in {real_matches}"


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
