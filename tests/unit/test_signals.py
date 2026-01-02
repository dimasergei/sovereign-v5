"""
Tests for Signal Generation - Verifies signal quality and correctness.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.sample_data import generate_ohlcv_data, create_sample_signals


class TestSignalGenerator:
    """Test main signal generator."""

    def test_signal_output_format(self):
        """Test signal output is in correct format."""
        from signals.generator import SignalGenerator

        generator = SignalGenerator()
        df = generate_ohlcv_data(n_bars=200, seed=42)

        signal = generator.generate_signal('BTCUSD', df)

        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'strength')
        assert hasattr(signal, 'confidence')

        assert signal.direction in [-1, 0, 1]
        assert 0 <= signal.strength <= 1
        assert 0 <= signal.confidence <= 1

    def test_signal_adapts_to_regime(self):
        """Test signals adapt to market regime."""
        from signals.generator import SignalGenerator

        generator = SignalGenerator()

        # Trending market
        trending = generate_ohlcv_data(n_bars=200, trend=0.001, volatility=0.01, seed=1)
        signal_trend = generator.generate_signal('BTCUSD', trending)

        # Ranging market
        ranging = generate_ohlcv_data(n_bars=200, trend=0.0, volatility=0.02, seed=2)
        signal_range = generator.generate_signal('BTCUSD', ranging)

        # Signals should differ based on regime
        # (At minimum, regime detection should be different)
        assert signal_trend.metadata.get('regime') != signal_range.metadata.get('regime') or \
               signal_trend.confidence != signal_range.confidence

    def test_signal_uses_calibrated_parameters(self):
        """Test signal uses market-calibrated parameters."""
        from signals.generator import SignalGenerator
        import inspect

        source = inspect.getsource(SignalGenerator)

        # Should not have hardcoded indicator periods
        forbidden = ['period=14', 'period = 14', 'lookback=14', 'lookback = 14']

        for pattern in forbidden:
            assert pattern not in source, f"Hardcoded period found: {pattern}"


class TestMicrostructureSignals:
    """Test microstructure signal generator."""

    def test_trade_imbalance_calculation(self):
        """Test trade imbalance is calculated correctly."""
        from signals.generators.microstructure import MicrostructureAnalyzer

        analyzer = MicrostructureAnalyzer()

        # Create tick data with buy imbalance
        ticks = [
            {'side': 'buy', 'volume': 100},
            {'side': 'buy', 'volume': 150},
            {'side': 'sell', 'volume': 50},
        ]

        imbalance = analyzer.calculate_trade_imbalance(ticks)

        assert imbalance > 0, "Should show buy imbalance"

    def test_spread_signal(self):
        """Test spread dynamics signal."""
        from signals.generators.microstructure import MicrostructureAnalyzer

        analyzer = MicrostructureAnalyzer()

        # Normal spread
        normal_signal = analyzer.calculate_spread_signal(
            current_spread=0.0001,
            avg_spread=0.0001,
            spread_std=0.00002
        )

        # Wide spread (risk off)
        wide_signal = analyzer.calculate_spread_signal(
            current_spread=0.0003,
            avg_spread=0.0001,
            spread_std=0.00002
        )

        assert wide_signal < normal_signal, "Wide spread should be bearish"

    def test_vpin_calculation(self):
        """Test VPIN (Volume-synchronized probability) calculation."""
        from signals.generators.microstructure import MicrostructureAnalyzer

        analyzer = MicrostructureAnalyzer()

        # Generate sample volume bars
        volumes = np.random.uniform(100, 1000, 50)
        buy_volumes = volumes * np.random.uniform(0.3, 0.7, 50)
        sell_volumes = volumes - buy_volumes

        vpin = analyzer.calculate_vpin(buy_volumes, sell_volumes)

        assert 0 <= vpin <= 1, "VPIN should be between 0 and 1"


class TestRegimeDetection:
    """Test market regime detection."""

    def test_volatility_regime_detection(self):
        """Test volatility regime is detected correctly."""
        from models.regime.volatility_regime import VolatilityRegimeDetector

        detector = VolatilityRegimeDetector()

        # Low volatility data
        df_low = generate_ohlcv_data(n_bars=200, volatility=0.005, seed=1)
        regime_low = detector.detect_regime(df_low)

        # High volatility data
        df_high = generate_ohlcv_data(n_bars=200, volatility=0.05, seed=2)
        regime_high = detector.detect_regime(df_high)

        assert regime_low != regime_high or regime_low['volatility_level'] != regime_high['volatility_level']

    def test_regime_uses_adaptive_thresholds(self):
        """Test regime detection uses adaptive thresholds."""
        from models.regime.volatility_regime import VolatilityRegimeDetector
        import inspect

        source = inspect.getsource(VolatilityRegimeDetector)

        # Should use percentile-based thresholds, not hardcoded
        assert 'percentile' in source.lower() or 'quantile' in source.lower(), \
            "Should use percentile-based thresholds"


class TestSignalCombination:
    """Test signal combination and ensemble."""

    def test_signal_aggregation(self):
        """Test multiple signals are aggregated correctly."""
        from signals.generator import SignalAggregator

        aggregator = SignalAggregator()

        signals = [
            {'direction': 1, 'confidence': 0.8, 'source': 'model_a'},
            {'direction': 1, 'confidence': 0.6, 'source': 'model_b'},
            {'direction': -1, 'confidence': 0.4, 'source': 'model_c'},
        ]

        combined = aggregator.combine(signals)

        # Majority bullish with higher confidence should win
        assert combined['direction'] == 1
        assert 0 < combined['confidence'] < 1

    def test_disagreement_reduces_confidence(self):
        """Test that model disagreement reduces confidence."""
        from signals.generator import SignalAggregator

        aggregator = SignalAggregator()

        # Agreement
        agreed_signals = [
            {'direction': 1, 'confidence': 0.8, 'source': 'a'},
            {'direction': 1, 'confidence': 0.7, 'source': 'b'},
            {'direction': 1, 'confidence': 0.6, 'source': 'c'},
        ]

        # Disagreement
        disagreed_signals = [
            {'direction': 1, 'confidence': 0.8, 'source': 'a'},
            {'direction': -1, 'confidence': 0.7, 'source': 'b'},
            {'direction': 1, 'confidence': 0.6, 'source': 'c'},
        ]

        agreed = aggregator.combine(agreed_signals)
        disagreed = aggregator.combine(disagreed_signals)

        assert disagreed['confidence'] < agreed['confidence']


class TestSignalFiltering:
    """Test signal filtering mechanisms."""

    def test_regime_filter(self):
        """Test signals are filtered by regime."""
        # Trend signals should be filtered in ranging market
        # Mean reversion signals should be filtered in trending market
        pass  # Implement when filter module is available

    def test_correlation_filter(self):
        """Test correlated signals are reduced."""
        pass  # Implement when filter module is available

    def test_volatility_filter(self):
        """Test signals are adjusted for volatility."""
        pass  # Implement when filter module is available


class TestSignalValidation:
    """Test signal validation and quality checks."""

    def test_signal_decay_detection(self):
        """Test detection of signal decay over time."""
        from signals.generator import SignalValidator

        validator = SignalValidator()

        # Simulate signal history
        signal_history = [
            {'time': datetime.now() - timedelta(hours=i), 'direction': 1, 'result': 1 if i < 5 else -1}
            for i in range(20)
        ]

        decay_metrics = validator.check_decay(signal_history)

        assert 'decay_rate' in decay_metrics
        assert decay_metrics['decay_rate'] > 0  # Should show decay

    def test_signal_quality_score(self):
        """Test overall signal quality scoring."""
        from signals.generator import SignalValidator

        validator = SignalValidator()

        # Good quality signal
        good_signal = {
            'confidence': 0.8,
            'model_agreement': 0.9,
            'regime_alignment': True,
            'correlation_check': 'pass',
        }

        # Poor quality signal
        poor_signal = {
            'confidence': 0.3,
            'model_agreement': 0.4,
            'regime_alignment': False,
            'correlation_check': 'fail',
        }

        good_score = validator.quality_score(good_signal)
        poor_score = validator.quality_score(poor_signal)

        assert good_score > poor_score


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
