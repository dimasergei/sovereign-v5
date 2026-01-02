"""
Tests for Signal Generation - Verifies signal quality and correctness.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.sample_data import generate_ohlcv_data


class TestSignalGenerator:
    """Test main signal generator."""

    def test_signal_output_format(self):
        """Test signal output is in correct format."""
        from signals.generator import SignalGenerator, TradingSignal
        from core.lossless.calibrator import MarketCalibrator
        from models.statistical_models import RegimeDetector

        # Create generator with required dependencies
        calibrator = MarketCalibrator(min_calibration_bars=100)
        regime_detector = RegimeDetector()

        generator = SignalGenerator(
            calibrator=calibrator,
            regime_detector=regime_detector
        )

        df = generate_ohlcv_data(n_bars=200, seed=42)

        signal = generator.generate_signal('BTCUSD', df)

        # Check TradingSignal attributes
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'action')

        assert signal.direction >= -1 and signal.direction <= 1
        assert 0 <= signal.confidence <= 1
        assert signal.action in ['long', 'short', 'neutral']

    def test_neutral_signal_on_insufficient_data(self):
        """Test neutral signal returned when data is insufficient."""
        from signals.generator import SignalGenerator
        from core.lossless.calibrator import MarketCalibrator
        from models.statistical_models import RegimeDetector

        calibrator = MarketCalibrator(min_calibration_bars=100)
        generator = SignalGenerator(
            calibrator=calibrator,
            regime_detector=RegimeDetector()
        )

        # Only 50 bars - not enough
        df = generate_ohlcv_data(n_bars=50, seed=42)

        signal = generator.generate_signal('BTCUSD', df)

        assert signal.action == 'neutral'

    def test_trading_signal_dataclass(self):
        """Test TradingSignal dataclass has required fields."""
        from signals.generator import TradingSignal

        signal = TradingSignal(
            symbol='BTCUSD',
            action='long',
            direction=0.8,
            confidence=0.75,
            position_scalar=0.5,
            stop_loss_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            regime='trending_up',
            model_agreement=0.9
        )

        assert signal.symbol == 'BTCUSD'
        assert signal.action == 'long'
        assert signal.direction == 0.8

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

    def test_microstructure_analyzer_initialization(self):
        """Test MicrostructureAnalyzer initialization."""
        from signals.generators.microstructure import MicrostructureAnalyzer

        analyzer = MicrostructureAnalyzer(
            buffer_size=1000,
            vpin_bucket_size=50
        )

        assert analyzer.buffer_size == 1000
        assert analyzer.vpin_bucket_size == 50

    def test_tick_data_properties(self):
        """Test TickData dataclass properties."""
        from signals.generators.microstructure import TickData

        tick = TickData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.5,
            last=100.25,
            bid_size=1000,
            ask_size=800,
            volume=500
        )

        assert tick.mid == 100.25
        assert tick.spread == 0.5
        assert tick.spread_bps > 0

    def test_process_tick_returns_signals(self):
        """Test processing a tick returns microstructure signals."""
        from signals.generators.microstructure import MicrostructureAnalyzer, TickData

        analyzer = MicrostructureAnalyzer()

        tick = TickData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.5,
            last=100.25,
            bid_size=1000,
            ask_size=800,
            volume=500
        )

        signals = analyzer.process_tick(tick)

        assert hasattr(signals, 'trade_imbalance')
        assert hasattr(signals, 'quote_pressure')
        assert hasattr(signals, 'vpin')
        assert hasattr(signals, 'aggregate_signal')

    def test_iceberg_detection_returns_dict(self):
        """Test iceberg order detection returns proper format."""
        from signals.generators.microstructure import MicrostructureAnalyzer

        analyzer = MicrostructureAnalyzer()

        result = analyzer.detect_iceberg_orders()

        assert isinstance(result, dict)
        assert 'detected' in result

    def test_toxic_flow_detection_returns_dict(self):
        """Test toxic flow detection returns proper format."""
        from signals.generators.microstructure import MicrostructureAnalyzer

        analyzer = MicrostructureAnalyzer()

        result = analyzer.detect_toxic_flow()

        assert isinstance(result, dict)
        assert 'toxic' in result


class TestRegimeDetection:
    """Test market regime detection."""

    def test_regime_detector_initialization(self):
        """Test RegimeDetector initialization."""
        from models.statistical_models import RegimeDetector

        detector = RegimeDetector()

        assert hasattr(detector, 'detect_regime')
        assert hasattr(detector, 'predict')

    def test_regime_detector_predict(self):
        """Test RegimeDetector predict method."""
        from models.statistical_models import RegimeDetector

        detector = RegimeDetector()

        # Generate price data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)

        prediction = detector.predict(prices)

        assert hasattr(prediction, 'direction')
        assert hasattr(prediction, 'confidence')
        assert hasattr(prediction, 'metadata')
        assert 'regime' in prediction.metadata

    def test_garch_model_volatility_regime(self):
        """Test GARCH model volatility regime detection."""
        pytest.importorskip('arch', reason="arch package not installed")

        from models.regime.volatility_regime import GARCHModel

        model = GARCHModel(variant='garch')

        # Generate returns
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02

        model.fit(returns)

        regime = model.detect_volatility_regime()

        assert isinstance(regime, dict)
        assert 'regime' in regime
        assert 'current_vol' in regime
        assert 'position_scalar' in regime


class TestVolatilityFilter:
    """Test volatility-based signal filtering."""

    def test_volatility_filter_scale_position(self):
        """Test position scaling based on volatility."""
        from models.regime.volatility_regime import VolatilityRegimeFilter

        filter = VolatilityRegimeFilter(lookback=100)

        # Generate returns
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02

        scaled = filter.scale_position(1.0, returns)

        assert isinstance(scaled, float)
        assert scaled > 0

    def test_volatility_filter_filter_signal(self):
        """Test signal filtering based on volatility."""
        from models.regime.volatility_regime import VolatilityRegimeFilter

        filter = VolatilityRegimeFilter(lookback=100)

        # Generate returns
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02

        filtered, adjustment = filter.filter_signal(0.8, returns)

        assert isinstance(filtered, float)
        assert isinstance(adjustment, float)


class TestSignalFiltering:
    """Test signal filtering mechanisms."""

    def test_regime_filter_placeholder(self):
        """Test signals are filtered by regime."""
        # Trend signals should be filtered in ranging market
        # Mean reversion signals should be filtered in trending market
        pass  # Placeholder for future implementation

    def test_correlation_filter_placeholder(self):
        """Test correlated signals are reduced."""
        pass  # Placeholder for future implementation

    def test_volatility_filter_placeholder(self):
        """Test signals are adjusted for volatility."""
        pass  # Placeholder for future implementation


class TestSignalIntegration:
    """Integration tests for signal generation."""

    def test_full_signal_pipeline(self):
        """Test complete signal generation pipeline."""
        from signals.generator import SignalGenerator
        from core.lossless.calibrator import MarketCalibrator
        from models.statistical_models import RegimeDetector

        # Create full pipeline
        calibrator = MarketCalibrator(min_calibration_bars=100)
        regime_detector = RegimeDetector()

        generator = SignalGenerator(
            calibrator=calibrator,
            regime_detector=regime_detector
        )

        # Generate OHLCV data
        df = generate_ohlcv_data(n_bars=500, seed=42)

        # Generate signal
        signal = generator.generate_signal('BTCUSD', df)

        # Verify complete signal
        assert signal.symbol == 'BTCUSD'
        assert signal.action in ['long', 'short', 'neutral']
        assert hasattr(signal, 'stop_loss_atr_mult')
        assert hasattr(signal, 'take_profit_atr_mult')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
