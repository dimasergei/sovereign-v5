"""
Tests for Execution Algorithms - TWAP, VWAP, Iceberg.

Verifies correct implementation of institutional execution algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.mock_mt5 import MockMT5
from fixtures.sample_data import generate_ohlcv_data


class TestTWAPExecution:
    """Test Time-Weighted Average Price execution."""

    def test_twap_splits_order(self):
        """Test TWAP splits order into equal parts."""
        from core.execution import TWAPExecutor

        executor = TWAPExecutor()

        total_volume = 1.0
        duration_minutes = 60
        interval_minutes = 10

        slices = executor.calculate_slices(
            total_volume=total_volume,
            duration_minutes=duration_minutes,
            interval_minutes=interval_minutes
        )

        expected_slices = duration_minutes // interval_minutes
        assert len(slices) == expected_slices

        # Each slice should be equal
        for slice_vol in slices:
            assert abs(slice_vol - total_volume / expected_slices) < 0.001

    def test_twap_randomization(self):
        """Test TWAP adds randomization to avoid detection."""
        from core.execution import TWAPExecutor

        executor = TWAPExecutor(randomize=True, randomize_pct=0.1)

        total_volume = 1.0

        # Generate multiple runs
        all_slices = []
        for _ in range(10):
            slices = executor.calculate_slices(
                total_volume=total_volume,
                duration_minutes=60,
                interval_minutes=10
            )
            all_slices.append(slices)

        # Slices should vary between runs
        first_slices = all_slices[0]
        has_variation = False

        for slices in all_slices[1:]:
            if not np.allclose(slices, first_slices):
                has_variation = True
                break

        assert has_variation, "Randomized TWAP should vary between runs"

    def test_twap_respects_min_volume(self):
        """Test TWAP respects minimum volume constraints."""
        from core.execution import TWAPExecutor

        executor = TWAPExecutor()

        slices = executor.calculate_slices(
            total_volume=0.1,
            duration_minutes=60,
            interval_minutes=10,
            min_volume=0.02
        )

        for slice_vol in slices:
            assert slice_vol >= 0.02 or slice_vol == 0


class TestIcebergExecution:
    """Test Iceberg order execution."""

    def test_iceberg_hides_size(self):
        """Test iceberg shows only visible portion."""
        from core.execution import IcebergExecutor

        executor = IcebergExecutor()

        total_volume = 10.0
        visible_pct = 0.2

        visible, hidden = executor.split_order(
            total_volume=total_volume,
            visible_pct=visible_pct
        )

        assert visible == total_volume * visible_pct
        assert hidden == total_volume * (1 - visible_pct)
        assert visible + hidden == total_volume

    def test_iceberg_replenishes(self):
        """Test iceberg replenishes visible portion after fill."""
        from core.execution import IcebergExecutor

        executor = IcebergExecutor()

        total_volume = 10.0
        visible_pct = 0.2

        state = executor.create_order_state(
            total_volume=total_volume,
            visible_pct=visible_pct
        )

        # Simulate fill of visible portion
        filled = state['visible']
        new_state = executor.replenish(state, filled)

        assert new_state['filled'] == filled
        assert new_state['remaining'] == total_volume - filled

        # Visible should be replenished up to remaining
        expected_visible = min(total_volume * visible_pct, new_state['remaining'])
        assert new_state['visible'] == expected_visible


class TestSlippageModel:
    """Test slippage estimation."""

    def test_slippage_increases_with_size(self):
        """Test slippage increases with order size."""
        from core.execution import SlippageModel

        model = SlippageModel()

        # Small order
        small_slippage = model.estimate_slippage(
            volume=0.1,
            avg_volume=1000,
            spread=0.0001,
            volatility=0.02
        )

        # Large order
        large_slippage = model.estimate_slippage(
            volume=10.0,
            avg_volume=1000,
            spread=0.0001,
            volatility=0.02
        )

        assert large_slippage > small_slippage

    def test_slippage_increases_with_volatility(self):
        """Test slippage increases with volatility."""
        from core.execution import SlippageModel

        model = SlippageModel()

        # Low volatility
        low_vol_slippage = model.estimate_slippage(
            volume=1.0,
            avg_volume=1000,
            spread=0.0001,
            volatility=0.01
        )

        # High volatility
        high_vol_slippage = model.estimate_slippage(
            volume=1.0,
            avg_volume=1000,
            spread=0.0001,
            volatility=0.05
        )

        assert high_vol_slippage > low_vol_slippage

    def test_slippage_uses_derived_parameters(self):
        """Test slippage model doesn't use hardcoded values."""
        from core.execution import SlippageModel
        import inspect

        source = inspect.getsource(SlippageModel)

        # Should not have hardcoded slippage values
        hardcoded = ['= 0.001', '= 0.0001', '= 0.0005']

        for pattern in hardcoded:
            matches = [l for l in source.split('\n')
                       if pattern in l
                       and not l.strip().startswith('#')
                       and 'default' not in l.lower()]
            # Allow in function signatures or as derived values
            assert all('*' in m or '/' in m or '+' in m or 'param' in m.lower()
                       for m in matches), \
                f"Potential hardcoded slippage: {pattern}"


class TestOrderRouter:
    """Test smart order routing."""

    def test_route_small_order_direct(self):
        """Test small orders are routed directly."""
        from core.execution import OrderRouter

        router = OrderRouter()

        route = router.determine_route(
            volume=0.1,
            avg_daily_volume=10000,
            urgency='normal'
        )

        assert route['method'] == 'market'
        assert route['algo'] is None

    def test_route_large_order_twap(self):
        """Test large orders use TWAP."""
        from core.execution import OrderRouter

        router = OrderRouter()

        route = router.determine_route(
            volume=100,
            avg_daily_volume=1000,
            urgency='normal'
        )

        assert route['algo'] in ['twap', 'vwap', 'iceberg']

    def test_route_urgent_order_market(self):
        """Test urgent orders bypass algos."""
        from core.execution import OrderRouter

        router = OrderRouter()

        route = router.determine_route(
            volume=100,
            avg_daily_volume=1000,
            urgency='high'
        )

        # High urgency should use faster execution
        assert route['method'] == 'market' or route['algo'] == 'twap'


class TestExecutionIntegration:
    """Integration tests for execution system."""

    def test_full_execution_flow(self):
        """Test complete execution flow."""
        from core.execution import ExecutionEngine

        mt5 = MockMT5()
        mt5.initialize()

        engine = ExecutionEngine(mt5_connector=mt5)

        # Place order
        result = engine.execute(
            symbol='BTCUSD.x',
            direction='buy',
            volume=0.1,
            order_type='market'
        )

        assert result['success'] is True or result.get('retcode') == 10009
        mt5.shutdown()

    def test_execution_respects_limits(self):
        """Test execution respects position limits."""
        from core.execution import ExecutionEngine

        mt5 = MockMT5()
        mt5.initialize()
        mt5.set_account_balance(1000)  # Small account

        engine = ExecutionEngine(
            mt5_connector=mt5,
            max_position_pct=0.5
        )

        # Try to place oversized order
        result = engine.execute(
            symbol='BTCUSD.x',
            direction='buy',
            volume=100,  # Way too large
            order_type='market'
        )

        # Should either reject or reduce size
        assert result.get('volume', 0) < 100 or result['success'] is False

        mt5.shutdown()


class TestFillAnalysis:
    """Test post-trade fill analysis."""

    def test_fill_analysis_metrics(self):
        """Test fill analysis calculates correct metrics."""
        from core.execution import FillAnalyzer

        analyzer = FillAnalyzer()

        fill = {
            'symbol': 'BTCUSD',
            'direction': 'buy',
            'volume': 1.0,
            'requested_price': 50000.0,
            'fill_price': 50010.0,
            'timestamp': datetime.now(),
        }

        market = {
            'vwap': 50005.0,
            'twap': 50003.0,
            'arrival_price': 50000.0,
        }

        analysis = analyzer.analyze(fill, market)

        assert 'slippage' in analysis
        assert 'implementation_shortfall' in analysis
        assert analysis['slippage'] == 10.0  # 50010 - 50000

    def test_fill_quality_scoring(self):
        """Test fill quality scoring."""
        from core.execution import FillAnalyzer

        analyzer = FillAnalyzer()

        # Good fill (below VWAP for buy)
        good_fill = {
            'direction': 'buy',
            'fill_price': 49990.0,
            'vwap': 50000.0,
        }

        good_score = analyzer.score_fill(good_fill)

        # Bad fill (above VWAP for buy)
        bad_fill = {
            'direction': 'buy',
            'fill_price': 50050.0,
            'vwap': 50000.0,
        }

        bad_score = analyzer.score_fill(bad_fill)

        assert good_score > bad_score


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
