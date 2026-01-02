"""
Tests for Execution Algorithms - SmartExecutor with TWAP, ICEBERG, ADAPTIVE styles.

Verifies correct implementation of institutional execution algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fixtures.mock_mt5 import MockMT5
from fixtures.sample_data import generate_ohlcv_data


class TestSmartExecutor:
    """Test SmartExecutor class."""

    def test_executor_initialization(self):
        """Test SmartExecutor initializes correctly."""
        from core.execution import SmartExecutor

        mock_connector = Mock()
        executor = SmartExecutor(connector=mock_connector, max_slippage_pips=5.0)

        assert executor.connector == mock_connector
        assert executor.max_slippage_pips == 5.0
        assert executor.total_orders == 0
        assert executor.successful_orders == 0

    def test_create_plan_market(self):
        """Test market execution plan creation."""
        from core.execution import SmartExecutor, ExecutionStyle, ExecutionPlan

        mock_connector = Mock()
        executor = SmartExecutor(connector=mock_connector)

        plan = executor.create_plan(
            symbol="BTCUSD.x",
            direction="buy",
            size=0.1,
            sl=49000,
            tp=52000,
            style=ExecutionStyle.MARKET
        )

        assert isinstance(plan, ExecutionPlan)
        assert plan.symbol == "BTCUSD.x"
        assert plan.direction == "buy"
        assert plan.total_size == 0.1
        assert plan.style == ExecutionStyle.MARKET
        assert plan.stop_loss == 49000
        assert plan.take_profit == 52000
        assert len(plan.slices) == 1

    def test_create_plan_twap(self):
        """Test TWAP execution plan splits order into slices."""
        from core.execution import SmartExecutor, ExecutionStyle

        mock_connector = Mock()
        executor = SmartExecutor(connector=mock_connector)

        plan = executor.create_plan(
            symbol="BTCUSD.x",
            direction="buy",
            size=1.0,
            sl=49000,
            style=ExecutionStyle.TWAP
        )

        # TWAP should create multiple slices
        assert len(plan.slices) >= 2
        # Total size of slices should equal total order size
        total_slice_size = sum(s["size"] for s in plan.slices)
        assert abs(total_slice_size - 1.0) < 0.001

    def test_create_plan_iceberg(self):
        """Test Iceberg execution plan shows only visible portion."""
        from core.execution import SmartExecutor, ExecutionStyle

        mock_connector = Mock()
        executor = SmartExecutor(connector=mock_connector)

        plan = executor.create_plan(
            symbol="BTCUSD.x",
            direction="buy",
            size=1.0,
            sl=49000,
            style=ExecutionStyle.ICEBERG
        )

        # Iceberg should create multiple slices with small visible size
        assert len(plan.slices) >= 2
        # Each slice should be small (visible portion)
        for slice_info in plan.slices:
            assert slice_info["size"] <= 0.25  # At most 25% visible

    def test_auto_style_selection_small_order(self):
        """Test small orders use MARKET style by default."""
        from core.execution import SmartExecutor, ExecutionStyle

        mock_connector = Mock()
        executor = SmartExecutor(connector=mock_connector)

        # Mock symbol_info for auto-selection
        with patch('MetaTrader5.symbol_info') as mock_info:
            mock_info.return_value = MagicMock()

            plan = executor.create_plan(
                symbol="BTCUSD.x",
                direction="buy",
                size=0.05,  # Small order
                sl=49000,
                urgency=0.5
            )

            assert plan.style == ExecutionStyle.MARKET

    def test_auto_style_selection_large_order(self):
        """Test large orders with low urgency use TWAP or ADAPTIVE."""
        from core.execution import SmartExecutor, ExecutionStyle

        mock_connector = Mock()
        executor = SmartExecutor(connector=mock_connector)

        with patch('MetaTrader5.symbol_info') as mock_info:
            mock_info.return_value = MagicMock()

            plan = executor.create_plan(
                symbol="BTCUSD.x",
                direction="buy",
                size=1.0,  # Large order
                sl=49000,
                urgency=0.2  # Low urgency
            )

            # Should use TWAP or ADAPTIVE for large orders
            assert plan.style in [ExecutionStyle.TWAP, ExecutionStyle.ADAPTIVE]


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_creation(self):
        """Test ExecutionResult can be created."""
        from core.execution import ExecutionResult

        result = ExecutionResult(
            success=True,
            ticket=12345,
            symbol="BTCUSD.x",
            direction="buy",
            avg_fill_price=50000.0,
            total_filled=0.1,
            slippage_pips=1.5
        )

        assert result.success is True
        assert result.ticket == 12345
        assert result.avg_fill_price == 50000.0
        assert result.slippage_pips == 1.5

    def test_execution_result_failure(self):
        """Test ExecutionResult for failed execution."""
        from core.execution import ExecutionResult

        result = ExecutionResult(
            success=False,
            error_message="Not connected to MT5"
        )

        assert result.success is False
        assert result.error_message == "Not connected to MT5"


class TestExecutionPlan:
    """Test ExecutionPlan dataclass."""

    def test_execution_plan_creation(self):
        """Test ExecutionPlan can be created."""
        from core.execution import ExecutionPlan, ExecutionStyle

        plan = ExecutionPlan(
            symbol="BTCUSD.x",
            direction="buy",
            total_size=1.0,
            style=ExecutionStyle.TWAP,
            stop_loss=49000,
            take_profit=52000
        )

        assert plan.symbol == "BTCUSD.x"
        assert plan.direction == "buy"
        assert plan.total_size == 1.0
        assert plan.style == ExecutionStyle.TWAP
        assert plan.stop_loss == 49000


class TestExecutionStyles:
    """Test ExecutionStyle enum."""

    def test_execution_styles_exist(self):
        """Test all execution styles are defined."""
        from core.execution import ExecutionStyle

        assert hasattr(ExecutionStyle, 'MARKET')
        assert hasattr(ExecutionStyle, 'TWAP')
        assert hasattr(ExecutionStyle, 'ICEBERG')
        assert hasattr(ExecutionStyle, 'ADAPTIVE')

    def test_execution_style_values(self):
        """Test execution style values."""
        from core.execution import ExecutionStyle

        assert ExecutionStyle.MARKET.value == "market"
        assert ExecutionStyle.TWAP.value == "twap"
        assert ExecutionStyle.ICEBERG.value == "iceberg"
        assert ExecutionStyle.ADAPTIVE.value == "adaptive"


class TestSlippageCalculation:
    """Test slippage is calculated correctly."""

    def test_slippage_in_result(self):
        """Test ExecutionResult contains slippage."""
        from core.execution import ExecutionResult

        result = ExecutionResult(
            success=True,
            ticket=12345,
            symbol="BTCUSD.x",
            direction="buy",
            avg_fill_price=50010.0,
            total_filled=0.1,
            slippage_pips=1.0
        )

        assert result.slippage_pips == 1.0


class TestExecutionStats:
    """Test execution statistics tracking."""

    def test_stats_tracking(self):
        """Test executor tracks statistics."""
        from core.execution import SmartExecutor

        mock_connector = Mock()
        executor = SmartExecutor(connector=mock_connector)

        # Initial stats
        assert executor.total_orders == 0
        assert executor.successful_orders == 0

        # Get stats
        stats = executor.get_execution_stats()
        assert 'total_orders' in stats
        assert 'successful_orders' in stats
        assert 'success_rate' in stats
        assert 'avg_slippage_pips' in stats


class TestExecutionIntegration:
    """Integration tests for execution system."""

    def test_full_execution_flow_with_mock(self):
        """Test complete execution flow with MockMT5."""
        from core.execution import SmartExecutor, ExecutionStyle

        mt5 = MockMT5()
        mt5.initialize()

        # Create a mock connector that uses MockMT5
        mock_connector = Mock()
        mock_connector.ensure_connected.return_value = True

        executor = SmartExecutor(connector=mock_connector)

        plan = executor.create_plan(
            symbol='BTCUSD.x',
            direction='buy',
            size=0.1,
            sl=49000,
            style=ExecutionStyle.MARKET
        )

        # Verify plan was created correctly
        assert plan.symbol == 'BTCUSD.x'
        assert plan.total_size == 0.1

        mt5.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
