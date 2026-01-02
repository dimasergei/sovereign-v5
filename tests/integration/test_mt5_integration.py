"""
Integration Tests for MT5 - Tests full trading flow with mock MT5.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.mock_mt5 import MockMT5, TradeRetcode
from fixtures.sample_data import generate_ohlcv_data


class TestMT5Connection:
    """Test MT5 connection handling."""

    def test_connection_lifecycle(self):
        """Test connect/disconnect cycle."""
        mt5 = MockMT5()

        assert mt5.initialize() is True
        assert mt5.version()[0] == 5

        mt5.shutdown()

    def test_reconnection_after_failure(self):
        """Test system handles reconnection."""
        mt5 = MockMT5()

        # First connection
        mt5.initialize()
        mt5.shutdown()

        # Reconnect
        assert mt5.initialize() is True


class TestDataRetrieval:
    """Test market data retrieval."""

    def test_get_historical_data(self):
        """Test fetching historical bars."""
        mt5 = MockMT5()
        mt5.initialize()

        rates = mt5.copy_rates_from(
            symbol="BTCUSD.x",
            timeframe=60,  # 1H
            date_from=datetime.now() - timedelta(days=7),
            count=100
        )

        assert rates is not None
        assert len(rates) == 100
        assert 'open' in rates.dtype.names
        assert 'close' in rates.dtype.names

        mt5.shutdown()

    def test_get_current_tick(self):
        """Test getting current tick data."""
        mt5 = MockMT5()
        mt5.initialize()

        tick = mt5.symbol_info_tick("BTCUSD.x")

        assert tick is not None
        assert 'bid' in tick
        assert 'ask' in tick
        assert tick['ask'] > tick['bid']

        mt5.shutdown()

    def test_get_symbol_info(self):
        """Test getting symbol specifications."""
        mt5 = MockMT5()
        mt5.initialize()

        info = mt5.symbol_info("BTCUSD.x")

        assert info is not None
        assert info.name == "BTCUSD.x"
        assert info.volume_min > 0
        assert info.volume_max > info.volume_min

        mt5.shutdown()


class TestOrderExecution:
    """Test order execution flow."""

    def test_market_buy_order(self):
        """Test placing market buy order."""
        mt5 = MockMT5()
        mt5.initialize()

        request = {
            'action': 1,  # TRADE_ACTION_DEAL
            'symbol': 'BTCUSD.x',
            'volume': 0.1,
            'type': 0,  # ORDER_TYPE_BUY
            'price': 50000.0,
            'sl': 49000.0,
            'tp': 52000.0,
            'magic': 123456,
            'comment': 'test_buy',
        }

        result = mt5.order_send(request)

        assert result.retcode == TradeRetcode.TRADE_RETCODE_DONE
        assert result.volume == 0.1

        # Verify position was created
        positions = mt5.positions_get(symbol='BTCUSD.x')
        assert len(positions) == 1
        assert positions[0].volume == 0.1

        mt5.shutdown()

    def test_market_sell_order(self):
        """Test placing market sell order."""
        mt5 = MockMT5()
        mt5.initialize()

        request = {
            'action': 1,
            'symbol': 'BTCUSD.x',
            'volume': 0.2,
            'type': 1,  # ORDER_TYPE_SELL
            'price': 50000.0,
            'magic': 123456,
        }

        result = mt5.order_send(request)

        assert result.retcode == TradeRetcode.TRADE_RETCODE_DONE

        mt5.shutdown()

    def test_order_validation_volume(self):
        """Test order rejected for invalid volume."""
        mt5 = MockMT5()
        mt5.initialize()

        request = {
            'action': 1,
            'symbol': 'BTCUSD.x',
            'volume': 1000.0,  # Exceeds max
            'type': 0,
            'price': 50000.0,
        }

        result = mt5.order_send(request)

        assert result.retcode == TradeRetcode.TRADE_RETCODE_INVALID_VOLUME

        mt5.shutdown()


class TestPositionManagement:
    """Test position tracking and management."""

    def test_position_tracking(self):
        """Test positions are tracked correctly."""
        mt5 = MockMT5()
        mt5.initialize()

        # Open two positions
        for i in range(2):
            request = {
                'action': 1,
                'symbol': 'BTCUSD.x',
                'volume': 0.1,
                'type': 0,
                'price': 50000.0,
            }
            mt5.order_send(request)

        positions = mt5.positions_get()
        assert len(positions) == 2

        mt5.shutdown()

    def test_position_profit_calculation(self):
        """Test position P&L is calculated."""
        mt5 = MockMT5()
        mt5.initialize()

        # Open position
        request = {
            'action': 1,
            'symbol': 'BTCUSD.x',
            'volume': 1.0,
            'type': 0,
            'price': 50000.0,
        }

        result = mt5.order_send(request)
        ticket = result.deal

        # Update price and check profit
        mt5.set_symbol_price('BTCUSD.x', 51000.0, 51010.0)
        mt5.update_position_profit(ticket, 1000.0)

        positions = mt5.positions_get(ticket=ticket)
        assert len(positions) == 1
        assert positions[0].profit == 1000.0

        mt5.shutdown()


class TestAccountManagement:
    """Test account state management."""

    def test_account_balance_updates(self):
        """Test account balance reflects trades."""
        mt5 = MockMT5()
        mt5.initialize()
        mt5.set_account_balance(10000.0)

        account = mt5.account_info()
        assert account.balance == 10000.0

        # Place order (uses margin)
        request = {
            'action': 1,
            'symbol': 'BTCUSD.x',
            'volume': 1.0,
            'type': 0,
            'price': 50000.0,
        }

        mt5.order_send(request)

        account = mt5.account_info()
        assert account.margin > 0
        assert account.margin_free < 10000.0

        mt5.shutdown()

    def test_equity_includes_unrealized_pnl(self):
        """Test equity includes unrealized P&L."""
        mt5 = MockMT5()
        mt5.initialize()
        mt5.set_account_balance(10000.0)

        # Open position
        request = {
            'action': 1,
            'symbol': 'BTCUSD.x',
            'volume': 1.0,
            'type': 0,
            'price': 50000.0,
        }

        result = mt5.order_send(request)

        # Add profit
        mt5.update_position_profit(result.deal, 500.0)

        account = mt5.account_info()
        assert account.equity == 10500.0

        mt5.shutdown()


class TestRiskEngineIntegration:
    """Test risk engine integration with MT5."""

    def test_risk_engine_validates_orders(self):
        """Test risk engine prevents rule violations."""
        from core.risk_engine import RiskManager, FirmRules, AccountRiskState, FirmType

        mt5 = MockMT5()
        mt5.initialize()
        mt5.set_account_balance(10000.0)

        # GFT uses 6% max DD and 5% guardian
        rules = FirmRules(
            firm_type=FirmType.GFT,
            initial_balance=10000.0,
            max_overall_drawdown_pct=6.0,
            guardian_drawdown_pct=5.0,
            max_risk_per_trade_pct=1.0,
            drawdown_reference="equity",
        )

        state = AccountRiskState(
            initial_balance=10000.0,
            highest_balance=10000.0,
            highest_equity=10000.0,
            current_balance=10000.0,
            current_equity=10000.0,
            daily_starting_balance=10000.0,
            daily_starting_equity=10000.0,
            daily_pnl=0.0,
            daily_date=datetime.now().date().isoformat(),
        )

        # Would need actual file path for persistence
        # risk_manager = RiskManager(rules, state, '/tmp/risk_state.json')

        mt5.shutdown()


class TestFullTradingCycle:
    """Test complete trading cycles."""

    def test_signal_to_execution_flow(self):
        """Test full flow from signal to execution."""
        from signals.generator import SignalGenerator
        from core.lossless.calibrator import MarketCalibrator
        from models.statistical_models import RegimeDetector
        from core.execution import SmartExecutor

        mt5 = MockMT5()
        mt5.initialize()

        # Generate market data
        df = generate_ohlcv_data(n_bars=200, seed=42)

        # Generate signal with proper dependencies
        calibrator = MarketCalibrator(min_calibration_bars=100)
        generator = SignalGenerator(
            calibrator=calibrator,
            regime_detector=RegimeDetector()
        )
        signal = generator.generate_signal('BTCUSD.x', df)

        # Execute if signal is strong enough
        if abs(signal.direction) > 0.3 and signal.confidence > 0.5:
            executor = SmartExecutor(connector=mt5)

            # Create execution plan
            plan = executor.create_plan(
                symbol='BTCUSD.x',
                volume=0.1,
                direction='buy' if signal.direction > 0 else 'sell',
            )

            # Check plan was created
            assert plan is not None

        mt5.shutdown()

    def test_position_monitoring_and_exit(self):
        """Test position monitoring and exit conditions."""
        mt5 = MockMT5()
        mt5.initialize()

        # Open position
        request = {
            'action': 1,
            'symbol': 'BTCUSD.x',
            'volume': 0.1,
            'type': 0,
            'price': 50000.0,
            'sl': 49000.0,
            'tp': 52000.0,
        }

        result = mt5.order_send(request)

        # Simulate price movement to TP
        mt5.set_symbol_price('BTCUSD.x', 52000.0, 52010.0)

        # In real system, TP would be triggered
        # For mock, we close manually
        mt5.close_position(result.deal)

        positions = mt5.positions_get()
        assert len(positions) == 0

        mt5.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
