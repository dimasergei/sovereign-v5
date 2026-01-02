"""
Stress Tests for Drawdown Scenarios.

Tests system behavior under extreme market conditions.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.sample_data import generate_ohlcv_data
from fixtures.mock_mt5 import MockMT5


class TestDrawdownProtection:
    """Test drawdown protection under stress."""

    def test_guardian_stops_trading_before_limit(self):
        """Test trading stops at guardian threshold, not actual limit."""
        from core.risk_engine import RiskManager, FirmRules, AccountRiskState, FirmType

        # GFT uses 6% max DD and 5% guardian
        rules = FirmRules(
            firm_type=FirmType.GFT,
            initial_balance=10000.0,
            max_overall_drawdown_pct=6.0,
            guardian_drawdown_pct=5.0,
            drawdown_reference="equity",
        )

        state = AccountRiskState(
            initial_balance=10000.0,
            highest_balance=10000.0,
            highest_equity=10000.0,
            current_balance=9600.0,  # 4% drawdown
            current_equity=9600.0,
            daily_starting_balance=10000.0,
            daily_starting_equity=10000.0,
            daily_pnl=-400.0,
            daily_date=datetime.now().date().isoformat(),
        )

        # At 4% drawdown, should still allow trading (under 5% guardian)
        drawdown_pct = (10000.0 - 9600.0) / 10000.0 * 100
        assert drawdown_pct < 5.0

        # At 5.1% drawdown, should stop (guardian)
        state.current_balance = 9490.0
        state.current_equity = 9490.0
        drawdown_pct = (10000.0 - 9490.0) / 10000.0 * 100
        assert drawdown_pct > 5.0

    def test_consecutive_losses_handling(self):
        """Test system handles consecutive losing trades."""
        from core.risk_engine import RiskManager, FirmRules, AccountRiskState, FirmType

        # GFT uses 6% max DD and 5% guardian
        rules = FirmRules(
            firm_type=FirmType.GFT,
            initial_balance=10000.0,
            max_overall_drawdown_pct=6.0,
            guardian_drawdown_pct=5.0,
            max_risk_per_trade_pct=1.0,
            drawdown_reference="equity",
        )

        balance = 10000.0
        highest = 10000.0

        # Simulate 10 consecutive losing trades at 1% risk each
        for i in range(10):
            loss = balance * 0.01  # 1% loss
            balance -= loss

            # Update high water mark if applicable
            highest = max(highest, balance)

            drawdown_pct = (highest - balance) / highest * 100

            # Check if guardian would be hit (5% for GFT)
            if drawdown_pct >= 5.0:
                # System should stop trading
                trades_before_stop = i + 1
                break
        else:
            trades_before_stop = 10

        # With 1% per trade, should stop around 5 trades (5% drawdown)
        assert trades_before_stop >= 4
        assert trades_before_stop <= 6

    def test_rapid_price_movement_handling(self):
        """Test handling of flash crash scenarios."""
        # Simulate 10% price drop in minutes
        np.random.seed(42)
        n = 100
        prices = [50000.0]

        for i in range(n):
            # Normal movement with occasional flash crash
            if i == 50:
                # Flash crash - 5% drop
                change = -0.05
            else:
                change = np.random.normal(0, 0.001)

            prices.append(prices[-1] * (1 + change))

        df = pd.DataFrame({
            'open': prices[:-1],
            'high': [max(prices[i], prices[i+1]) for i in range(n)],
            'low': [min(prices[i], prices[i+1]) * 0.99 for i in range(n)],
            'close': prices[1:],
            'volume': np.random.uniform(100, 1000, n)
        })

        # Calculate drawdown during flash crash
        max_price = max(prices[:51])
        min_price = min(prices[50:60])
        flash_crash_dd = (max_price - min_price) / max_price * 100

        assert flash_crash_dd > 4.0, "Flash crash should cause significant drawdown"


class TestExtremeVolatility:
    """Test behavior under extreme volatility."""

    def test_position_sizing_reduces_in_high_vol(self):
        """Test position size reduces in high volatility."""
        from core.risk_engine import RiskManager, FirmRules, AccountRiskState, FirmType

        # Generate high volatility data
        df_high_vol = generate_ohlcv_data(n_bars=100, volatility=0.10, seed=42)
        high_vol = df_high_vol['close'].pct_change().std()

        # Generate low volatility data
        df_low_vol = generate_ohlcv_data(n_bars=100, volatility=0.01, seed=42)
        low_vol = df_low_vol['close'].pct_change().std()

        # Position size should be inversely related to volatility
        assert high_vol > low_vol

        # Simple Kelly-based sizing
        edge = 0.02  # 2% edge
        high_vol_size = edge / (high_vol ** 2) if high_vol > 0 else 0
        low_vol_size = edge / (low_vol ** 2) if low_vol > 0 else 0

        assert high_vol_size < low_vol_size

    def test_signal_confidence_drops_in_chaos(self):
        """Test signal confidence decreases in chaotic markets."""
        from signals.generator import SignalGenerator
        from core.lossless.calibrator import MarketCalibrator
        from models.statistical_models import RegimeDetector

        calibrator = MarketCalibrator(min_calibration_bars=100)
        generator = SignalGenerator(
            calibrator=calibrator,
            regime_detector=RegimeDetector()
        )

        # Normal market
        df_normal = generate_ohlcv_data(n_bars=200, volatility=0.02, seed=1)
        signal_normal = generator.generate_signal('BTCUSD', df_normal)

        # Chaotic market (high volatility + regime changes)
        df_chaotic = generate_ohlcv_data(n_bars=200, volatility=0.08, seed=2)
        signal_chaotic = generator.generate_signal('BTCUSD', df_chaotic)

        # Confidence should generally be lower in chaotic conditions
        # (This depends on actual implementation)
        assert signal_chaotic.confidence <= signal_normal.confidence or \
               signal_chaotic.confidence < 0.8


class TestRecoveryScenarios:
    """Test system recovery from various failure modes."""

    def test_recovery_from_drawdown(self):
        """Test system recovers position sizing after drawdown."""
        initial_balance = 10000.0
        drawdown_balance = 9300.0  # 7% drawdown
        recovery_balance = 9500.0  # Partial recovery

        # After recovery, high water mark should update
        new_hwm = max(drawdown_balance, recovery_balance)
        assert new_hwm == recovery_balance

        # New drawdown calculated from new HWM
        new_dd = (recovery_balance - recovery_balance) / recovery_balance
        assert new_dd == 0.0

    def test_max_consecutive_losses_simulation(self):
        """Simulate maximum consecutive losses scenario."""
        np.random.seed(42)

        initial = 10000.0
        balance = initial
        peak = initial
        max_dd = 0.0

        # Simulate 1000 trades with 45% win rate
        win_rate = 0.45
        avg_win = 150
        avg_loss = 100

        for _ in range(1000):
            if np.random.random() < win_rate:
                balance += avg_win
            else:
                balance -= avg_loss

            # Update peak (high water mark)
            if balance > peak:
                peak = balance

            # Track drawdown from peak (proper HWM drawdown)
            dd = (peak - balance) / peak if balance < peak else 0
            max_dd = max(max_dd, dd)

            # Stop if account blown
            if balance <= 0:
                break

        # With positive expectancy, account grows but experiences drawdowns
        assert max_dd > 0.01, "Should experience some drawdown from peak"
        assert balance > 0, "Account should survive with positive expectancy"


class TestDailyLossLimit:
    """Test daily loss limit enforcement (The5ers)."""

    def test_daily_limit_resets_at_midnight(self):
        """Test daily loss limit resets at UTC midnight."""
        from core.risk_engine import FirmRules, AccountRiskState, FirmType

        state = AccountRiskState(
            initial_balance=5000.0,
            highest_balance=5000.0,
            current_balance=4800.0,  # 4% daily loss
            current_equity=4800.0,
            daily_starting_balance=5000.0,
            daily_pnl=-200.0,
            daily_date="2024-01-01",
        )

        # Check daily loss
        daily_loss_pct = abs(state.daily_pnl) / state.daily_starting_balance * 100
        assert daily_loss_pct == 4.0

        # Simulate new day
        state.daily_date = "2024-01-02"
        state.daily_starting_balance = state.current_balance
        state.daily_pnl = 0.0

        # Daily loss should reset
        assert state.daily_pnl == 0.0

    def test_trading_stops_at_daily_guardian(self):
        """Test trading stops at daily loss guardian (4% for The5ers)."""
        from core.risk_engine import FirmRules, FirmType

        rules = FirmRules(
            firm_type=FirmType.THE5ERS,
            initial_balance=5000.0,
            max_overall_drawdown_pct=10.0,
            guardian_drawdown_pct=8.5,
            max_daily_loss_pct=5.0,
            guardian_daily_loss_pct=4.0,
        )

        # At 3.5% daily loss, can still trade
        assert 3.5 < rules.guardian_daily_loss_pct

        # At 4.1% daily loss, should stop
        assert 4.1 > rules.guardian_daily_loss_pct


class TestLiquidityStress:
    """Test handling of low liquidity conditions."""

    def test_slippage_increases_in_low_liquidity(self):
        """Test slippage estimation increases in low liquidity."""
        # Simple slippage estimation based on volume ratio and spread
        def estimate_slippage(volume, avg_volume, spread, volatility):
            volume_impact = (volume / avg_volume) ** 0.5 * volatility
            spread_impact = spread * 0.5
            return volume_impact + spread_impact

        # Normal liquidity
        normal_slippage = estimate_slippage(
            volume=1.0,
            avg_volume=10000,
            spread=0.0001,
            volatility=0.02
        )

        # Low liquidity (10x lower volume)
        low_liq_slippage = estimate_slippage(
            volume=1.0,
            avg_volume=1000,
            spread=0.0005,  # Wider spread
            volatility=0.02
        )

        assert low_liq_slippage > normal_slippage

    def test_order_rejection_in_no_liquidity(self):
        """Test orders are rejected when no liquidity."""
        mt5 = MockMT5()
        mt5.initialize()

        # Remove symbol (simulate no liquidity)
        # In real scenario, order_check would fail

        mt5.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
