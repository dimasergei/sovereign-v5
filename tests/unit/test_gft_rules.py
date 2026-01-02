"""
Tests for GFT Instant Funding GOAT Model rules.

These tests verify that the risk engine correctly implements
the GFT rules which have HARD BREACH conditions that cause
immediate account closure.

CRITICAL LIMITS (as of 2024):
- 6% max total drawdown (trailing from equity HWM)
- 3% max daily drawdown (resets 5 PM EST)
- 2% max floating loss per trade (HARD BREACH - immediate closure!)
- 15% consistency rule (blocks payout, not hard breach)
- 5 minimum trading days @ 0.5% profit each for payout
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, date, timedelta
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.risk_engine import (
    RiskManager,
    FirmRules,
    AccountRiskState,
    FirmType,
    ViolationType,
    create_gft_rules,
)


class TestGFTRulesConfiguration:
    """Test that GFT rules are configured correctly."""

    def test_gft_rules_limits(self):
        """Verify GFT limits match actual firm requirements."""
        rules = create_gft_rules(initial_balance=10000)

        # Total drawdown - CRITICAL: 6%, not 8%
        assert rules.max_overall_drawdown_pct == 6.0
        assert rules.guardian_drawdown_pct == 5.0
        assert rules.drawdown_reference == "equity"

        # Daily drawdown - CRITICAL: 3%, resets 5 PM EST
        assert rules.max_daily_loss_pct == 3.0
        assert rules.guardian_daily_loss_pct == 2.5
        assert rules.daily_reset_time == "17:00"
        assert rules.daily_reset_timezone == "US/Eastern"

        # Per-trade floating loss (CRITICAL - HARD BREACH)
        assert rules.max_trade_floating_loss_pct == 2.0
        assert rules.guardian_trade_floating_loss_pct == 1.5

        # Consistency rule
        assert rules.consistency_max_single_day_pct == 15.0
        assert rules.consistency_is_hard_breach == False

        # Payout requirements
        assert rules.min_trading_days_for_payout == 5
        assert rules.min_profit_per_trading_day_pct == 0.5

    def test_gft_guardian_buffer(self):
        """Verify guardian limits have appropriate safety buffer."""
        rules = create_gft_rules(initial_balance=10000)

        # At least 0.5% buffer on all critical limits
        total_dd_buffer = rules.max_overall_drawdown_pct - rules.guardian_drawdown_pct
        assert total_dd_buffer >= 0.5, f"Total DD buffer {total_dd_buffer}% < 0.5%"

        daily_dd_buffer = rules.max_daily_loss_pct - rules.guardian_daily_loss_pct
        assert daily_dd_buffer >= 0.5, f"Daily DD buffer {daily_dd_buffer}% < 0.5%"

        trade_loss_buffer = rules.max_trade_floating_loss_pct - rules.guardian_trade_floating_loss_pct
        assert trade_loss_buffer >= 0.5, f"Trade loss buffer {trade_loss_buffer}% < 0.5%"

    def test_gft_uses_equity_not_balance(self):
        """Verify GFT uses equity for drawdown, not balance."""
        rules = create_gft_rules(initial_balance=10000)
        assert rules.drawdown_reference == "equity"

    def test_gft_daily_reset_timezone(self):
        """Verify daily reset uses correct timezone."""
        rules = create_gft_rules(initial_balance=10000)
        assert rules.daily_reset_timezone == "US/Eastern"
        assert rules.daily_reset_time == "17:00"


class TestGFTDrawdownCalculation:
    """Test that drawdown uses equity, not balance."""

    @pytest.fixture
    def gft_risk_manager(self, tmp_path):
        rules = create_gft_rules(initial_balance=10000)
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
            highest_equity=10000,
            current_balance=10000,
            current_equity=10000,
            daily_starting_balance=10000,
            daily_starting_equity=10000,
            daily_pnl=0,
            daily_date=datetime.now().strftime("%Y-%m-%d")
        )
        return RiskManager(
            rules=rules,
            state=state,
            state_file=str(tmp_path / "state.json")
        )

    def test_drawdown_from_equity_hwm(self, gft_risk_manager):
        """Verify drawdown calculated from equity high water mark."""
        gft_risk_manager.state.highest_equity = 10500
        gft_risk_manager.state.current_equity = 10000

        dd = gft_risk_manager.get_current_drawdown_pct()

        # (10500 - 10000) / 10500 = 4.76%
        expected = (10500 - 10000) / 10500 * 100
        assert abs(dd - expected) < 0.01, f"Expected {expected:.2f}%, got {dd:.2f}%"

    def test_drawdown_ignores_balance(self, gft_risk_manager):
        """Verify drawdown ignores balance when reference is equity."""
        gft_risk_manager.state.highest_balance = 11000  # Higher balance
        gft_risk_manager.state.highest_equity = 10000
        gft_risk_manager.state.current_balance = 10500
        gft_risk_manager.state.current_equity = 9500

        dd = gft_risk_manager.get_current_drawdown_pct()

        # Should use equity, not balance
        expected = (10000 - 9500) / 10000 * 100  # 5%
        assert abs(dd - expected) < 0.01, f"Expected {expected:.2f}%, got {dd:.2f}%"


class TestGFTDailyLossLimit:
    """Test 3% daily loss limit."""

    @pytest.fixture
    def gft_risk_manager(self, tmp_path):
        rules = create_gft_rules(initial_balance=10000)
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
            highest_equity=10000,
            current_balance=10000,
            current_equity=10000,
            daily_starting_balance=10000,
            daily_starting_equity=10000,
            daily_pnl=0,
            daily_date=datetime.now().strftime("%Y-%m-%d")
        )
        return RiskManager(
            rules=rules,
            state=state,
            state_file=str(tmp_path / "state.json")
        )

    def test_daily_loss_limit_exists(self, gft_risk_manager):
        """Verify daily loss limit is set."""
        assert gft_risk_manager.rules.max_daily_loss_pct == 3.0
        assert gft_risk_manager.rules.guardian_daily_loss_pct == 2.5

    def test_daily_loss_blocks_trading(self, gft_risk_manager):
        """Verify exceeding daily guardian blocks trading."""
        # Simulate 2.5% daily loss
        gft_risk_manager.state.daily_starting_equity = 10000
        gft_risk_manager.state.current_equity = 9750
        gft_risk_manager.state.daily_pnl = -250

        valid, violation, msg = gft_risk_manager.validate_trade(
            "BTCUSD.x", 0.1, "buy", 50000, 49000, 52000
        )

        assert not valid
        assert violation == ViolationType.DAILY_LOSS_EXCEEDED


class TestGFTPayoutEligibility:
    """Test payout eligibility rules."""

    @pytest.fixture
    def gft_risk_manager(self, tmp_path):
        rules = create_gft_rules(initial_balance=10000)
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
            highest_equity=10000,
            current_balance=10000,
            current_equity=10000,
            daily_starting_balance=10000,
            daily_starting_equity=10000,
            daily_pnl=0,
            daily_date=datetime.now().strftime("%Y-%m-%d")
        )
        return RiskManager(
            rules=rules,
            state=state,
            state_file=str(tmp_path / "state.json")
        )

    def test_minimum_trading_days(self, gft_risk_manager):
        """Test 5 minimum trading days requirement."""
        gft_risk_manager.state.qualifying_trading_days = 4
        gft_risk_manager._update_payout_eligibility()

        assert gft_risk_manager.state.payout_blocked == True
        assert "5 trading days" in gft_risk_manager.state.payout_blocked_reason

    def test_minimum_trading_days_met(self, gft_risk_manager):
        """Test payout allowed when min days met."""
        gft_risk_manager.state.qualifying_trading_days = 5
        gft_risk_manager.state.total_realized_profit = 500
        gft_risk_manager.state.daily_profit_distribution = {
            "2024-01-01": 100,
            "2024-01-02": 100,
            "2024-01-03": 100,
            "2024-01-04": 100,
            "2024-01-05": 100,
        }
        gft_risk_manager._update_payout_eligibility()

        assert gft_risk_manager.state.payout_blocked == False

    def test_consistency_rule_blocks_payout(self, gft_risk_manager):
        """Test 15% consistency rule blocks payout but doesn't close account."""
        gft_risk_manager.state.qualifying_trading_days = 5
        gft_risk_manager.state.total_realized_profit = 1000
        gft_risk_manager.state.daily_profit_distribution = {
            "2024-01-01": 200,  # 20% of total - violates rule
            "2024-01-02": 100,
            "2024-01-03": 200,
            "2024-01-04": 300,
            "2024-01-05": 200,
        }

        gft_risk_manager._update_payout_eligibility()

        assert gft_risk_manager.state.payout_blocked == True
        assert "15%" in gft_risk_manager.state.payout_blocked_reason
        assert gft_risk_manager.state.is_locked == False  # Account NOT closed


class TestGFTTradeValidation:
    """Test trade validation with new GFT rules."""

    @pytest.fixture
    def gft_risk_manager(self, tmp_path):
        rules = create_gft_rules(initial_balance=10000)
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
            highest_equity=10000,
            current_balance=10000,
            current_equity=10000,
            daily_starting_balance=10000,
            daily_starting_equity=10000,
            daily_pnl=0,
            daily_date=datetime.now().strftime("%Y-%m-%d")
        )
        return RiskManager(
            rules=rules,
            state=state,
            state_file=str(tmp_path / "state.json")
        )

    def test_trade_rejected_at_5pct_dd(self, gft_risk_manager):
        """Verify trades rejected at 5% guardian (not old 7%)."""
        gft_risk_manager.state.highest_equity = 10000
        gft_risk_manager.state.current_equity = 9500  # 5% DD

        valid, violation, msg = gft_risk_manager.validate_trade(
            "BTCUSD.x", 0.1, "buy", 50000, 49000, 52000
        )

        assert not valid
        assert violation == ViolationType.MAX_DRAWDOWN_EXCEEDED

    def test_trade_allowed_under_guardian(self, gft_risk_manager):
        """Verify trades allowed under guardian limit."""
        gft_risk_manager.state.highest_equity = 10000
        gft_risk_manager.state.current_equity = 9600  # 4% DD, under 5% guardian
        gft_risk_manager.state.daily_starting_equity = 9600
        gft_risk_manager.state.daily_pnl = 0

        valid, violation, msg = gft_risk_manager.validate_trade(
            "BTCUSD.x", 0.1, "buy", 50000, 49000, 52000
        )

        # Should pass drawdown check (may fail other checks)
        if not valid:
            assert violation != ViolationType.MAX_DRAWDOWN_EXCEEDED


class TestGFTViolationTypes:
    """Test new GFT-specific violation types exist."""

    def test_trade_floating_loss_violation_exists(self):
        """Verify TRADE_FLOATING_LOSS_EXCEEDED violation type exists."""
        assert hasattr(ViolationType, 'TRADE_FLOATING_LOSS_EXCEEDED')

    def test_trade_risk_too_high_violation_exists(self):
        """Verify TRADE_RISK_TOO_HIGH violation type exists."""
        assert hasattr(ViolationType, 'TRADE_RISK_TOO_HIGH')


class TestGFTAccountRiskState:
    """Test new AccountRiskState fields."""

    def test_highest_equity_field_exists(self):
        """Verify highest_equity field exists."""
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
        )
        assert hasattr(state, 'highest_equity')

    def test_daily_starting_equity_field_exists(self):
        """Verify daily_starting_equity field exists."""
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
        )
        assert hasattr(state, 'daily_starting_equity')

    def test_qualifying_trading_days_field_exists(self):
        """Verify qualifying_trading_days field exists."""
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
        )
        assert hasattr(state, 'qualifying_trading_days')

    def test_payout_blocked_field_exists(self):
        """Verify payout_blocked field exists."""
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
        )
        assert hasattr(state, 'payout_blocked')


class TestGFTFirmRulesFields:
    """Test new FirmRules fields for GFT."""

    def test_drawdown_reference_field_exists(self):
        """Verify drawdown_reference field exists."""
        rules = create_gft_rules(initial_balance=10000)
        assert hasattr(rules, 'drawdown_reference')
        assert rules.drawdown_reference == "equity"

    def test_daily_reset_timezone_field_exists(self):
        """Verify daily_reset_timezone field exists."""
        rules = create_gft_rules(initial_balance=10000)
        assert hasattr(rules, 'daily_reset_timezone')
        assert rules.daily_reset_timezone == "US/Eastern"

    def test_max_trade_floating_loss_field_exists(self):
        """Verify max_trade_floating_loss_pct field exists."""
        rules = create_gft_rules(initial_balance=10000)
        assert hasattr(rules, 'max_trade_floating_loss_pct')
        assert rules.max_trade_floating_loss_pct == 2.0

    def test_consistency_is_hard_breach_field_exists(self):
        """Verify consistency_is_hard_breach field exists."""
        rules = create_gft_rules(initial_balance=10000)
        assert hasattr(rules, 'consistency_is_hard_breach')
        assert rules.consistency_is_hard_breach == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
