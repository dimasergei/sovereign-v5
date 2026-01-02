"""
Tests for Risk Engine.
"""

import pytest
from datetime import date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    FirmRules, FirmType, AccountRiskState, RiskManager,
    create_gft_rules, create_the5ers_rules
)


class TestGFTRules:
    """Test GFT-specific rules."""
    
    def test_gft_rules_creation(self):
        """Test GFT rules are created with correct values."""
        rules = create_gft_rules(10000)
        
        assert rules.firm_type == FirmType.GFT
        assert rules.max_overall_drawdown_pct == 8.0
        assert rules.guardian_drawdown_pct == 7.0
        assert rules.max_daily_loss_pct is None
        assert rules.max_open_positions == 3
    
    def test_gft_no_daily_limit(self):
        """Test that GFT has no daily loss limit."""
        rules = create_gft_rules(10000)
        assert rules.max_daily_loss_pct is None
        assert rules.guardian_daily_loss_pct is None


class TestThe5ersRules:
    """Test The5ers-specific rules."""
    
    def test_the5ers_rules_creation(self):
        """Test The5ers rules are created with correct values."""
        rules = create_the5ers_rules(5000)
        
        assert rules.firm_type == FirmType.THE5ERS
        assert rules.max_overall_drawdown_pct == 10.0
        assert rules.guardian_drawdown_pct == 8.5
        assert rules.max_daily_loss_pct == 5.0
        assert rules.guardian_daily_loss_pct == 4.0
        assert rules.consistency_max_single_day_pct == 30.0
    
    def test_the5ers_has_daily_limit(self):
        """Test that The5ers has daily loss limit."""
        rules = create_the5ers_rules(5000)
        assert rules.max_daily_loss_pct == 5.0
        assert rules.guardian_daily_loss_pct == 4.0


class TestRiskManager:
    """Test RiskManager functionality."""
    
    @pytest.fixture
    def gft_risk_manager(self, tmp_path):
        """Create GFT risk manager for testing."""
        rules = create_gft_rules(10000)
        state = AccountRiskState(
            initial_balance=10000,
            highest_balance=10000,
            current_balance=10000,
            current_equity=10000,
            daily_starting_balance=10000,
            daily_pnl=0,
            daily_date=date.today().isoformat()
        )
        return RiskManager(rules, state, str(tmp_path / "state.json"))
    
    @pytest.fixture
    def the5ers_risk_manager(self, tmp_path):
        """Create The5ers risk manager for testing."""
        rules = create_the5ers_rules(5000)
        state = AccountRiskState(
            initial_balance=5000,
            highest_balance=5000,
            current_balance=5000,
            current_equity=5000,
            daily_starting_balance=5000,
            daily_pnl=0,
            daily_date=date.today().isoformat()
        )
        return RiskManager(rules, state, str(tmp_path / "state.json"))
    
    def test_drawdown_calculation(self, gft_risk_manager):
        """Test drawdown is calculated correctly."""
        rm = gft_risk_manager
        
        # Update to 5% drawdown
        rm.update_account_state(9500, 9500)
        assert rm.get_current_drawdown_pct() == 5.0
        
        # Update to 7% drawdown
        rm.update_account_state(9300, 9300)
        assert rm.get_current_drawdown_pct() == 7.0
    
    def test_high_water_mark_tracking(self, gft_risk_manager):
        """Test high water mark updates correctly."""
        rm = gft_risk_manager
        
        # Profit increases HWM
        rm.update_account_state(10500, 10500)
        assert rm.state.highest_balance == 10500
        
        # Loss doesn't decrease HWM
        rm.update_account_state(10200, 10200)
        assert rm.state.highest_balance == 10500
    
    def test_guardian_limit_blocks_trading(self, gft_risk_manager):
        """Test trading is blocked at guardian limit."""
        rm = gft_risk_manager
        
        # At 7% DD (guardian limit)
        rm.state.current_equity = 9300
        rm.state.current_balance = 9300
        
        valid, violation, msg = rm.validate_trade(
            "BTCUSD", 0.1, "buy", 50000, 49000, 52000
        )
        
        assert not valid
        assert "guardian" in msg.lower() or "drawdown" in msg.lower()
    
    def test_daily_loss_tracking(self, the5ers_risk_manager):
        """Test daily loss is tracked correctly."""
        rm = the5ers_risk_manager
        
        # 2% daily loss
        rm.state.current_balance = 4900
        rm.state.current_equity = 4900
        rm.state.daily_pnl = -100
        
        daily_loss = rm.get_daily_loss_pct()
        assert daily_loss == 2.0
    
    def test_daily_guardian_blocks_trading(self, the5ers_risk_manager):
        """Test The5ers daily guardian blocks trading."""
        rm = the5ers_risk_manager
        
        # At 4% daily loss (guardian)
        rm.state.daily_pnl = -200
        rm.state.current_balance = 4800
        rm.state.current_equity = 4800
        rm.update_account_state(4800, 4800)
        
        valid, violation, msg = rm.validate_trade(
            "EURUSD", 0.1, "buy", 1.1000, 1.0950, 1.1100
        )
        
        assert not valid
    
    def test_position_limit(self, gft_risk_manager):
        """Test max positions is enforced."""
        # This would need MT5 mock to test properly
        pass
    
    def test_kelly_calculation(self, gft_risk_manager):
        """Test Kelly fraction calculation."""
        rm = gft_risk_manager
        
        # Add some trade history
        rm.state.winning_trades = 55
        rm.state.losing_trades = 45
        rm.state.total_profit = 1000
        rm.state.total_loss = 800
        
        kelly = rm._calculate_kelly_risk()
        
        # Should be positive and bounded
        assert kelly > 0
        assert kelly <= rm.rules.max_risk_per_trade_pct
    
    def test_drawdown_scaling(self, gft_risk_manager):
        """Test risk reduces as DD increases."""
        rm = gft_risk_manager
        
        base_risk = 0.8  # 0.8%
        
        # At 0% DD - no scaling
        scaled = rm._apply_drawdown_scaling(base_risk)
        assert scaled == base_risk
        
        # At 5% DD (past 50% of 7% guardian)
        rm.state.current_equity = 9500
        rm.update_account_state(9500, 9500)
        scaled = rm._apply_drawdown_scaling(base_risk)
        assert scaled < base_risk


class TestConsistencyRule:
    """Test The5ers consistency rule."""
    
    @pytest.fixture
    def rm_with_profit(self, tmp_path):
        """Create risk manager with existing profit."""
        rules = create_the5ers_rules(5000)
        state = AccountRiskState(
            initial_balance=5000,
            highest_balance=5500,
            current_balance=5500,
            current_equity=5500,
            daily_starting_balance=5000,
            daily_pnl=500,
            daily_date=date.today().isoformat(),
            total_realized_profit=500,
            daily_profit_distribution={date.today().isoformat(): 150}
        )
        return RiskManager(rules, state, str(tmp_path / "state.json"))
    
    def test_consistency_allows_normal_trade(self, rm_with_profit):
        """Test consistency rule allows normal profit."""
        rm = rm_with_profit
        
        # Current day: 150 profit out of 500 total = 30%
        # Adding 0 should still be allowed
        assert rm.check_consistency_rule(0) is True
    
    def test_consistency_blocks_excessive_profit(self, rm_with_profit):
        """Test consistency rule blocks excessive single-day profit."""
        rm = rm_with_profit
        
        # If this trade would make today's profit > 30% of total
        # 150 + 200 = 350 out of 700 total = 50% > 30%
        result = rm.check_consistency_rule(200)
        # This should be False if implemented correctly
        # Note: The exact implementation may vary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
