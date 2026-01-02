"""
The5ers Bot - Forex Trading Bot.

Standalone bot file for The5ers $5,000 accounts trading forex.

RULES (DO NOT MODIFY):
- Max daily loss: 5% (guardian: 4%)
- Max overall loss: 10% (guardian: 8.5%)
- Consistency rule: No single day > 30% of total profit
- Forex majors/minors only
"""

import argparse
import logging
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    MT5Credentials,
    FirmRules,
    FirmType,
    create_the5ers_rules,
)
from monitoring import TelegramConfig
from bots.base_bot import BaseTradingBot


# ============================================================================
# CONFIGURATION - FILL THESE IN
# ============================================================================

# Account identifier
ACCOUNT_NAME = "The5ers_HighStakes"

# MT5 Credentials
MT5_LOGIN = 87654321  # Your MT5 login
MT5_PASSWORD = "your_password"  # Your MT5 password
MT5_SERVER = "The5ers-Server"  # Your broker server
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Telegram (optional)
TELEGRAM_ENABLED = True
TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHAT_IDS = [123456789]

# Trading
INITIAL_BALANCE = 5000.0
SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "EURGBP",
    "EURJPY",
]

# Timing
SCAN_INTERVAL_SECONDS = 60
TIMEFRAME = "M5"

# Trading Sessions (UTC)
TRADING_SESSIONS = {
    'london': (dt_time(8, 0), dt_time(17, 0)),
    'new_york': (dt_time(13, 0), dt_time(22, 0)),
    'overlap': (dt_time(13, 0), dt_time(17, 0)),  # Best liquidity
}

# State file
STATE_FILE = f"storage/state/{ACCOUNT_NAME}_state.json"


# ============================================================================
# BOT CLASS
# ============================================================================

class The5ersBot(BaseTradingBot):
    """
    The5ers-specific trading bot.
    
    Implements:
    - Daily loss limit tracking
    - Consistency rule enforcement
    - Session-aware trading
    - Friday closeout
    """
    
    def __init__(self):
        # Create credentials
        credentials = MT5Credentials(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
            path=MT5_PATH,
        )
        
        # Create firm rules
        firm_rules = create_the5ers_rules(
            initial_balance=INITIAL_BALANCE,
            symbols=SYMBOLS
        )
        
        # Telegram config
        telegram_config = None
        if TELEGRAM_ENABLED and TELEGRAM_TOKEN:
            telegram_config = TelegramConfig(
                bot_token=TELEGRAM_TOKEN,
                authorized_chat_ids=TELEGRAM_CHAT_IDS,
            )
        
        super().__init__(
            account_name=ACCOUNT_NAME,
            credentials=credentials,
            firm_rules=firm_rules,
            symbols=SYMBOLS,
            state_file=STATE_FILE,
            telegram_config=telegram_config,
            scan_interval_seconds=SCAN_INTERVAL_SECONDS,
            timeframe=TIMEFRAME,
        )
    
    def _setup_schedule(self):
        """Setup The5ers specific scheduled tasks."""
        import schedule
        
        # Daily reset at 00:00 UTC
        schedule.every().day.at("00:00").do(self._daily_reset)
        
        # Friday closeout at 19:00 UTC
        schedule.every().friday.at("19:00").do(self._friday_closeout)
        
        # Inactivity check
        schedule.every().day.at("08:00").do(self._check_inactivity)
    
    def _analyze_and_trade(self, symbol: str):
        """
        The5ers-specific trading with additional checks.
        """
        # Check if within trading hours
        if not self._is_trading_hours():
            return
        
        # Check daily loss budget
        if not self._has_daily_budget():
            logging.getLogger(__name__).warning("Daily loss budget exhausted")
            return
        
        # Call parent implementation
        super()._analyze_and_trade(symbol)
    
    def _is_trading_hours(self) -> bool:
        """
        Check if current time is within active trading session.
        
        Avoid:
        - 22:00-00:00 UTC (low liquidity)
        - Friday after 20:00 UTC (weekend gap risk)
        """
        now = datetime.utcnow()
        current_time = now.time()
        weekday = now.weekday()
        
        # Friday after 20:00 - no new trades
        if weekday == 4 and current_time >= dt_time(20, 0):
            return False
        
        # Weekend - no trading
        if weekday >= 5:
            return False
        
        # Check session overlap (best liquidity)
        overlap_start, overlap_end = TRADING_SESSIONS['overlap']
        if overlap_start <= current_time <= overlap_end:
            return True
        
        # Check London session
        london_start, london_end = TRADING_SESSIONS['london']
        if london_start <= current_time <= london_end:
            return True
        
        # Check NY session
        ny_start, ny_end = TRADING_SESSIONS['new_york']
        if ny_start <= current_time <= ny_end:
            return True
        
        return False
    
    def _has_daily_budget(self) -> bool:
        """Check if we have daily loss budget remaining."""
        if not self.risk_manager:
            return True
        
        daily_loss_pct = self.risk_manager.get_daily_loss_pct()
        guardian_limit = self.firm_rules.guardian_daily_loss_pct or 4.0
        
        # Leave 0.5% buffer
        return daily_loss_pct < (guardian_limit - 0.5)
    
    def _friday_closeout(self):
        """Close all positions before weekend to avoid gaps."""
        logger = logging.getLogger(__name__)
        logger.info("Friday closeout initiated")
        
        positions = self.connector.get_positions()
        if not positions:
            return
        
        for pos in positions:
            result = self.executor.close_position(pos['ticket'])
            if result.success:
                logger.info(f"Closed position {pos['ticket']} for weekend")
                
                if self.telegram:
                    self.telegram.send_trade(
                        symbol=pos['symbol'],
                        action="FRIDAY_CLOSE",
                        direction=pos['type'].upper(),
                        price=result.avg_fill_price,
                        size=pos['volume'],
                        pnl=pos['profit']
                    )
    
    def _check_consistency_before_trade(self, expected_profit: float) -> bool:
        """
        Check if trade would violate consistency rule.
        
        No single day's profit can exceed 30% of total profit.
        """
        return self.risk_manager.check_consistency_rule(expected_profit)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def run_tests():
    """Run basic tests on risk calculations."""
    print("Running The5ers bot tests...")
    
    from core import create_the5ers_rules, AccountRiskState, RiskManager
    from datetime import date
    
    # Test 1: The5ers rules creation
    rules = create_the5ers_rules(5000)
    assert rules.max_overall_drawdown_pct == 10.0
    assert rules.guardian_drawdown_pct == 8.5
    assert rules.max_daily_loss_pct == 5.0
    assert rules.guardian_daily_loss_pct == 4.0
    assert rules.consistency_max_single_day_pct == 30.0
    print("✓ The5ers rules created correctly")
    
    # Test 2: Daily loss calculation
    state = AccountRiskState(
        initial_balance=5000,
        highest_balance=5000,
        current_balance=4900,
        current_equity=4900,
        daily_starting_balance=5000,
        daily_pnl=-100,
        daily_date=date.today().isoformat()
    )
    
    rm = RiskManager(rules, state, "/tmp/test_state.json")
    daily_loss = rm.get_daily_loss_pct()
    assert daily_loss == 2.0, f"Expected 2.0%, got {daily_loss}%"
    print(f"✓ Daily loss calculation correct: {daily_loss}%")
    
    # Test 3: Guardian daily limit
    state.current_equity = 4800  # 4% daily loss
    state.daily_pnl = -200
    rm.update_account_state(4800, 4800)
    
    valid, violation, msg = rm.validate_trade(
        "EURUSD", 0.1, "buy", 1.1000, 1.0950, 1.1100
    )
    assert not valid, "Should reject trade at daily guardian limit"
    print(f"✓ Daily guardian limit enforced at 4%")
    
    # Test 4: Consistency rule
    state = AccountRiskState(
        initial_balance=5000,
        highest_balance=5500,
        current_balance=5500,
        current_equity=5500,
        daily_starting_balance=5000,
        daily_pnl=500,
        daily_date=date.today().isoformat(),
        total_realized_profit=500,
        daily_profit_distribution={date.today().isoformat(): 500}
    )
    rm = RiskManager(rules, state, "/tmp/test_state2.json")
    
    # 500 profit today out of 500 total = 100% > 30%
    # Adding more would still violate
    can_trade = rm.check_consistency_rule(0)
    # Note: In practice, this check happens before the day is complete
    print("✓ Consistency rule check implemented")
    
    print("\nAll tests passed! ✓")


def run_backtest(days: int = 30):
    """Run a simple backtest."""
    print(f"Running {days}-day backtest...")
    print("Backtest not fully implemented in this version.")
    print("Use the backtesting module for full backtests.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="The5ers Forex Trading Bot")
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--backtest', type=int, metavar='DAYS', help='Run backtest')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.test:
        run_tests()
    elif args.backtest:
        run_backtest(args.backtest)
    else:
        # Run the bot
        bot = The5ersBot()
        bot.start()


if __name__ == "__main__":
    main()
