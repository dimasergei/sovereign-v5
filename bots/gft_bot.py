"""
GFT Bot - Goat Funded Trader Crypto Bot.

Standalone bot file for GFT $10,000 accounts trading crypto CFDs.

RULES (DO NOT MODIFY):
- Max overall drawdown: 8% (trailing from high water mark)
- Guardian limit: 7% (stop trading before actual limit)
- No daily loss limit
- No hedging, martingale, or HFT
- Max inactivity: 30 days
- Crypto CFDs only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    MT5Credentials,
    FirmRules,
    FirmType,
    create_gft_rules,
)
from monitoring import TelegramConfig
from bots.base_bot import BaseTradingBot


# ============================================================================
# CONFIGURATION - FILL THESE IN
# ============================================================================

# Account identifier
ACCOUNT_NAME = "GFT_Account1"

# MT5 Credentials
MT5_LOGIN = 12345678  # Your MT5 login
MT5_PASSWORD = "your_password"  # Your MT5 password
MT5_SERVER = "GoatFundedTrader-Server"  # Your broker server
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Telegram (optional)
TELEGRAM_ENABLED = True
TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHAT_IDS = [123456789]  # Your Telegram chat ID(s)

# Trading
INITIAL_BALANCE = 10000.0
SYMBOLS = [
    "BTCUSD.x",
    "ETHUSD.x",
    "SOLUSD.x",
    "XRPUSD.x",
    "LTCUSD.x",
]

# Timing
SCAN_INTERVAL_SECONDS = 60
TIMEFRAME = "M5"

# State file
STATE_FILE = f"storage/state/{ACCOUNT_NAME}_state.json"


# ============================================================================
# BOT CLASS
# ============================================================================

class GFTBot(BaseTradingBot):
    """
    GFT-specific trading bot.
    
    Implements GFT-specific rules and overrides.
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
        firm_rules = create_gft_rules(
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
    
    def _check_inactivity(self):
        """
        GFT-specific inactivity check.
        
        GFT requires at least one trade every 30 days.
        We warn at 25 days.
        """
        super()._check_inactivity()
        
        days_inactive = self.risk_manager.state.days_since_last_trade
        
        if days_inactive >= 28:
            # Critical - place a minimal trade
            self._place_ping_trade()
    
    def _place_ping_trade(self):
        """Place minimal trade to avoid inactivity violation."""
        logger = logging.getLogger(__name__)
        logger.warning("Placing ping trade to avoid inactivity violation")
        
        # Find most liquid symbol
        symbol = self.symbols[0] if self.symbols else "BTCUSD.x"
        
        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)
        if not symbol_info:
            return
        
        min_lot = symbol_info.get('volume_min', 0.01)
        
        # Get current price
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return
        
        # Place tiny long with tight SL/TP (will likely close quickly)
        entry = tick.ask
        sl = entry - (entry * 0.001)  # 0.1% SL
        tp = entry + (entry * 0.001)  # 0.1% TP
        
        plan = self.executor.create_plan(
            symbol=symbol,
            direction="buy",
            size=min_lot,
            stop_loss=sl,
            take_profit=tp,
            comment="ping_trade"
        )
        
        result = self.executor.execute(plan)
        
        if result.success:
            logger.info("Ping trade placed successfully")
            if self.telegram:
                self.telegram.send_alert(
                    f"📍 Ping trade placed to avoid inactivity\n"
                    f"Symbol: {symbol}\n"
                    f"Size: {min_lot}",
                    level="info"
                )


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def run_tests():
    """Run basic tests on risk calculations."""
    print("Running GFT bot tests...")
    
    from core import create_gft_rules, AccountRiskState, RiskManager
    from datetime import date
    
    # Test 1: GFT rules creation
    rules = create_gft_rules(10000)
    assert rules.max_overall_drawdown_pct == 8.0
    assert rules.guardian_drawdown_pct == 7.0
    assert rules.max_daily_loss_pct is None
    print("✓ GFT rules created correctly")
    
    # Test 2: Drawdown calculation
    state = AccountRiskState(
        initial_balance=10000,
        highest_balance=10000,
        current_balance=9500,
        current_equity=9500,
        daily_starting_balance=10000,
        daily_pnl=-500,
        daily_date=date.today().isoformat()
    )
    
    rm = RiskManager(rules, state, "/tmp/test_state.json")
    dd = rm.get_current_drawdown_pct()
    assert dd == 5.0, f"Expected 5.0%, got {dd}%"
    print(f"✓ Drawdown calculation correct: {dd}%")
    
    # Test 3: Guardian limit
    state.current_equity = 9300  # 7% DD
    rm.update_account_state(9300, 9300)
    
    valid, violation, msg = rm.validate_trade(
        "BTCUSD.x", 0.1, "buy", 50000, 49000, 52000
    )
    assert not valid, "Should reject trade at guardian limit"
    print(f"✓ Guardian limit enforced at 7% DD")
    
    # Test 4: Symbol filtering
    assert rm._is_instrument_allowed("BTCUSD.x")
    assert rm._is_instrument_allowed("ETHUSD")
    print("✓ Symbol filtering works")
    
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
    parser = argparse.ArgumentParser(description="GFT Crypto Trading Bot")
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
        bot = GFTBot()
        bot.start()


if __name__ == "__main__":
    main()
