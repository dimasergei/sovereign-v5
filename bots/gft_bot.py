"""
GFT Bot - Goat Funded Trader Crypto Bot.

Standalone bot file for GFT Instant Funding GOAT Model accounts trading crypto CFDs.

CRITICAL RULES (DO NOT MODIFY) - As of 2024:
- Max overall drawdown: 6% (trailing from EQUITY high water mark)
- Guardian limit: 5% (stop trading before actual limit)
- Max daily drawdown: 3% (resets 5 PM EST)
- Guardian daily: 2.5%
- Max floating loss per trade: 2% (HARD BREACH - IMMEDIATE ACCOUNT CLOSURE!)
- Guardian per-trade: 1.5%
- Consistency rule: 15% (no single day > 15% of total profits) - blocks payout only
- Min trading days: 5 @ 0.5% profit each for payout
- No hedging, martingale, or HFT
- Max inactivity: 30 days
- Crypto CFDs only

WARNING: The 2% per-trade floating loss limit is the most dangerous rule.
It causes IMMEDIATE account closure if breached. We monitor this every second.
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

    Implements GFT Instant Funding GOAT Model specific rules and overrides.

    CRITICAL LIMITS:
    - 6% max total DD (trailing from equity HWM)
    - 3% max daily DD (resets 5 PM EST)
    - 2% max floating loss per trade (HARD BREACH!)
    """

    def __init__(self):
        # Create credentials
        credentials = MT5Credentials(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
            path=MT5_PATH,
        )

        # Create firm rules with CORRECT GFT limits
        firm_rules = create_gft_rules(
            initial_balance=INITIAL_BALANCE,
            symbols=SYMBOLS
        )

        # Log critical limits on startup
        logger = logging.getLogger(__name__)
        logger.critical("=" * 60)
        logger.critical("GFT INSTANT FUNDING GOAT MODEL RULES LOADED:")
        logger.critical(f"  Max Total DD: {firm_rules.max_overall_drawdown_pct}% (Guardian: {firm_rules.guardian_drawdown_pct}%)")
        logger.critical(f"  Drawdown Reference: {firm_rules.drawdown_reference.upper()}")
        logger.critical(f"  Max Daily DD: {firm_rules.max_daily_loss_pct}% (Guardian: {firm_rules.guardian_daily_loss_pct}%)")
        logger.critical(f"  Max Trade Loss: {firm_rules.max_trade_floating_loss_pct}% (Guardian: {firm_rules.guardian_trade_floating_loss_pct}%)")
        logger.critical(f"  Daily Reset: {firm_rules.daily_reset_time} {firm_rules.daily_reset_timezone}")
        logger.critical(f"  Consistency Rule: {firm_rules.consistency_max_single_day_pct}% of total profits")
        logger.critical(f"  Min Trading Days: {firm_rules.min_trading_days_for_payout} @ {firm_rules.min_profit_per_trading_day_pct}%")
        logger.critical("=" * 60)

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
        Guardian warning at 25 days, auto-ping at 28 days.
        """
        super()._check_inactivity()

        days_inactive = self.risk_manager.state.days_since_last_trade
        logger = logging.getLogger(__name__)

        if days_inactive >= 28:
            # Critical - place a minimal trade immediately
            logger.critical(f"Inactivity critical: {days_inactive} days - placing ping trade")
            self._place_ping_trade()
        elif days_inactive >= 25:
            # Warning - approaching limit
            logger.warning(f"Inactivity warning: {days_inactive} days since last trade")
            if self.telegram:
                self.telegram.send_alert(
                    f"⚠️ Inactivity Warning\n"
                    f"Days since last trade: {days_inactive}\n"
                    f"Auto-ping at 28 days\n"
                    f"Max limit: 30 days",
                    level="warning"
                )
        elif days_inactive >= 20:
            # Info - reminder
            logger.info(f"Inactivity reminder: {days_inactive} days since last trade")
    
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

    # Test 1: GFT rules creation - CORRECTED LIMITS
    rules = create_gft_rules(10000)
    assert rules.max_overall_drawdown_pct == 6.0, f"Expected 6.0%, got {rules.max_overall_drawdown_pct}%"
    assert rules.guardian_drawdown_pct == 5.0, f"Expected 5.0%, got {rules.guardian_drawdown_pct}%"
    assert rules.max_daily_loss_pct == 3.0, f"Expected 3.0%, got {rules.max_daily_loss_pct}%"
    assert rules.max_trade_floating_loss_pct == 2.0, f"Expected 2.0%, got {rules.max_trade_floating_loss_pct}%"
    assert rules.drawdown_reference == "equity", f"Expected 'equity', got {rules.drawdown_reference}"
    assert rules.daily_reset_timezone == "US/Eastern", f"Expected 'US/Eastern', got {rules.daily_reset_timezone}"
    print("✓ GFT rules created correctly with CORRECTED limits")

    # Test 2: Drawdown calculation from EQUITY HWM
    state = AccountRiskState(
        initial_balance=10000,
        highest_balance=10000,
        highest_equity=10000,
        current_balance=9500,
        current_equity=9500,
        daily_starting_balance=10000,
        daily_starting_equity=10000,
        daily_pnl=-500,
        daily_date=date.today().isoformat()
    )

    rm = RiskManager(rules, state, "/tmp/test_state.json")
    dd = rm.get_current_drawdown_pct()
    assert dd == 5.0, f"Expected 5.0%, got {dd}%"
    print(f"✓ Drawdown calculation correct (from equity): {dd}%")

    # Test 3: Guardian limit - should reject at 5% DD (not 7%)
    state.current_equity = 9500  # 5% DD
    state.highest_equity = 10000
    rm.update_account_state(9500, 9500)

    valid, violation, msg = rm.validate_trade(
        "BTCUSD.x", 0.1, "buy", 50000, 49000, 52000
    )
    assert not valid, "Should reject trade at 5% guardian limit"
    print(f"✓ Guardian limit enforced at 5% DD (corrected from 7%)")

    # Test 4: Symbol filtering
    assert rm._is_instrument_allowed("BTCUSD.x")
    assert rm._is_instrument_allowed("ETHUSD")
    print("✓ Symbol filtering works")

    # Test 5: Verify critical per-trade limit exists
    assert rules.guardian_trade_floating_loss_pct == 1.5
    print("✓ Per-trade floating loss guardian set to 1.5%")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("CRITICAL: GFT limits are now CORRECT:")
    print(f"  - 6% max DD (was 8%)")
    print(f"  - 3% daily DD (was None)")
    print(f"  - 2% per-trade (was None)")
    print("=" * 60)


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
