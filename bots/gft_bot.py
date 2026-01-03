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
- CFDs (Gold, Silver, Indices, Forex)

WARNING: The 2% per-trade floating loss limit is the most dangerous rule.
It causes IMMEDIATE account closure if breached. We monitor this every second.

Usage:
    python gft_bot.py --config config/gft_account_1.py
    python gft_bot.py --config config/gft_account_1.py --paper
    python gft_bot.py --config config/gft_account_1.py --debug
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    MT5Credentials,
    FirmRules,
    FirmType,
    create_gft_rules,
)
from core.position_sizer import PositionSizer
from monitoring import TelegramConfig, AlertLevel
from bots.base_bot import BaseTradingBot
from signals.trend_filter import TrendFilter, TrendDirection


# Default configuration (used if no config file provided)
DEFAULT_CONFIG = {
    'ACCOUNT_NAME': 'GFT_Account1',
    'ACCOUNT_SIZE': 10000.0,
    'FIRM': 'GFT',
    'MT5_LOGIN': 12345678,
    'MT5_PASSWORD': 'your_password',
    'MT5_SERVER': 'GoatFundedTrader-Server',
    'MT5_PATH': r'C:\Program Files\MetaTrader 5\terminal64.exe',
    'TELEGRAM_TOKEN': '',
    'TELEGRAM_CHAT_IDS': [],
    'SYMBOLS': ['XAUUSD.x', 'XAGUSD.x', 'NAS100.x', 'UK100.x', 'SPX500.x', 'EURUSD.x'],
    'TIMEFRAME': 'M5',
    'SCAN_INTERVAL': 60,
    'MAX_DRAWDOWN_PERCENT': 5.0,
    'DAILY_LOSS_PERCENT': 2.5,
    'MAX_POSITIONS': 3,
    'MAX_RISK_PER_TRADE': 1.5,
    'DEFAULT_EXECUTION_STYLE': 'market',
    'SLIPPAGE_TOLERANCE': 10,
    'MAX_RETRIES': 3,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a Python file.

    Args:
        config_path: Path to the config file

    Returns:
        Dictionary with configuration values
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the config module dynamically
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Extract configuration values
    config = {}
    for key in DEFAULT_CONFIG.keys():
        if hasattr(config_module, key):
            config[key] = getattr(config_module, key)
        else:
            config[key] = DEFAULT_CONFIG[key]

    return config


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

    def __init__(self, config: Dict[str, Any], paper_mode: bool = False):
        """
        Initialize GFT Bot with config.

        Args:
            config: Configuration dictionary
            paper_mode: If True, simulate trades without executing
        """
        self.config = config
        self.paper_mode = paper_mode

        # Create credentials
        credentials = MT5Credentials(
            login=config['MT5_LOGIN'],
            password=config['MT5_PASSWORD'],
            server=config['MT5_SERVER'],
            path=config['MT5_PATH'],
        )

        # Create firm rules with CORRECT GFT limits
        firm_rules = create_gft_rules(
            initial_balance=config['ACCOUNT_SIZE'],
            symbols=config['SYMBOLS']
        )

        # Override guardian limits from config if more conservative
        if config.get('MAX_DRAWDOWN_PERCENT'):
            firm_rules.guardian_drawdown_pct = min(
                firm_rules.guardian_drawdown_pct,
                config['MAX_DRAWDOWN_PERCENT']
            )
        if config.get('DAILY_LOSS_PERCENT'):
            firm_rules.guardian_daily_loss_pct = min(
                firm_rules.guardian_daily_loss_pct or 999,
                config['DAILY_LOSS_PERCENT']
            )
        if config.get('MAX_RISK_PER_TRADE'):
            firm_rules.guardian_trade_floating_loss_pct = min(
                firm_rules.guardian_trade_floating_loss_pct or 999,
                config['MAX_RISK_PER_TRADE']
            )
        if config.get('MAX_POSITIONS'):
            firm_rules.max_positions = config['MAX_POSITIONS']

        # Log critical limits on startup
        logger = logging.getLogger(__name__)
        logger.critical("=" * 60)
        logger.critical("GFT INSTANT FUNDING GOAT MODEL RULES LOADED:")
        logger.critical(f"  Account: {config['ACCOUNT_NAME']}")
        logger.critical(f"  Paper Mode: {paper_mode}")
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
        if config.get('TELEGRAM_TOKEN') and config['TELEGRAM_TOKEN'] != 'your_bot_token':
            telegram_config = TelegramConfig(
                bot_token=config['TELEGRAM_TOKEN'],
                authorized_chat_ids=config.get('TELEGRAM_CHAT_IDS', []),
            )

        # State file
        state_file = f"storage/state/{config['ACCOUNT_NAME']}_state.json"

        super().__init__(
            account_name=config['ACCOUNT_NAME'],
            credentials=credentials,
            firm_rules=firm_rules,
            symbols=config['SYMBOLS'],
            state_file=state_file,
            telegram_config=telegram_config,
            scan_interval_seconds=config.get('SCAN_INTERVAL', 60),
            timeframe=config.get('TIMEFRAME', 'M5'),
        )

        # Store execution settings
        self.execution_style = config.get('DEFAULT_EXECUTION_STYLE', 'market')
        self.slippage_tolerance = config.get('SLIPPAGE_TOLERANCE', 10)
        self.max_retries = config.get('MAX_RETRIES', 3)

        # CRITICAL: Add trend filter and position sizer for drawdown prevention
        self.trend_filter = TrendFilter()
        self.position_sizer = PositionSizer(
            max_risk_pct=1.0,  # Max 1% per trade (GFT allows 2%, we use 1% for safety)
            min_risk_pct=0.25,
            kelly_fraction=0.25  # Quarter-Kelly for safety
        )

        # Track blocked trades for monitoring
        self.blocked_trades = {"counter_trend": 0, "guardian": 0, "risk_limit": 0}

    def _init_connection(self):
        """Initialize MT5 connection (skip in paper mode)."""
        if self.paper_mode:
            logger = logging.getLogger(__name__)
            logger.info("PAPER MODE: Skipping MT5 connection")
            logger.info(f"PAPER MODE: Simulating account with ${self.config['ACCOUNT_SIZE']:.2f}")

            # Initialize paper trading state
            self._paper_balance = self.config['ACCOUNT_SIZE']
            self._paper_equity = self.config['ACCOUNT_SIZE']
            self._paper_positions = []

            # Initialize paper data fetcher for market data
            from data.paper_fetcher import PaperDataFetcher
            self.data_fetcher = PaperDataFetcher()
            logger.info("PAPER MODE: Using PaperDataFetcher for market data")
            return

        # Normal connection
        super()._init_connection()

    def _update_account_state(self) -> bool:
        """Update account state (simulated in paper mode)."""
        if self.paper_mode:
            # Simulate account state
            risk_status = self.risk_manager.update_account_state(
                balance=self._paper_balance,
                equity=self._paper_equity
            )
            if risk_status.get('violations'):
                logging.getLogger(__name__).warning(f"Risk violations: {risk_status['violations']}")
            return True

        return super()._update_account_state()

    def _analyze_and_trade(self, symbol: str):
        """Analyze and trade (simulated execution in paper mode)."""
        if self.paper_mode:
            self._paper_analyze_and_trade(symbol)
        else:
            super()._analyze_and_trade(symbol)

    def _paper_analyze_and_trade(self, symbol: str):
        """
        Paper trading with full trend filter and risk checks.

        This simulates the complete trading flow including:
        1. Trend filter (blocks counter-trend trades)
        2. Position sizing based on risk budget
        3. Pre-trade risk checks
        """
        logger = logging.getLogger(__name__)

        try:
            # Get historical data
            df = self.data_fetcher.get_historical_bars(
                symbol, self.timeframe, 500
            )

            if df.empty or len(df) < 100:
                return

            # Generate signal (now includes trend filtering via SignalGenerator)
            signal = self.signal_generator.generate_signal(symbol, df)

            # Log trend filter action
            if signal.filter_reason == "counter_trend_blocked":
                self.blocked_trades["counter_trend"] += 1
                logger.info(
                    f"[{symbol}] BLOCKED {signal.trend_direction} - counter-trend trade prevented"
                )
                return

            # Skip if neutral or low confidence
            if signal.action == "neutral" or signal.confidence < 0.4:
                return

            # Calculate position size using position sizer
            current_dd = 0.0  # Paper mode starts at 0% DD
            if hasattr(self, '_paper_equity') and hasattr(self, 'config'):
                if self._paper_equity < self.config['ACCOUNT_SIZE']:
                    current_dd = (1 - self._paper_equity / self.config['ACCOUNT_SIZE']) * 100

            position_info = self.position_sizer.calculate(
                account_balance=self._paper_balance,
                current_drawdown_pct=current_dd,
                max_drawdown_pct=self.firm_rules.max_overall_drawdown_pct,
                stop_loss_pct=signal.stop_loss_atr_mult * 0.5,  # Estimate SL %
                signal_confidence=signal.confidence,
                regime=signal.regime,
                trend_strength=signal.trend_strength
            )

            # Skip if position size is zero (guardian proximity)
            if position_info.size <= 0:
                self.blocked_trades["guardian"] += 1
                logger.warning(f"[{symbol}] BLOCKED - position size zero ({position_info.reason})")
                return

            # Log the signal with trend info (paper mode)
            logger.info(
                f"PAPER SIGNAL: {signal.action.upper()} {symbol} "
                f"(conf: {signal.confidence:.2f}, regime: {signal.regime}, "
                f"trend: {signal.trend_direction}, strength: {signal.trend_strength:.2f}, "
                f"aligned: {signal.higher_tf_aligned}, risk: {position_info.risk_pct:.2f}%)"
            )

            if self.telegram:
                self.telegram.send_alert(
                    f"[PAPER] Signal: {signal.action.upper()} {symbol}\n"
                    f"Confidence: {signal.confidence:.2f}\n"
                    f"Regime: {signal.regime}\n"
                    f"Trend: {signal.trend_direction} (strength: {signal.trend_strength:.2f})\n"
                    f"Risk: {position_info.risk_pct:.2f}%",
                    level=AlertLevel.INFO
                )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

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
            if not self.paper_mode:
                self._place_ping_trade()
            else:
                logger.info("PAPER MODE: Would place ping trade")
        elif days_inactive >= 25:
            # Warning - approaching limit
            logger.warning(f"Inactivity warning: {days_inactive} days since last trade")
            if self.telegram:
                self.telegram.send_alert(
                    f"Inactivity Warning\n"
                    f"Days since last trade: {days_inactive}\n"
                    f"Auto-ping at 28 days\n"
                    f"Max limit: 30 days",
                    level=AlertLevel.WARNING
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
                    f"Ping trade placed to avoid inactivity\n"
                    f"Symbol: {symbol}\n"
                    f"Size: {min_lot}",
                    level=AlertLevel.INFO
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
    print("GFT rules created correctly with CORRECTED limits")

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
    print(f"Drawdown calculation correct (from equity): {dd}%")

    # Test 3: Guardian limit - should reject at 5% DD (not 7%)
    state.current_equity = 9500  # 5% DD
    state.highest_equity = 10000
    rm.update_account_state(9500, 9500)

    valid, violation, msg = rm.validate_trade(
        "BTCUSD.x", 0.1, "buy", 50000, 49000, 52000
    )
    assert not valid, "Should reject trade at 5% guardian limit"
    print("Guardian limit enforced at 5% DD (corrected from 7%)")

    # Test 4: Symbol filtering
    assert rm._is_instrument_allowed("BTCUSD.x")
    assert rm._is_instrument_allowed("ETHUSD")
    print("Symbol filtering works")

    # Test 5: Verify critical per-trade limit exists
    assert rules.guardian_trade_floating_loss_pct == 1.5
    print("Per-trade floating loss guardian set to 1.5%")

    print("\n" + "=" * 60)
    print("All tests passed!")
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
    parser.add_argument('--config', type=str, help='Path to config file (e.g., config/gft_account_1.py)')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode (simulate without executing)')
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
        # Load configuration
        if args.config:
            config = load_config(args.config)
            logging.getLogger(__name__).info(f"Loaded config from {args.config}")
        else:
            config = DEFAULT_CONFIG.copy()
            logging.getLogger(__name__).warning("No config file specified, using defaults")

        # Run the bot
        bot = GFTBot(config, paper_mode=args.paper)
        bot.start()


if __name__ == "__main__":
    main()
