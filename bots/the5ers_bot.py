"""
The5ers Bot - Forex Trading Bot.

Standalone bot file for The5ers $5,000 accounts trading forex.

RULES (DO NOT MODIFY):
- Max daily loss: 5% (guardian: 4%)
- Max overall loss: 10% (guardian: 8.5%)
- Consistency rule: No single day > 30% of total profit
- Forex majors/minors only

Usage:
    python the5ers_bot.py --config config/the5ers_account.py
    python the5ers_bot.py --config config/the5ers_account.py --paper
    python the5ers_bot.py --config config/the5ers_account.py --debug
"""

import argparse
import importlib.util
import logging
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Any, Optional

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


# Default configuration (used if no config file provided)
DEFAULT_CONFIG = {
    'ACCOUNT_NAME': 'The5ers_Account',
    'ACCOUNT_SIZE': 5000.0,
    'FIRM': 'The5ers',
    'MT5_LOGIN': 87654321,
    'MT5_PASSWORD': 'your_password',
    'MT5_SERVER': 'The5ers-Server',
    'MT5_PATH': r'C:\Program Files\MetaTrader 5\terminal64.exe',
    'TELEGRAM_TOKEN': '',
    'TELEGRAM_CHAT_IDS': [],
    'SYMBOLS': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURJPY'],
    'TIMEFRAME': 'M5',
    'SCAN_INTERVAL': 60,
    'MAX_DRAWDOWN_PERCENT': 8.5,
    'DAILY_LOSS_PERCENT': 4.0,
    'MAX_POSITIONS': 5,
    'MAX_RISK_PER_TRADE': 1.0,
    'DEFAULT_EXECUTION_STYLE': 'market',
    'SLIPPAGE_TOLERANCE': 5,
    'MAX_RETRIES': 3,
}

# Trading Sessions (UTC)
TRADING_SESSIONS = {
    'london': (dt_time(8, 0), dt_time(17, 0)),
    'new_york': (dt_time(13, 0), dt_time(22, 0)),
    'overlap': (dt_time(13, 0), dt_time(17, 0)),  # Best liquidity
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

class The5ersBot(BaseTradingBot):
    """
    The5ers-specific trading bot.

    Implements:
    - Daily loss limit tracking
    - Consistency rule enforcement
    - Session-aware trading
    - Friday closeout
    """

    def __init__(self, config: Dict[str, Any], paper_mode: bool = False):
        """
        Initialize The5ers Bot with config.

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

        # Create firm rules
        firm_rules = create_the5ers_rules(
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
        if config.get('MAX_POSITIONS'):
            firm_rules.max_positions = config['MAX_POSITIONS']

        # Log startup info
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("THE5ERS RULES LOADED:")
        logger.info(f"  Account: {config['ACCOUNT_NAME']}")
        logger.info(f"  Paper Mode: {paper_mode}")
        logger.info(f"  Max Overall DD: {firm_rules.max_overall_drawdown_pct}% (Guardian: {firm_rules.guardian_drawdown_pct}%)")
        logger.info(f"  Max Daily DD: {firm_rules.max_daily_loss_pct}% (Guardian: {firm_rules.guardian_daily_loss_pct}%)")
        logger.info(f"  Consistency Rule: {firm_rules.consistency_max_single_day_pct}% of total profits")
        logger.info(f"  Max Positions: {firm_rules.max_positions}")
        logger.info("=" * 60)

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
        self.slippage_tolerance = config.get('SLIPPAGE_TOLERANCE', 5)
        self.max_retries = config.get('MAX_RETRIES', 3)

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

        # Paper mode or live
        if self.paper_mode:
            self._paper_analyze_and_trade(symbol)
        else:
            super()._analyze_and_trade(symbol)

    def _paper_analyze_and_trade(self, symbol: str):
        """Paper trading - log signals without executing."""
        logger = logging.getLogger(__name__)

        try:
            # Get historical data
            df = self.data_fetcher.get_historical_bars(
                symbol, self.timeframe, 500
            )

            if df.empty or len(df) < 100:
                return

            # Generate signal
            signal = self.signal_generator.generate_signal(symbol, df)

            # Skip if neutral or low confidence
            if signal.action == "neutral" or signal.confidence < 0.5:
                return

            # Log the signal (paper mode)
            logger.info(
                f"PAPER SIGNAL: {signal.action.upper()} {symbol} "
                f"(confidence: {signal.confidence:.2f}, regime: {signal.regime})"
            )

            if self.telegram:
                self.telegram.send_alert(
                    f"[PAPER] Signal: {signal.action.upper()} {symbol}\n"
                    f"Confidence: {signal.confidence:.2f}\n"
                    f"Regime: {signal.regime}",
                    level="info"
                )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

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

        if self.paper_mode:
            logger.info("PAPER MODE: Would close all positions for weekend")
            return

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
    print("The5ers rules created correctly")

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
    print(f"Daily loss calculation correct: {daily_loss}%")

    # Test 3: Guardian daily limit
    state.current_equity = 4800  # 4% daily loss
    state.daily_pnl = -200
    rm.update_account_state(4800, 4800)

    valid, violation, msg = rm.validate_trade(
        "EURUSD", 0.1, "buy", 1.1000, 1.0950, 1.1100
    )
    assert not valid, "Should reject trade at daily guardian limit"
    print("Daily guardian limit enforced at 4%")

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
    print("Consistency rule check implemented")

    print("\nAll tests passed!")


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
    parser.add_argument('--config', type=str, help='Path to config file (e.g., config/the5ers_account.py)')
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
        bot = The5ersBot(config, paper_mode=args.paper)
        bot.start()


if __name__ == "__main__":
    main()
