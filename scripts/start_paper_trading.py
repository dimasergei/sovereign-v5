#!/usr/bin/env python3
"""
Sovereign V5 Paper Trading Runner.

Runs paper trading simulation across 4 prop firm accounts:
- GFT_1, GFT_2, GFT_3: $10,000 each (GFT Instant GOAT rules)
- THE5ERS_1: $5,000 (The5ers High Stakes rules)

Elite Portfolio: XAUUSD, XAGUSD, NAS100, UK100, SPX500, EURUSD

Features:
- Real-time signal generation and paper execution
- Telegram notifications for trades and status
- Compliance checking (GFT/The5ers rules)
- Daily reports saved to logs/

Usage:
    python scripts/start_paper_trading.py

Press Ctrl+C to stop and get final report.
"""

import json
import logging
import os
import signal
import sys
import time
import requests
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paper_executor import PaperExecutor, AccountState
from data.paper_fetcher import PaperDataFetcher
from signals.generator import SignalGenerator, TradingSignal
from core.news_calendar import get_news_calendar
from core.position_sizer import PositionSizer

# MT5 imports (optional - for real data)
try:
    from core.mt5_connector import MT5Connector, MT5Credentials
    from data.mt5_fetcher import MT5DataFetcher
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8044940173:AAE6eEz3NjXxWaHkZq3nP903m2LHZAhuYjM"
TELEGRAM_CHAT_IDS = [7898079111]

# MT5 Configuration - Use real data from MT5 demo account
USE_MT5_DATA = True  # Set to False to use synthetic data
MT5_LOGIN = 314329147
MT5_PASSWORD = "e#sBIdI0sV"
MT5_SERVER = "GoatFunded-Server"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Account configurations
ACCOUNTS = [
    {
        "name": "GFT_1",
        "initial_balance": 10000.0,
        "account_type": "GFT",
    },
    {
        "name": "GFT_2",
        "initial_balance": 10000.0,
        "account_type": "GFT",
    },
    {
        "name": "GFT_3",
        "initial_balance": 10000.0,
        "account_type": "GFT",
    },
    {
        "name": "THE5ERS_1",
        "initial_balance": 5000.0,
        "account_type": "THE5ERS",
    },
]

# Elite Portfolio - Top 6 by Sharpe ratio (use .x suffix for GFT broker)
SYMBOLS = ["XAUUSD.x", "XAGUSD.x", "NAS100.x", "UK100.x", "SPX500.x", "EURUSD.x"]

# Trading parameters
TIMEFRAME = "M15"
BARS_TO_FETCH = 100
SCAN_INTERVAL_SECONDS = 30  # Scan every 30 seconds
STATUS_LOG_INTERVAL_SECONDS = 300  # Log status every 5 minutes
TELEGRAM_STATUS_INTERVAL_SECONDS = 1800  # Telegram status every 30 minutes
DAILY_REPORT_HOUR = 17  # 5 PM for daily report

# Risk Management - ADJUSTED FOR STABILITY
STOP_LOSS_ATR_MULT = 2.5      # Wider stops to avoid whipsaws (was 1.5)
TAKE_PROFIT_ATR_MULT = 5.0    # 2:1 reward-to-risk ratio
MIN_SIGNAL_CONFIDENCE = 0.60  # Higher confidence threshold (was 0.40)
SIGNAL_COOLDOWN_SECONDS = 300 # 5 min cooldown per symbol after signal

# Paths
LOGS_DIR = "logs"
STATE_DIR = "storage/state"


# ============================================================================
# TELEGRAM NOTIFIER WITH COMMAND HANDLING
# ============================================================================

class TelegramNotifier:
    """Telegram notification sender with command handling."""

    def __init__(self, bot_token: str, chat_ids: List[int]):
        self.bot_token = bot_token
        self.chat_ids = chat_ids
        self.enabled = bool(bot_token and chat_ids)
        self.logger = logging.getLogger('Telegram')
        self.last_update_id = 0
        self.runner = None  # Will be set by PaperTradingRunner

        if not self.enabled:
            self.logger.warning("Telegram notifications disabled (no token/chat_ids)")

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to all configured chat IDs."""
        if not self.enabled:
            return False

        success = True
        for chat_id in self.chat_ids:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                }
                response = requests.post(url, json=payload, timeout=10)

                if response.status_code != 200:
                    self.logger.warning(f"Telegram send failed: {response.text}")
                    success = False

            except Exception as e:
                self.logger.warning(f"Telegram error: {e}")
                success = False

        return success

    def send_startup(self, accounts: List[Dict], symbols: List[str]):
        """Send startup notification."""
        total_balance = sum(a['initial_balance'] for a in accounts)

        msg = (
            "🚀 <b>SOVEREIGN V5 - PAPER TRADING STARTED</b>\n\n"
            f"📊 <b>Accounts:</b>\n"
        )
        for acc in accounts:
            msg += f"  • {acc['name']}: ${acc['initial_balance']:,.0f} ({acc['account_type']})\n"

        msg += f"\n💰 <b>Total Capital:</b> ${total_balance:,.0f}\n"
        msg += f"📈 <b>Symbols:</b> {', '.join(symbols)}\n"
        msg += f"⏰ <b>Scan Interval:</b> {SCAN_INTERVAL_SECONDS}s\n"
        msg += f"\n✅ Ready to trade!"

        self.send(msg)

    def send_shutdown(self, summary: Dict):
        """Send shutdown notification with summary."""
        msg = (
            "🛑 <b>SOVEREIGN V5 - PAPER TRADING STOPPED</b>\n\n"
            f"⏱ <b>Runtime:</b> {summary.get('runtime', 'N/A')}\n"
            f"📊 <b>Signals:</b> {summary.get('total_signals', 0)}\n"
            f"📈 <b>Trades:</b> {summary.get('total_trades', 0)}\n\n"
        )

        for name, data in summary.get('accounts', {}).items():
            pnl = data.get('total_pnl', 0)
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            msg += f"{pnl_emoji} <b>{name}:</b> ${pnl:+,.2f}\n"

        total_pnl = summary.get('total_pnl', 0)
        total_emoji = "🟢" if total_pnl >= 0 else "🔴"
        msg += f"\n{total_emoji} <b>Total P&L:</b> ${total_pnl:+,.2f}"

        self.send(msg)

    def send_trade_opened(self, account: str, symbol: str, direction: str,
                          size: float, price: float, sl: float, tp: float,
                          risk_amount: float = 0, risk_pct: float = 0):
        """Send trade opened notification."""
        emoji = "📈" if direction.upper() in ("LONG", "BUY") else "📉"
        msg = (
            f"{emoji} <b>TRADE OPENED</b>\n\n"
            f"📊 <b>Account:</b> {account}\n"
            f"💱 <b>Symbol:</b> {symbol}\n"
            f"↗️ <b>Direction:</b> {direction.upper()}\n"
            f"📏 <b>Risk:</b> ${risk_amount:.2f} ({risk_pct:.2f}%)\n"
            f"💵 <b>Entry:</b> {price:.2f}\n"
            f"🛑 <b>SL:</b> {sl:.2f}\n"
            f"🎯 <b>TP:</b> {tp:.2f}"
        )
        self.send(msg)

    def send_trade_closed(self, account: str, symbol: str, pnl: float, reason: str):
        """Send trade closed notification."""
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} <b>TRADE CLOSED</b>\n\n"
            f"📊 <b>Account:</b> {account}\n"
            f"💱 <b>Symbol:</b> {symbol}\n"
            f"💰 <b>P&L:</b> ${pnl:+,.2f}\n"
            f"📝 <b>Reason:</b> {reason}"
        )
        self.send(msg)

    def send_status(self, accounts: Dict[str, AccountState], runtime: str):
        """Send periodic status update."""
        total_equity = sum(s.equity for s in accounts.values())
        total_pnl = sum(s.realized_pnl + s.unrealized_pnl for s in accounts.values())
        total_positions = sum(s.open_positions for s in accounts.values())

        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"

        msg = (
            f"📊 <b>STATUS UPDATE</b>\n\n"
            f"⏱ <b>Runtime:</b> {runtime}\n"
            f"💰 <b>Total Equity:</b> ${total_equity:,.2f}\n"
            f"{pnl_emoji} <b>Total P&L:</b> ${total_pnl:+,.2f}\n"
            f"📈 <b>Open Positions:</b> {total_positions}\n\n"
        )

        for name, state in accounts.items():
            dd_warning = "⚠️" if state.current_dd_pct > 3 else ""
            msg += (
                f"<b>{name}:</b> ${state.equity:,.0f} | "
                f"DD: {state.current_dd_pct:.1f}% {dd_warning}\n"
            )

        self.send(msg)

    def send_compliance_warning(self, account: str, warning: str, dd_pct: float):
        """Send compliance warning."""
        msg = (
            f"⚠️ <b>COMPLIANCE WARNING</b>\n\n"
            f"📊 <b>Account:</b> {account}\n"
            f"⚠️ <b>Warning:</b> {warning}\n"
            f"📉 <b>Drawdown:</b> {dd_pct:.2f}%\n\n"
            f"🛑 Trading may be paused soon!"
        )
        self.send(msg)

    def check_commands(self):
        """Check for incoming Telegram commands and process them."""
        if not self.enabled or not self.runner:
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {"offset": self.last_update_id + 1, "timeout": 1}
            response = requests.get(url, params=params, timeout=5)

            if response.status_code != 200:
                return

            data = response.json()
            if not data.get("ok"):
                return

            for update in data.get("result", []):
                self.last_update_id = update["update_id"]
                message = update.get("message", {})
                chat_id = message.get("chat", {}).get("id")
                text = message.get("text", "").strip()

                # Only respond to authorized chat IDs
                if chat_id not in self.chat_ids:
                    continue

                # Process commands
                if text.startswith("/"):
                    self._handle_command(text, chat_id)

        except Exception as e:
            pass  # Silently ignore polling errors

    def _handle_command(self, text: str, chat_id: int):
        """Handle a Telegram command."""
        command = text.split()[0].lower()

        if command == "/status":
            self._cmd_status(chat_id)
        elif command == "/positions":
            self._cmd_positions(chat_id)
        elif command == "/trades":
            self._cmd_trades(chat_id)
        elif command == "/help":
            self._cmd_help(chat_id)
        elif command == "/ping":
            self._send_to(chat_id, "🏓 <b>Pong!</b> Bot is running.")

    def _send_to(self, chat_id: int, message: str):
        """Send message to specific chat ID."""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass

    def _cmd_help(self, chat_id: int):
        """Send help message."""
        msg = (
            "📖 <b>SOVEREIGN V5 COMMANDS</b>\n\n"
            "/status - Account summary & P&L\n"
            "/positions - Open positions\n"
            "/trades - Recent trade history\n"
            "/ping - Check if bot is running\n"
            "/help - Show this help"
        )
        self._send_to(chat_id, msg)

    def _cmd_status(self, chat_id: int):
        """Send status summary."""
        if not self.runner:
            self._send_to(chat_id, "❌ Runner not initialized")
            return

        runtime = str(datetime.now() - self.runner.start_time).split('.')[0]
        total_equity = 0
        total_pnl = 0
        total_positions = 0

        msg = f"📊 <b>ACCOUNT STATUS</b>\n\n"
        msg += f"⏱ <b>Runtime:</b> {runtime}\n\n"

        for name, executor in self.runner.executors.items():
            state = executor.get_account_state()
            pnl = state.realized_pnl + state.unrealized_pnl
            total_equity += state.equity
            total_pnl += pnl
            total_positions += state.open_positions

            emoji = "🟢" if pnl >= 0 else "🔴"
            dd_warn = " ⚠️" if state.current_dd_pct > 3 else ""

            msg += (
                f"{emoji} <b>{name}</b>\n"
                f"   Equity: ${state.equity:,.0f} | P&L: ${pnl:+,.0f}\n"
                f"   DD: {state.current_dd_pct:.1f}%{dd_warn} | Pos: {state.open_positions}\n\n"
            )

        total_emoji = "🟢" if total_pnl >= 0 else "🔴"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"{total_emoji} <b>TOTAL:</b> ${total_equity:,.0f}\n"
        msg += f"💰 <b>P&L:</b> ${total_pnl:+,.2f}\n"
        msg += f"📈 <b>Positions:</b> {total_positions}\n"
        msg += f"📊 <b>Signals:</b> {self.runner.total_signals} | <b>Trades:</b> {self.runner.total_trades}"

        self._send_to(chat_id, msg)

    def _cmd_positions(self, chat_id: int):
        """Send open positions."""
        if not self.runner:
            self._send_to(chat_id, "❌ Runner not initialized")
            return

        msg = "📈 <b>OPEN POSITIONS</b>\n\n"
        total_positions = 0

        for name, executor in self.runner.executors.items():
            positions = executor.get_open_positions()
            if not positions:
                continue

            msg += f"<b>{name}</b>\n"
            for pos in positions:
                total_positions += 1
                emoji = "📈" if pos.direction.upper() == "LONG" else "📉"
                msg += (
                    f"  {emoji} {pos.symbol}\n"
                    f"     {pos.direction.upper()} x{pos.size:.2f} @ {pos.entry_price:.2f}\n"
                    f"     P&L: ${pos.unrealized_pnl:+,.2f}\n"
                )
            msg += "\n"

        if total_positions == 0:
            msg += "No open positions."

        self._send_to(chat_id, msg)

    def _cmd_trades(self, chat_id: int):
        """Send recent trades."""
        if not self.runner:
            self._send_to(chat_id, "❌ Runner not initialized")
            return

        msg = "📜 <b>RECENT TRADES</b>\n\n"
        all_trades = []

        for name, executor in self.runner.executors.items():
            history = executor.get_trade_history()
            for trade in history[-3:]:  # Last 3 per account
                all_trades.append((name, trade))

        if not all_trades:
            msg += "No closed trades yet."
        else:
            # Show last 10 overall
            for name, trade in all_trades[-10:]:
                emoji = "✅" if trade.realized_pnl >= 0 else "❌"
                msg += (
                    f"{emoji} <b>{name}</b> | {trade.symbol}\n"
                    f"   {trade.direction.upper()} | P&L: ${trade.realized_pnl:+,.2f}\n\n"
                )

        self._send_to(chat_id, msg)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for paper trading."""
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, f"paper_trading_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return logging.getLogger('PaperTrading')


# ============================================================================
# PAPER TRADING ENGINE
# ============================================================================

class PaperTradingRunner:
    """
    Runs paper trading across multiple accounts.
    """

    def __init__(self):
        """Initialize paper trading runner."""
        self.logger = setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("  SOVEREIGN V5 - PAPER TRADING MODE")
        self.logger.info("=" * 60)

        # Initialize MT5 connection if available and enabled
        self.mt5_connector = None
        if USE_MT5_DATA and MT5_AVAILABLE:
            self.logger.info("[MT5] Connecting to MetaTrader 5 for real market data...")
            try:
                credentials = MT5Credentials(
                    login=MT5_LOGIN,
                    password=MT5_PASSWORD,
                    server=MT5_SERVER,
                    path=MT5_PATH
                )
                self.mt5_connector = MT5Connector(credentials)
                if self.mt5_connector.connect():
                    self.logger.info("[MT5] Connected successfully - using REAL market data")
                    self.data_fetcher = MT5DataFetcher(self.mt5_connector)
                else:
                    self.logger.warning("[MT5] Connection failed - falling back to synthetic data")
                    self.data_fetcher = PaperDataFetcher()
            except Exception as e:
                self.logger.error(f"[MT5] Error: {e} - falling back to synthetic data")
                self.data_fetcher = PaperDataFetcher()
        else:
            if not MT5_AVAILABLE:
                self.logger.warning("[MT5] MetaTrader5 library not installed - using synthetic data")
            else:
                self.logger.info("[CONFIG] USE_MT5_DATA=False - using synthetic data")
            self.data_fetcher = PaperDataFetcher()

        # Initialize other components
        self.signal_generator = SignalGenerator(min_confidence=MIN_SIGNAL_CONFIDENCE)
        self.position_sizer = PositionSizer()
        self.news_calendar = get_news_calendar()
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS)
        self.telegram.runner = self  # Enable command handling

        # Initialize accounts
        self.executors: Dict[str, PaperExecutor] = {}
        self._init_accounts()

        # State tracking
        self.running = True
        self.start_time = datetime.now()
        self.last_status_log = datetime.now()
        self.last_telegram_status = datetime.now()
        self.last_daily_report = datetime.now().date()
        self.total_signals = 0  # Only actionable signals (long/short)
        self.neutral_signals = 0  # Neutral signals (for debugging)
        self.total_trades = 0

        # Price cache
        self.current_prices: Dict[str, Dict[str, float]] = {}

        # Signal cooldown tracking (prevents flip-flopping)
        self.last_signal_time: Dict[str, datetime] = {}  # symbol -> last signal time

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_accounts(self):
        """Initialize paper trading accounts."""
        os.makedirs(STATE_DIR, exist_ok=True)

        for config in ACCOUNTS:
            executor = PaperExecutor(
                initial_balance=config["initial_balance"],
                account_type=config["account_type"],
                account_name=config["name"],
                state_file=os.path.join(STATE_DIR, f"{config['name']}_paper.json"),
                config={"MAX_POSITIONS": 3}
            )
            self.executors[config["name"]] = executor
            self.logger.info(
                f"  Account {config['name']}: ${config['initial_balance']:,.0f} "
                f"({config['account_type']})"
            )

        self.logger.info(f"  Symbols: {', '.join(SYMBOLS)}")
        self.logger.info("=" * 60)

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency (trades 24/7)."""
        crypto_patterns = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'AVAX', 'MATIC', 'LTC', 'BNB', 'DOGE', 'SHIB']
        upper = symbol.upper()
        return any(pattern in upper for pattern in crypto_patterns)

    def _is_market_open(self, symbol: str) -> bool:
        """
        Check if market is open for the given symbol.

        Crypto: Always open (24/7)
        Forex/Indices: Sunday 5 PM EST to Friday 5 PM EST
        """
        # Crypto trades 24/7
        if self._is_crypto(symbol):
            return True

        # Get current time in EST (UTC-5)
        from datetime import timezone
        utc_now = datetime.now(timezone.utc)
        est_offset = timedelta(hours=-5)
        est_now = utc_now + est_offset

        weekday = est_now.weekday()  # Monday=0, Sunday=6
        hour = est_now.hour

        # Market closed: Saturday all day
        if weekday == 5:  # Saturday
            return False

        # Market closed: Sunday before 5 PM EST
        if weekday == 6 and hour < 17:  # Sunday before 5 PM
            return False

        # Market closed: Friday after 5 PM EST
        if weekday == 4 and hour >= 17:  # Friday after 5 PM
            return False

        return True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("\n[SHUTDOWN] Received signal, stopping gracefully...")
        self.running = False

    def run(self):
        """Main paper trading loop."""
        self.logger.info("[START] Paper trading started")
        self.logger.info(f"[CONFIG] Scan interval: {SCAN_INTERVAL_SECONDS}s")
        self.logger.info(f"[CONFIG] Status log interval: {STATUS_LOG_INTERVAL_SECONDS}s")
        self.logger.info(f"[CONFIG] Telegram status interval: {TELEGRAM_STATUS_INTERVAL_SECONDS}s")

        # Log market status for each symbol
        open_symbols = [s for s in SYMBOLS if self._is_market_open(s)]
        closed_symbols = [s for s in SYMBOLS if not self._is_market_open(s)]
        if closed_symbols:
            self.logger.warning(f"[MARKET] Closed: {', '.join(closed_symbols)}")
        if open_symbols:
            self.logger.info(f"[MARKET] Open: {', '.join(open_symbols)}")
        else:
            self.logger.warning("[MARKET] All markets closed - no trading until markets reopen")

        # Refresh news calendar
        self.news_calendar.refresh()

        # Send Telegram startup notification
        self.telegram.send_startup(ACCOUNTS, SYMBOLS)

        scan_count = 0

        try:
            while self.running:
                scan_count += 1

                try:
                    # Fetch latest prices
                    self._fetch_prices()

                    # Update existing positions
                    self._update_positions()

                    # Check for new signals and execute trades
                    self._scan_for_signals()

                    # Log status periodically
                    self._maybe_log_status()

                    # Send Telegram status periodically
                    self._maybe_send_telegram_status()

                    # Save daily report
                    self._maybe_save_daily_report()

                    # Check for Telegram commands
                    self.telegram.check_commands()

                except Exception as e:
                    self.logger.error(f"[ERROR] Scan error: {e}", exc_info=True)

                # Wait for next scan
                if self.running:
                    time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            self.logger.info("[SHUTDOWN] Keyboard interrupt received")
        finally:
            self._generate_final_report()

    def _fetch_prices(self):
        """Fetch current prices for all symbols."""
        for symbol in SYMBOLS:
            try:
                # Get historical bars for signal generation
                df = self.data_fetcher.get_historical_bars(
                    symbol, TIMEFRAME, BARS_TO_FETCH
                )

                if not df.empty:
                    # Cache for signal generation
                    self.signal_generator.update_related_data(symbol, df)

                    # Get current price
                    last_bar = df.iloc[-1]
                    self.current_prices[symbol] = {
                        'bid': last_bar['close'] - (last_bar['close'] * 0.0001),
                        'ask': last_bar['close'] + (last_bar['close'] * 0.0001),
                        'price': last_bar['close'],
                        'close': last_bar['close'],
                        'high': last_bar['high'],
                        'low': last_bar['low'],
                    }

            except Exception as e:
                self.logger.warning(f"Failed to fetch {symbol}: {e}")

    def _update_positions(self):
        """Update all open positions with current prices."""
        for name, executor in self.executors.items():
            if executor.open_positions:
                # Map prices to symbol variants (with .x suffix for GFT)
                prices = {}
                for symbol, price_data in self.current_prices.items():
                    prices[symbol] = price_data
                    prices[f"{symbol}.x"] = price_data

                closed = executor.update_positions(prices)

                for pos in closed:
                    self.logger.info(
                        f"[{name}] Position closed: {pos.symbol} "
                        f"{pos.status.value} P&L=${pos.realized_pnl:+.2f}"
                    )
                    # Send Telegram notification
                    self.telegram.send_trade_closed(
                        name, pos.symbol, pos.realized_pnl, pos.status.value
                    )

    def _scan_for_signals(self):
        """Scan for trading signals across all symbols."""
        for symbol in SYMBOLS:
            try:
                # Skip if market is closed (except crypto which trades 24/7)
                if not self._is_market_open(symbol):
                    continue

                # Get data for signal generation
                df = self.data_fetcher.get_historical_bars(
                    symbol, TIMEFRAME, BARS_TO_FETCH
                )

                if df.empty or len(df) < 50:
                    continue

                # Generate signal
                sig = self.signal_generator.generate_signal(df, symbol)

                # Check if actionable
                if sig.action == "neutral":
                    self.neutral_signals += 1
                    continue

                # Check signal cooldown (prevent flip-flopping)
                now = datetime.now()
                last_signal = self.last_signal_time.get(symbol)
                if last_signal:
                    cooldown_remaining = (now - last_signal).total_seconds()
                    if cooldown_remaining < SIGNAL_COOLDOWN_SECONDS:
                        self.logger.debug(
                            f"[COOLDOWN] {symbol}: {SIGNAL_COOLDOWN_SECONDS - cooldown_remaining:.0f}s remaining"
                        )
                        continue

                # Update cooldown timer
                self.last_signal_time[symbol] = now

                # Count actionable signals
                self.total_signals += 1

                self.logger.info(
                    f"[SIGNAL] {symbol}: {sig.action.upper()} "
                    f"conf={sig.confidence:.2f} "
                    f"reason={sig.entry_reason}"
                )

                # Try to execute on each account
                self._execute_signal(sig)

            except Exception as e:
                self.logger.warning(f"Signal scan error for {symbol}: {e}")

    def _execute_signal(self, sig: TradingSignal):
        """Execute signal on all eligible accounts."""
        for name, executor in self.executors.items():
            try:
                # Check news blackout
                if self.news_calendar.is_blackout_period(executor.account_type):
                    blackout_info = self.news_calendar.get_blackout_info(executor.account_type)
                    self.logger.warning(
                        f"[{name}] News blackout: {blackout_info.get('event', 'unknown')}"
                    )
                    continue

                # Get current account state
                state = executor.get_account_state()

                # Check if we can take more positions
                if state.open_positions >= 3:
                    continue

                # Check if we already have a position in this symbol
                # Symbols come with .x suffix (from GFT MT5), strip for The5ers
                if executor.account_type == "GFT":
                    symbol_with_suffix = sig.symbol  # Already has .x
                else:
                    symbol_with_suffix = sig.symbol.replace('.x', '')  # Strip .x for The5ers
                existing = [
                    p for p in executor.get_open_positions()
                    if sig.symbol in p.symbol
                ]
                if existing:
                    continue

                # Calculate position size with compliance
                current_price = sig.current_price
                if current_price <= 0:
                    price_data = self.current_prices.get(sig.symbol, {})
                    current_price = price_data.get('close', 0)

                if current_price <= 0:
                    continue

                # Calculate stop distance
                stop_distance = abs(current_price - sig.stop_loss) if sig.stop_loss else current_price * 0.01
                stop_distance_pct = (stop_distance / current_price) * 100

                # Get compliant position size
                size_result, rejection = self.position_sizer.calculate_with_compliance(
                    account_balance=state.balance,
                    account_equity=state.equity,
                    stop_distance_pct=stop_distance_pct,
                    signal_confidence=sig.confidence,
                    account_type=executor.account_type,
                    current_dd_pct=state.current_dd_pct,
                    symbol=sig.symbol
                )

                if rejection:
                    self.logger.warning(f"[{name}] Position rejected: {rejection}")
                    continue

                if size_result.get('size', 0) <= 0:
                    continue

                # Normalize position size (round to reasonable lot size)
                position_size = self._normalize_size(sig.symbol, size_result['size'], current_price)

                if position_size <= 0:
                    continue

                # Calculate stop loss and take profit using wider stops
                atr = sig.atr if sig.atr > 0 else current_price * 0.01
                if sig.action == "long":
                    stop_loss = current_price - (atr * STOP_LOSS_ATR_MULT)
                    take_profit = current_price + (atr * TAKE_PROFIT_ATR_MULT)
                else:
                    stop_loss = current_price + (atr * STOP_LOSS_ATR_MULT)
                    take_profit = current_price - (atr * TAKE_PROFIT_ATR_MULT)

                # Execute trade
                success, msg, position = executor.open_position(
                    symbol=symbol_with_suffix,
                    direction=sig.action,
                    size=position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_info={
                        "confidence": sig.confidence,
                        "strategies": sig.strategies_agreeing,
                        "primary": sig.primary_strategy,
                    },
                    comment=sig.entry_reason
                )

                if success:
                    self.total_trades += 1
                    self.logger.info(
                        f"[{name}] TRADE OPENED: {sig.action.upper()} {symbol_with_suffix} "
                        f"size={position_size:.4f} @ {current_price:.2f} "
                        f"SL={stop_loss:.2f} TP={take_profit:.2f}"
                    )
                    # Send Telegram notification
                    self.telegram.send_trade_opened(
                        name, symbol_with_suffix, sig.action,
                        position_size, current_price, stop_loss, take_profit,
                        risk_amount=size_result.get('risk_amount', 0),
                        risk_pct=size_result.get('risk_pct', 0)
                    )
                else:
                    self.logger.warning(f"[{name}] Trade blocked: {msg}")

            except Exception as e:
                self.logger.error(f"[{name}] Execution error: {e}", exc_info=True)

    def _normalize_size(self, symbol: str, raw_size: float, price: float) -> float:
        """Normalize position size based on symbol type."""
        symbol_upper = symbol.upper()

        # Forex pairs (100,000 unit lots)
        if 'EUR' in symbol_upper or 'GBP' in symbol_upper or 'JPY' in symbol_upper:
            # Convert to lots (min 0.01)
            lots = raw_size / 100000
            return max(0.01, round(lots, 2)) * 100000

        # Gold (100 oz per lot)
        elif 'XAU' in symbol_upper:
            lots = raw_size / (price * 100)
            return max(0.01, round(lots, 2)) * 100

        # Silver (5000 oz per lot)
        elif 'XAG' in symbol_upper:
            lots = raw_size / (price * 5000)
            return max(0.01, round(lots, 2)) * 5000

        # Indices
        elif 'NAS' in symbol_upper or 'SPX' in symbol_upper or 'UK' in symbol_upper:
            # 1 contract per point
            lots = raw_size / price
            return max(0.1, round(lots, 1))

        # Default: round to 2 decimals
        return max(0.01, round(raw_size, 2))

    def _maybe_log_status(self):
        """Log status if interval has passed."""
        now = datetime.now()
        elapsed = (now - self.last_status_log).total_seconds()

        if elapsed >= STATUS_LOG_INTERVAL_SECONDS:
            self._log_status()
            self.last_status_log = now

    def _maybe_send_telegram_status(self):
        """Send Telegram status if interval has passed."""
        now = datetime.now()
        elapsed = (now - self.last_telegram_status).total_seconds()

        if elapsed >= TELEGRAM_STATUS_INTERVAL_SECONDS:
            states = {name: exec.get_account_state() for name, exec in self.executors.items()}
            runtime = str(now - self.start_time).split('.')[0]
            self.telegram.send_status(states, runtime)
            self.last_telegram_status = now

    def _log_status(self):
        """Log current account status."""
        self.logger.info("-" * 60)
        self.logger.info("  ACCOUNT STATUS")
        self.logger.info("-" * 60)

        total_equity = 0
        total_pnl = 0

        for name, executor in self.executors.items():
            state = executor.get_account_state()
            total_equity += state.equity
            total_pnl += state.realized_pnl + state.unrealized_pnl

            self.logger.info(
                f"  {name:12} | Equity: ${state.equity:>10,.2f} | "
                f"P&L: ${state.realized_pnl + state.unrealized_pnl:>+8,.2f} | "
                f"DD: {state.current_dd_pct:>5.2f}% | "
                f"Pos: {state.open_positions}"
            )

            # Check for compliance warnings
            if state.current_dd_pct >= 4.0:
                self.telegram.send_compliance_warning(
                    name, "Drawdown approaching limit", state.current_dd_pct
                )

        self.logger.info("-" * 60)
        self.logger.info(
            f"  TOTAL        | Equity: ${total_equity:>10,.2f} | "
            f"P&L: ${total_pnl:>+8,.2f}"
        )
        self.logger.info(
            f"  Signals: {self.total_signals} (actionable) | "
            f"Neutral: {self.neutral_signals} | "
            f"Trades: {self.total_trades} | "
            f"Runtime: {datetime.now() - self.start_time}"
        )
        self.logger.info("-" * 60)

    def _maybe_save_daily_report(self):
        """Save daily report at configured hour."""
        now = datetime.now()

        # Check if we've passed the report hour and haven't saved today
        if now.hour >= DAILY_REPORT_HOUR and now.date() > self.last_daily_report:
            self._save_daily_report()
            self.last_daily_report = now.date()

    def _save_daily_report(self):
        """Save daily trading report."""
        os.makedirs(LOGS_DIR, exist_ok=True)

        report = {
            "date": datetime.now().isoformat(),
            "runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "total_signals": self.total_signals,
            "neutral_signals": self.neutral_signals,
            "total_trades": self.total_trades,
            "accounts": {}
        }

        for name, executor in self.executors.items():
            state = executor.get_account_state()
            report["accounts"][name] = {
                "account_type": executor.account_type,
                "initial_balance": state.initial_balance,
                "balance": state.balance,
                "equity": state.equity,
                "realized_pnl": state.realized_pnl,
                "unrealized_pnl": state.unrealized_pnl,
                "total_pnl": state.realized_pnl + state.unrealized_pnl,
                "pnl_pct": ((state.equity - state.initial_balance) / state.initial_balance) * 100,
                "current_dd_pct": state.current_dd_pct,
                "max_dd_pct": state.max_dd_pct,
                "total_trades": state.total_trades,
                "win_rate": state.win_rate,
                "profit_factor": state.profit_factor,
                "open_positions": state.open_positions,
            }

        # Calculate totals
        total_initial = sum(r["initial_balance"] for r in report["accounts"].values())
        total_equity = sum(r["equity"] for r in report["accounts"].values())
        total_pnl = sum(r["total_pnl"] for r in report["accounts"].values())

        report["summary"] = {
            "total_initial_balance": total_initial,
            "total_equity": total_equity,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / total_initial) * 100 if total_initial > 0 else 0,
        }

        # Save report
        report_file = os.path.join(
            LOGS_DIR,
            f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        )

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"[REPORT] Daily report saved: {report_file}")

    def _generate_final_report(self):
        """Generate and display final report."""
        self.logger.info("\n")
        self.logger.info("=" * 60)
        self.logger.info("  FINAL PAPER TRADING REPORT")
        self.logger.info("=" * 60)

        runtime = datetime.now() - self.start_time
        self.logger.info(f"  Start Time:     {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"  End Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"  Runtime:        {runtime}")
        self.logger.info(f"  Total Signals:  {self.total_signals} (actionable)")
        self.logger.info(f"  Neutral Signals: {self.neutral_signals}")
        self.logger.info(f"  Total Trades:   {self.total_trades}")
        self.logger.info("=" * 60)

        total_initial = 0
        total_equity = 0
        total_realized = 0
        total_unrealized = 0

        # Build summary for Telegram
        summary = {
            "runtime": str(runtime).split('.')[0],
            "total_signals": self.total_signals,
            "total_trades": self.total_trades,
            "accounts": {},
            "total_pnl": 0,
        }

        for name, executor in self.executors.items():
            state = executor.get_account_state()

            total_initial += state.initial_balance
            total_equity += state.equity
            total_realized += state.realized_pnl
            total_unrealized += state.unrealized_pnl

            pnl = state.realized_pnl + state.unrealized_pnl
            pnl_pct = (pnl / state.initial_balance) * 100

            summary["accounts"][name] = {"total_pnl": pnl}

            self.logger.info(f"\n  [{name}] ({executor.account_type})")
            self.logger.info(f"  {'-' * 40}")
            self.logger.info(f"  Initial Balance:  ${state.initial_balance:>10,.2f}")
            self.logger.info(f"  Current Balance:  ${state.balance:>10,.2f}")
            self.logger.info(f"  Current Equity:   ${state.equity:>10,.2f}")
            self.logger.info(f"  Realized P&L:     ${state.realized_pnl:>+10,.2f}")
            self.logger.info(f"  Unrealized P&L:   ${state.unrealized_pnl:>+10,.2f}")
            self.logger.info(f"  Total P&L:        ${pnl:>+10,.2f} ({pnl_pct:+.2f}%)")
            self.logger.info(f"  Max Drawdown:     {state.max_dd_pct:>10.2f}%")
            self.logger.info(f"  Total Trades:     {state.total_trades:>10}")
            self.logger.info(f"  Win Rate:         {state.win_rate:>10.1f}%")
            self.logger.info(f"  Profit Factor:    {state.profit_factor:>10.2f}")
            self.logger.info(f"  Open Positions:   {state.open_positions:>10}")

            # Print trade history (last 5)
            history = executor.get_trade_history()
            if history:
                self.logger.info(f"\n  Recent Trades (last 5):")
                for trade in history[-5:]:
                    self.logger.info(
                        f"    {trade.symbol} {trade.direction.upper()} "
                        f"${trade.realized_pnl:+.2f} ({trade.status})"
                    )

        # Summary
        total_pnl = total_realized + total_unrealized
        total_pnl_pct = (total_pnl / total_initial) * 100 if total_initial > 0 else 0
        summary["total_pnl"] = total_pnl

        self.logger.info("\n" + "=" * 60)
        self.logger.info("  COMBINED SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"  Total Initial:    ${total_initial:>10,.2f}")
        self.logger.info(f"  Total Equity:     ${total_equity:>10,.2f}")
        self.logger.info(f"  Total P&L:        ${total_pnl:>+10,.2f} ({total_pnl_pct:+.2f}%)")
        self.logger.info(f"  Realized P&L:     ${total_realized:>+10,.2f}")
        self.logger.info(f"  Unrealized P&L:   ${total_unrealized:>+10,.2f}")
        self.logger.info("=" * 60)

        # Save final report
        self._save_daily_report()

        # Send Telegram shutdown notification
        self.telegram.send_shutdown(summary)

        # Disconnect MT5 if connected
        if self.mt5_connector:
            self.logger.info("[MT5] Disconnecting from MetaTrader 5...")
            self.mt5_connector.disconnect()

        self.logger.info("\n[END] Paper trading session complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  SOVEREIGN V5 - PAPER TRADING")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    runner = PaperTradingRunner()
    runner.run()


if __name__ == "__main__":
    main()
