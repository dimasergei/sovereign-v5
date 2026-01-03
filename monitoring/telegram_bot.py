"""
Telegram Command Center - Bot monitoring and control.

Provides real-time alerts, status reports, and remote control
for the trading bots.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

# Try to import telegram library
try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application, CommandHandler, ContextTypes, MessageHandler, filters
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"
    TRADE = "trade"


ALERT_EMOJIS = {
    AlertLevel.INFO: "ℹ️",
    AlertLevel.WARNING: "⚠️",
    AlertLevel.CRITICAL: "🚨",
    AlertLevel.SUCCESS: "✅",
    AlertLevel.TRADE: "📊",
}


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str
    authorized_chat_ids: List[int]
    rate_limit_messages: int = 30
    rate_limit_window: int = 60  # seconds


class TelegramCommandCenter:
    """
    Telegram bot for monitoring and controlling trading bots.
    
    Commands:
        /start - Welcome message
        /help - List all commands
        /status - Account status, positions, drawdown
        /pause - Pause trading
        /resume - Resume trading
        /stop - EMERGENCY: close all positions
        /performance - Win rate, profit factor, Sharpe
        /risk - Current DD%, daily P&L
        /trades - Recent trade history
        /unlock - Unlock account after review
    """
    
    def __init__(
        self,
        config: TelegramConfig,
        callbacks: Dict[str, Callable] = None
    ):
        """
        Initialize Telegram command center.
        
        Args:
            config: Telegram configuration
            callbacks: Dictionary of callback functions:
                - on_pause: Called when /pause command received
                - on_resume: Called when /resume command received
                - on_stop: Called when /stop command received
                - get_status: Returns status dict
                - get_performance: Returns performance dict
                - get_risk: Returns risk dict
                - get_trades: Returns recent trades list
                - run_backtest: Runs backtest with params
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot not installed")
        
        self.config = config
        self.callbacks = callbacks or {}
        
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self._running = False
        
        # Rate limiting
        self._message_times: Dict[int, List[float]] = defaultdict(list)
    
    async def start(self):
        """Start the Telegram bot."""
        logger.info("Starting Telegram bot...")
        
        self.application = Application.builder().token(self.config.bot_token).build()
        self.bot = self.application.bot
        
        # Register handlers
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("pause", self._cmd_pause))
        self.application.add_handler(CommandHandler("resume", self._cmd_resume))
        self.application.add_handler(CommandHandler("stop", self._cmd_stop))
        self.application.add_handler(CommandHandler("performance", self._cmd_performance))
        self.application.add_handler(CommandHandler("risk", self._cmd_risk))
        self.application.add_handler(CommandHandler("trades", self._cmd_trades))
        self.application.add_handler(CommandHandler("unlock", self._cmd_unlock))
        
        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        self._running = True
        logger.info("Telegram bot started")
    
    async def stop(self):
        """Stop the Telegram bot."""
        if self.application:
            logger.info("Stopping Telegram bot...")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            self._running = False
            logger.info("Telegram bot stopped")
    
    async def send_alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        chat_id: int = None
    ):
        """
        Send an alert message.
        
        Args:
            message: Alert message
            level: Alert severity level
            chat_id: Specific chat to send to (None = all authorized)
        """
        if not self.bot:
            logger.warning("Bot not initialized, cannot send alert")
            return
        
        emoji = ALERT_EMOJIS.get(level, "")
        formatted = f"{emoji} *{level.value.upper()}*\n{message}"
        
        targets = [chat_id] if chat_id else self.config.authorized_chat_ids
        
        for target in targets:
            try:
                await self.bot.send_message(
                    chat_id=target,
                    text=formatted,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Failed to send alert to {target}: {e}")
    
    async def send_trade_notification(
        self,
        symbol: str,
        action: str,
        direction: str,
        price: float,
        size: float,
        pnl: float = None,
        chat_id: int = None,
        risk_amount: float = None,
        risk_pct: float = None
    ):
        """Send trade notification."""
        pnl_str = f"\nP&L: ${pnl:+.2f}" if pnl is not None else ""

        # Show risk amount and percentage if provided, otherwise fall back to lots
        if risk_amount is not None and risk_pct is not None:
            size_str = f"Risk: ${risk_amount:.2f} ({risk_pct:.2f}%)"
        else:
            size_str = f"Size: {size:.2f} lots"

        message = (
            f"🔔 *{action}*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: {direction}\n"
            f"Price: {price:.5f}\n"
            f"{size_str}"
            f"{pnl_str}"
        )

        await self.send_alert(message, AlertLevel.TRADE, chat_id)
    
    def _is_authorized(self, chat_id: int) -> bool:
        """Check if chat is authorized."""
        return chat_id in self.config.authorized_chat_ids
    
    def _is_rate_limited(self, chat_id: int) -> bool:
        """Check if chat is rate limited."""
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        # Clean old messages
        self._message_times[chat_id] = [
            t for t in self._message_times[chat_id] if t > window_start
        ]
        
        if len(self._message_times[chat_id]) >= self.config.rate_limit_messages:
            return True
        
        self._message_times[chat_id].append(now)
        return False
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        chat_id = update.effective_chat.id
        
        if not self._is_authorized(chat_id):
            await update.message.reply_text(
                f"❌ Unauthorized. Your chat ID is: {chat_id}"
            )
            return
        
        await update.message.reply_text(
            "🤖 *Prop Trading Bot*\n\n"
            "Welcome! Use /help to see available commands.\n\n"
            "Bot is monitoring your trading accounts.",
            parse_mode="Markdown"
        )
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        help_text = """
📋 *Available Commands*

*Information*
/status - Account status and positions
/performance - Trading statistics
/risk - Risk metrics and drawdown
/trades - Recent trade history

*Control*
/pause - Pause trading
/resume - Resume trading
/stop - ⚠️ EMERGENCY close all

*Admin*
/unlock - Unlock account after review
/help - Show this help
        """
        
        await update.message.reply_text(help_text.strip(), parse_mode="Markdown")
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        if self._is_rate_limited(update.effective_chat.id):
            await update.message.reply_text("⏳ Rate limited, please wait")
            return
        
        if 'get_status' in self.callbacks:
            try:
                status = self.callbacks['get_status']()
                
                message = (
                    f"📊 *Account Status*\n\n"
                    f"Balance: ${status.get('balance', 0):.2f}\n"
                    f"Equity: ${status.get('equity', 0):.2f}\n"
                    f"Profit: ${status.get('profit', 0):.2f}\n"
                    f"Drawdown: {status.get('drawdown_pct', 0):.2f}%\n"
                    f"Open Positions: {status.get('open_positions', 0)}\n"
                    f"Trading: {'🟢 Active' if status.get('is_trading', False) else '🔴 Paused'}"
                )
                
                await update.message.reply_text(message, parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text("Status callback not configured")
    
    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        if 'on_pause' in self.callbacks:
            try:
                self.callbacks['on_pause']()
                await update.message.reply_text("⏸️ Trading paused")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text("Pause callback not configured")
    
    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        if 'on_resume' in self.callbacks:
            try:
                self.callbacks['on_resume']()
                await update.message.reply_text("▶️ Trading resumed")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text("Resume callback not configured")
    
    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - EMERGENCY."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        await update.message.reply_text(
            "⚠️ *EMERGENCY STOP*\n\n"
            "This will close ALL positions immediately.\n"
            "Type `/stop CONFIRM` to proceed.",
            parse_mode="Markdown"
        )
        
        if context.args and context.args[0] == "CONFIRM":
            if 'on_stop' in self.callbacks:
                try:
                    self.callbacks['on_stop']()
                    await update.message.reply_text("🛑 All positions closed, trading stopped")
                except Exception as e:
                    await update.message.reply_text(f"❌ Error: {e}")
    
    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        if self._is_rate_limited(update.effective_chat.id):
            await update.message.reply_text("⏳ Rate limited, please wait")
            return
        
        if 'get_performance' in self.callbacks:
            try:
                perf = self.callbacks['get_performance']()
                
                message = (
                    f"📈 *Performance*\n\n"
                    f"Total Trades: {perf.get('total_trades', 0)}\n"
                    f"Win Rate: {perf.get('win_rate', 0):.1f}%\n"
                    f"Profit Factor: {perf.get('profit_factor', 0):.2f}\n"
                    f"Total Profit: ${perf.get('total_profit', 0):.2f}\n"
                    f"Best Trade: ${perf.get('best_trade', 0):.2f}\n"
                    f"Worst Trade: ${perf.get('worst_trade', 0):.2f}"
                )
                
                await update.message.reply_text(message, parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text("Performance callback not configured")
    
    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        if self._is_rate_limited(update.effective_chat.id):
            await update.message.reply_text("⏳ Rate limited, please wait")
            return
        
        if 'get_risk' in self.callbacks:
            try:
                risk = self.callbacks['get_risk']()
                
                dd = risk.get('drawdown', {})
                daily = risk.get('daily_loss', {})
                
                message = (
                    f"🎯 *Risk Status*\n\n"
                    f"*Drawdown*\n"
                    f"Current: {dd.get('current_pct', 0):.2f}%\n"
                    f"Guardian: {dd.get('guardian_pct', 0):.2f}%\n"
                    f"Max: {dd.get('max_pct', 0):.2f}%\n"
                    f"Distance: {dd.get('distance_to_guardian', 0):.2f}%\n"
                )
                
                if daily:
                    message += (
                        f"\n*Daily Loss*\n"
                        f"Current: {daily.get('current_pct', 0):.2f}%\n"
                        f"Guardian: {daily.get('guardian_pct', 0):.2f}%\n"
                        f"P&L: ${daily.get('pnl_dollars', 0):.2f}"
                    )
                
                await update.message.reply_text(message, parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text("Risk callback not configured")
    
    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        if 'get_trades' in self.callbacks:
            try:
                trades = self.callbacks['get_trades']()
                
                if not trades:
                    await update.message.reply_text("No recent trades")
                    return
                
                message = "📜 *Recent Trades*\n\n"
                
                for t in trades[:10]:  # Last 10
                    emoji = "🟢" if t.get('pnl', 0) >= 0 else "🔴"
                    message += (
                        f"{emoji} {t.get('symbol', '?')} "
                        f"${t.get('pnl', 0):+.2f}\n"
                    )
                
                await update.message.reply_text(message, parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text("Trades callback not configured")
    
    async def _cmd_unlock(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unlock command."""
        if not self._is_authorized(update.effective_chat.id):
            return
        
        await update.message.reply_text(
            "🔓 To unlock the account, type `/unlock CONFIRM`",
            parse_mode="Markdown"
        )
        
        if context.args and context.args[0] == "CONFIRM":
            if 'unlock' in self.callbacks:
                try:
                    self.callbacks['unlock']()
                    await update.message.reply_text("✅ Account unlocked")
                except Exception as e:
                    await update.message.reply_text(f"❌ Error: {e}")


# Synchronous wrapper for use in trading bots
class TelegramNotifier:
    """
    Synchronous wrapper for Telegram notifications.
    
    Use this in the main bot loop for sending alerts without
    dealing with async/await.
    """
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bot: Optional[Bot] = None
        
        if TELEGRAM_AVAILABLE:
            self._bot = Bot(token=config.bot_token)
    
    def send_alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO
    ):
        """Send alert synchronously."""
        if not self._bot:
            return
        
        emoji = ALERT_EMOJIS.get(level, "")
        formatted = f"{emoji} *{level.value.upper()}*\n{message}"
        
        for chat_id in self.config.authorized_chat_ids:
            try:
                asyncio.run(
                    self._bot.send_message(
                        chat_id=chat_id,
                        text=formatted,
                        parse_mode="Markdown"
                    )
                )
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
    
    def send_trade(
        self,
        symbol: str,
        action: str,
        direction: str,
        price: float,
        size: float,
        pnl: float = None,
        risk_amount: float = None,
        risk_pct: float = None
    ):
        """Send trade notification synchronously."""
        pnl_str = f"\nP&L: ${pnl:+.2f}" if pnl is not None else ""

        # Show risk amount and percentage if provided, otherwise fall back to lots
        if risk_amount is not None and risk_pct is not None:
            size_str = f"Risk: ${risk_amount:.2f} ({risk_pct:.2f}%)"
        else:
            size_str = f"Size: {size:.2f} lots"

        message = (
            f"🔔 *{action}*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: {direction}\n"
            f"Price: {price:.5f}\n"
            f"{size_str}"
            f"{pnl_str}"
        )

        self.send_alert(message, AlertLevel.TRADE)
