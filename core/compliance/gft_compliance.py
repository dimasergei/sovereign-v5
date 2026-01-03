# core/compliance/gft_compliance.py
"""
GFT Instant GOAT Account Compliance Checker.

CRITICAL RULES (as of 2024):
- Max daily drawdown: 3% trailing (resets 5PM EST)
- Max total drawdown: 6% trailing from equity HWM
- Max floating loss per position: 2% (HARD BREACH - IMMEDIATE CLOSURE!)
- Max risk per trade: 2% per instrument/direction
- News blackout: +/- 5 min high-impact (profit capped at 1% if violated)
- Daily profit cap: $3,000
- Consistency rule: No single day >15% of total profits
- Min profitable days: 5 days at 0.5%+ each
- Max inactivity: 30 days
- Leverage: 1:50 forex, 1:10 indices/commodities

PROHIBITED:
- Martingale / Grid trading
- Hedging within same account
- HFT / Tick scalping
- Latency arbitrage
- Third-party prop firm EAs
- Copy trading between accounts
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplianceAction(Enum):
    """Actions to take based on compliance check."""
    OK = "ok"
    BLOCK_TRADE = "block_trade"
    CLOSE_IMMEDIATELY = "close_immediately"
    STOP_TRADING = "stop_trading"
    WARNING = "warning"


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    action: ComplianceAction
    reason: str
    details: Optional[Dict] = None

    def is_ok(self) -> bool:
        return self.action == ComplianceAction.OK


@dataclass
class NewsEvent:
    """High-impact news event."""
    time: datetime
    name: str
    currency: str
    impact: str  # "high", "medium", "low"


class GFTComplianceChecker:
    """
    GFT Instant GOAT account compliance checker.

    GUARDIAN LIMITS (stop before actual breach):
    - Daily DD: 2.5% guardian (actual 3%)
    - Total DD: 5% guardian (actual 6%)
    - Floating loss: 1.8% guardian (actual 2%)
    """

    # Actual limits
    MAX_DAILY_DD_PCT = 3.0
    MAX_TOTAL_DD_PCT = 6.0
    MAX_FLOATING_LOSS_PCT = 2.0  # Per position - HARD BREACH!
    MAX_RISK_PER_TRADE_PCT = 2.0

    # Guardian limits (stop before breach)
    GUARDIAN_DAILY_DD_PCT = 2.5
    GUARDIAN_TOTAL_DD_PCT = 5.0
    GUARDIAN_FLOATING_PCT = 1.8

    # Other limits
    DAILY_PROFIT_CAP_USD = 3000
    NEWS_BLACKOUT_MINUTES = 5
    MIN_PROFITABLE_DAYS = 5
    PROFITABLE_DAY_THRESHOLD_PCT = 0.5
    CONSISTENCY_MAX_SINGLE_DAY_PCT = 15.0
    MAX_INACTIVITY_DAYS = 30
    INACTIVITY_WARNING_DAYS = 25
    INACTIVITY_CRITICAL_DAYS = 28

    # Leverage limits
    LEVERAGE = {
        "forex": 50,
        "indices": 10,
        "commodities": 10,
        "crypto": 2,  # Very conservative for crypto
    }

    def __init__(self, initial_balance: float, account_name: str = "GFT"):
        """
        Initialize compliance checker.

        Args:
            initial_balance: Initial account balance
            account_name: Account identifier for logging
        """
        self.initial_balance = initial_balance
        self.account_name = account_name

        # Track daily state (resets at 5PM EST)
        self.daily_high_equity = initial_balance
        self.daily_starting_equity = initial_balance
        self.daily_profit = 0.0
        self.last_reset_date = None

        # Track equity high water mark
        self.equity_hwm = initial_balance

        # Track profitable days for payout eligibility
        self.profitable_days = []

        # Track daily profits for consistency rule
        self.daily_profits = {}  # date -> profit

        logger.info(f"[{account_name}] GFT Compliance initialized with ${initial_balance:.2f}")
        logger.warning(f"[{account_name}] CRITICAL: -2% floating loss = IMMEDIATE ACCOUNT CLOSURE")

    def check_position_floating_loss(
        self,
        unrealized_pnl: float,
        account_equity: float
    ) -> ComplianceResult:
        """
        CRITICAL: Check if position approaching -2% floating loss.

        This is the most dangerous GFT rule - breaching causes IMMEDIATE
        account closure with no warning.

        Args:
            unrealized_pnl: Position's unrealized P&L (negative for loss)
            account_equity: Current account equity

        Returns:
            ComplianceResult with action to take
        """
        if account_equity <= 0:
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason="Account equity is zero or negative"
            )

        floating_pnl_pct = (unrealized_pnl / account_equity) * 100

        # CRITICAL: Check for breach
        if floating_pnl_pct <= -self.MAX_FLOATING_LOSS_PCT:
            logger.critical(
                f"[{self.account_name}] BREACH! Floating loss {floating_pnl_pct:.2f}% "
                f"exceeds -2% limit!"
            )
            return ComplianceResult(
                action=ComplianceAction.CLOSE_IMMEDIATELY,
                reason=f"FLOATING LOSS BREACH: {floating_pnl_pct:.2f}% (limit -2%)",
                details={"floating_pct": floating_pnl_pct, "limit": -self.MAX_FLOATING_LOSS_PCT}
            )

        # Guardian: Close before breach
        if floating_pnl_pct <= -self.GUARDIAN_FLOATING_PCT:
            logger.warning(
                f"[{self.account_name}] Guardian triggered! Floating loss {floating_pnl_pct:.2f}% "
                f"approaching -2% limit"
            )
            return ComplianceResult(
                action=ComplianceAction.CLOSE_IMMEDIATELY,
                reason=f"Guardian: Floating loss {floating_pnl_pct:.2f}% approaching -2% breach",
                details={"floating_pct": floating_pnl_pct, "guardian": -self.GUARDIAN_FLOATING_PCT}
            )

        # Warning at 1.5%
        if floating_pnl_pct <= -1.5:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"Floating loss {floating_pnl_pct:.2f}% - monitor closely",
                details={"floating_pct": floating_pnl_pct}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Floating loss within limits")

    def check_daily_drawdown(
        self,
        current_equity: float,
        daily_high_equity: Optional[float] = None
    ) -> ComplianceResult:
        """
        Check daily drawdown from 5PM EST reset.

        Daily DD is calculated from the highest equity since daily reset.

        Args:
            current_equity: Current account equity
            daily_high_equity: Highest equity since daily reset (uses tracked value if None)

        Returns:
            ComplianceResult with action to take
        """
        if daily_high_equity is None:
            daily_high_equity = self.daily_high_equity

        # Update HWM
        if current_equity > self.daily_high_equity:
            self.daily_high_equity = current_equity

        if daily_high_equity <= 0:
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason="Daily high equity is zero or negative"
            )

        daily_dd_pct = ((daily_high_equity - current_equity) / daily_high_equity) * 100

        # Check for breach
        if daily_dd_pct >= self.MAX_DAILY_DD_PCT:
            logger.critical(
                f"[{self.account_name}] DAILY DD BREACH! {daily_dd_pct:.2f}% (limit 3%)"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Daily DD breach: {daily_dd_pct:.2f}% (limit 3%)",
                details={"daily_dd_pct": daily_dd_pct, "limit": self.MAX_DAILY_DD_PCT}
            )

        # Guardian
        if daily_dd_pct >= self.GUARDIAN_DAILY_DD_PCT:
            logger.warning(
                f"[{self.account_name}] Daily DD guardian: {daily_dd_pct:.2f}% - stop trading"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Daily DD guardian: {daily_dd_pct:.2f}% (guardian 2.5%)",
                details={"daily_dd_pct": daily_dd_pct, "guardian": self.GUARDIAN_DAILY_DD_PCT}
            )

        # Warning at 2%
        if daily_dd_pct >= 2.0:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"Daily DD warning: {daily_dd_pct:.2f}%",
                details={"daily_dd_pct": daily_dd_pct}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Daily DD within limits")

    def check_total_drawdown(self, current_equity: float) -> ComplianceResult:
        """
        Check total drawdown from equity HWM.

        GFT uses TRAILING drawdown from equity high water mark.

        Args:
            current_equity: Current account equity

        Returns:
            ComplianceResult with action to take
        """
        # Update equity HWM
        if current_equity > self.equity_hwm:
            self.equity_hwm = current_equity

        if self.equity_hwm <= 0:
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason="Equity HWM is zero or negative"
            )

        total_dd_pct = ((self.equity_hwm - current_equity) / self.equity_hwm) * 100

        # Check for breach
        if total_dd_pct >= self.MAX_TOTAL_DD_PCT:
            logger.critical(
                f"[{self.account_name}] TOTAL DD BREACH! {total_dd_pct:.2f}% (limit 6%)"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Total DD breach: {total_dd_pct:.2f}% (limit 6%)",
                details={"total_dd_pct": total_dd_pct, "limit": self.MAX_TOTAL_DD_PCT}
            )

        # Guardian
        if total_dd_pct >= self.GUARDIAN_TOTAL_DD_PCT:
            logger.warning(
                f"[{self.account_name}] Total DD guardian: {total_dd_pct:.2f}% - stop trading"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Total DD guardian: {total_dd_pct:.2f}% (guardian 5%)",
                details={"total_dd_pct": total_dd_pct, "guardian": self.GUARDIAN_TOTAL_DD_PCT}
            )

        # Warning at 4%
        if total_dd_pct >= 4.0:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"Total DD warning: {total_dd_pct:.2f}%",
                details={"total_dd_pct": total_dd_pct}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Total DD within limits")

    def check_news_blackout(
        self,
        current_time: datetime,
        news_events: List[NewsEvent]
    ) -> ComplianceResult:
        """
        Block trades within +/- 5 min of high-impact news.

        If violated, profit is capped at 1% of account.

        Args:
            current_time: Current UTC time
            news_events: List of upcoming high-impact news events

        Returns:
            ComplianceResult with action to take
        """
        blackout_seconds = self.NEWS_BLACKOUT_MINUTES * 60

        for event in news_events:
            time_diff = abs((current_time - event.time).total_seconds())

            if time_diff <= blackout_seconds:
                logger.warning(
                    f"[{self.account_name}] News blackout: {event.name} "
                    f"({event.currency}) in {time_diff/60:.1f} min"
                )
                return ComplianceResult(
                    action=ComplianceAction.BLOCK_TRADE,
                    reason=f"News blackout: {event.name} ({event.currency})",
                    details={
                        "event": event.name,
                        "currency": event.currency,
                        "time_to_event_sec": time_diff
                    }
                )

        return ComplianceResult(action=ComplianceAction.OK, reason="No news blackout")

    def check_daily_profit_cap(self, daily_profit: float) -> ComplianceResult:
        """
        Check if approaching $3000 daily profit cap.

        Args:
            daily_profit: Today's profit in USD

        Returns:
            ComplianceResult with action to take
        """
        self.daily_profit = daily_profit

        if daily_profit >= self.DAILY_PROFIT_CAP_USD:
            logger.info(
                f"[{self.account_name}] Daily profit cap reached: ${daily_profit:.2f}"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Daily profit cap reached: ${daily_profit:.2f}",
                details={"daily_profit": daily_profit, "cap": self.DAILY_PROFIT_CAP_USD}
            )

        # Warning at 90%
        if daily_profit >= self.DAILY_PROFIT_CAP_USD * 0.9:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"Approaching daily profit cap: ${daily_profit:.2f}",
                details={"daily_profit": daily_profit, "cap": self.DAILY_PROFIT_CAP_USD}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Daily profit within cap")

    def check_risk_per_trade(
        self,
        risk_amount: float,
        account_balance: float
    ) -> ComplianceResult:
        """
        Check if trade risk exceeds 2% limit.

        Args:
            risk_amount: Dollar risk for the trade
            account_balance: Current account balance

        Returns:
            ComplianceResult with action to take
        """
        if account_balance <= 0:
            return ComplianceResult(
                action=ComplianceAction.BLOCK_TRADE,
                reason="Account balance is zero or negative"
            )

        risk_pct = (risk_amount / account_balance) * 100

        if risk_pct > self.MAX_RISK_PER_TRADE_PCT:
            logger.warning(
                f"[{self.account_name}] Risk per trade too high: {risk_pct:.2f}% (max 2%)"
            )
            return ComplianceResult(
                action=ComplianceAction.BLOCK_TRADE,
                reason=f"Risk per trade {risk_pct:.2f}% exceeds 2% limit",
                details={"risk_pct": risk_pct, "limit": self.MAX_RISK_PER_TRADE_PCT}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Risk per trade within limits")

    def check_stop_distance(
        self,
        stop_distance_pct: float,
        account_balance: float,
        position_size: float
    ) -> ComplianceResult:
        """
        Check if stop distance would breach -2% floating loss on full adverse move.

        Args:
            stop_distance_pct: Stop loss distance as percentage
            account_balance: Current account balance
            position_size: Position size in units

        Returns:
            ComplianceResult with action to take
        """
        # Calculate max loss at stop
        max_loss = position_size * (stop_distance_pct / 100)
        max_loss_pct = (max_loss / account_balance) * 100

        if max_loss_pct > self.GUARDIAN_FLOATING_PCT:
            return ComplianceResult(
                action=ComplianceAction.BLOCK_TRADE,
                reason=f"Stop too wide: {max_loss_pct:.2f}% potential loss exceeds 1.8% guardian",
                details={
                    "potential_loss_pct": max_loss_pct,
                    "guardian": self.GUARDIAN_FLOATING_PCT
                }
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Stop distance acceptable")

    def check_inactivity(self, days_since_last_trade: int) -> ComplianceResult:
        """
        Check inactivity status.

        Args:
            days_since_last_trade: Days since last trade

        Returns:
            ComplianceResult with action to take
        """
        if days_since_last_trade >= self.MAX_INACTIVITY_DAYS:
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Inactivity breach: {days_since_last_trade} days",
                details={"days_inactive": days_since_last_trade}
            )

        if days_since_last_trade >= self.INACTIVITY_CRITICAL_DAYS:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"CRITICAL: {days_since_last_trade} days inactive - place trade NOW",
                details={"days_inactive": days_since_last_trade}
            )

        if days_since_last_trade >= self.INACTIVITY_WARNING_DAYS:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"Inactivity warning: {days_since_last_trade} days",
                details={"days_inactive": days_since_last_trade}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Activity within limits")

    def get_leverage_limit(self, asset_class: str) -> int:
        """
        Get leverage limit for asset class.

        Args:
            asset_class: "forex", "indices", "commodities", "crypto"

        Returns:
            Maximum leverage allowed
        """
        return self.LEVERAGE.get(asset_class.lower(), 10)

    def run_all_checks(
        self,
        current_equity: float,
        positions: List[Dict],
        proposed_trade: Optional[Dict] = None,
        news_events: Optional[List[NewsEvent]] = None,
        days_since_last_trade: int = 0
    ) -> List[ComplianceResult]:
        """
        Run all compliance checks.

        Args:
            current_equity: Current account equity
            positions: List of open positions with unrealized_pnl
            proposed_trade: Optional proposed trade to validate
            news_events: Optional list of news events
            days_since_last_trade: Days since last trade

        Returns:
            List of ComplianceResult objects
        """
        results = []

        # Check each position for floating loss
        for pos in positions:
            result = self.check_position_floating_loss(
                pos.get('unrealized_pnl', 0),
                current_equity
            )
            if not result.is_ok():
                result.details = result.details or {}
                result.details['position'] = pos.get('symbol', 'unknown')
            results.append(result)

        # Check daily drawdown
        results.append(self.check_daily_drawdown(current_equity))

        # Check total drawdown
        results.append(self.check_total_drawdown(current_equity))

        # Check daily profit cap
        results.append(self.check_daily_profit_cap(self.daily_profit))

        # Check inactivity
        results.append(self.check_inactivity(days_since_last_trade))

        # Check news blackout if events provided
        if news_events and proposed_trade:
            results.append(self.check_news_blackout(datetime.utcnow(), news_events))

        # Check proposed trade risk
        if proposed_trade:
            results.append(self.check_risk_per_trade(
                proposed_trade.get('risk_amount', 0),
                current_equity
            ))

        return results

    def get_status(self, current_equity: float) -> Dict[str, Any]:
        """
        Get current compliance status.

        Args:
            current_equity: Current account equity

        Returns:
            Dict with compliance status
        """
        daily_dd = ((self.daily_high_equity - current_equity) / self.daily_high_equity * 100) if self.daily_high_equity > 0 else 0
        total_dd = ((self.equity_hwm - current_equity) / self.equity_hwm * 100) if self.equity_hwm > 0 else 0

        return {
            "account": self.account_name,
            "equity_hwm": self.equity_hwm,
            "current_equity": current_equity,
            "daily_high_equity": self.daily_high_equity,
            "daily_dd_pct": daily_dd,
            "daily_dd_limit": self.MAX_DAILY_DD_PCT,
            "daily_dd_guardian": self.GUARDIAN_DAILY_DD_PCT,
            "total_dd_pct": total_dd,
            "total_dd_limit": self.MAX_TOTAL_DD_PCT,
            "total_dd_guardian": self.GUARDIAN_TOTAL_DD_PCT,
            "daily_profit": self.daily_profit,
            "daily_profit_cap": self.DAILY_PROFIT_CAP_USD,
            "floating_loss_limit": self.MAX_FLOATING_LOSS_PCT,
            "floating_loss_guardian": self.GUARDIAN_FLOATING_PCT,
        }
