# core/compliance/the5ers_compliance.py
"""
The5ers High Stakes Account Compliance Checker.

CRITICAL RULES (as of 2024):
- Daily loss limit: 5% of previous day's max(balance, equity)
- Max total drawdown: 10% STATIC from initial balance
- News blackout: +/- 2 min high-impact (blocks ALL order execution)
- Min profitable days: 3 days at 0.5%+ per phase
- Max inactivity: 30 days
- Leverage: 1:100 forex, 1:25 indices, 1:15 metals

ALLOWED (unlike GFT):
- Martingale / Grid trading (allowed but risky)

PROHIBITED:
- Hedging within same account
- HFT / Tick scalping
- Latency arbitrage
- Third-party prop firm EAs
- Copy trading between accounts
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from .gft_compliance import ComplianceAction, ComplianceResult, NewsEvent

logger = logging.getLogger(__name__)


class The5ersComplianceChecker:
    """
    The5ers High Stakes account compliance checker.

    KEY DIFFERENCES FROM GFT:
    - 10% STATIC DD (not trailing) from initial balance
    - 5% daily loss from previous day's max(balance, equity)
    - 2 min news blackout (stricter timing)
    - No per-trade floating loss limit
    - Martingale/Grid allowed (but we still avoid it)
    """

    # Actual limits
    MAX_DAILY_LOSS_PCT = 5.0
    MAX_TOTAL_DD_PCT = 10.0  # STATIC from initial balance

    # Guardian limits (stop before breach)
    GUARDIAN_DAILY_LOSS_PCT = 4.0
    GUARDIAN_TOTAL_DD_PCT = 8.0

    # Other limits
    NEWS_BLACKOUT_MINUTES = 2  # Stricter than GFT
    MIN_PROFITABLE_DAYS = 3
    PROFITABLE_DAY_THRESHOLD_PCT = 0.5
    MAX_INACTIVITY_DAYS = 30
    INACTIVITY_WARNING_DAYS = 25
    INACTIVITY_CRITICAL_DAYS = 28

    # Leverage limits (higher than GFT)
    LEVERAGE = {
        "forex": 100,
        "indices": 25,
        "metals": 15,
        "commodities": 15,
        "crypto": 5,
    }

    def __init__(self, initial_balance: float, account_name: str = "The5ers"):
        """
        Initialize compliance checker.

        Args:
            initial_balance: Initial account balance (STATIC reference for DD)
            account_name: Account identifier for logging
        """
        self.initial_balance = initial_balance
        self.account_name = account_name

        # Track previous day's max for daily loss calculation
        self.previous_day_max = initial_balance  # max(balance, equity) at end of day

        # Track profitable days for phase completion
        self.profitable_days = []

        logger.info(f"[{account_name}] The5ers Compliance initialized with ${initial_balance:.2f}")
        logger.info(f"[{account_name}] Static DD from initial: ${initial_balance:.2f}")

    def check_daily_loss(
        self,
        current_equity: float,
        previous_day_max: Optional[float] = None
    ) -> ComplianceResult:
        """
        Check 5% daily loss from previous day's max(balance, equity).

        Args:
            current_equity: Current account equity
            previous_day_max: Previous day's max(balance, equity), uses tracked if None

        Returns:
            ComplianceResult with action to take
        """
        if previous_day_max is None:
            previous_day_max = self.previous_day_max

        if previous_day_max <= 0:
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason="Previous day max is zero or negative"
            )

        daily_loss_pct = ((previous_day_max - current_equity) / previous_day_max) * 100

        # Only count if actually in loss
        if daily_loss_pct < 0:
            daily_loss_pct = 0

        # Check for breach
        if daily_loss_pct >= self.MAX_DAILY_LOSS_PCT:
            logger.critical(
                f"[{self.account_name}] DAILY LOSS BREACH! {daily_loss_pct:.2f}% (limit 5%)"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Daily loss breach: {daily_loss_pct:.2f}% (limit 5%)",
                details={"daily_loss_pct": daily_loss_pct, "limit": self.MAX_DAILY_LOSS_PCT}
            )

        # Guardian
        if daily_loss_pct >= self.GUARDIAN_DAILY_LOSS_PCT:
            logger.warning(
                f"[{self.account_name}] Daily loss guardian: {daily_loss_pct:.2f}% - stop trading"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Daily loss guardian: {daily_loss_pct:.2f}% (guardian 4%)",
                details={"daily_loss_pct": daily_loss_pct, "guardian": self.GUARDIAN_DAILY_LOSS_PCT}
            )

        # Warning at 3%
        if daily_loss_pct >= 3.0:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"Daily loss warning: {daily_loss_pct:.2f}%",
                details={"daily_loss_pct": daily_loss_pct}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Daily loss within limits")

    def check_total_drawdown(self, current_equity: float) -> ComplianceResult:
        """
        Check STATIC 10% DD from initial balance.

        KEY DIFFERENCE: The5ers uses STATIC DD from initial balance,
        NOT trailing from equity HWM like GFT.

        Args:
            current_equity: Current account equity

        Returns:
            ComplianceResult with action to take
        """
        if self.initial_balance <= 0:
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason="Initial balance is zero or negative"
            )

        # STATIC DD from initial balance
        dd_pct = ((self.initial_balance - current_equity) / self.initial_balance) * 100

        # Only count if actually in drawdown
        if dd_pct < 0:
            dd_pct = 0

        # Check for breach
        if dd_pct >= self.MAX_TOTAL_DD_PCT:
            logger.critical(
                f"[{self.account_name}] TOTAL DD BREACH! {dd_pct:.2f}% (limit 10%)"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Total DD breach: {dd_pct:.2f}% (limit 10%)",
                details={"total_dd_pct": dd_pct, "limit": self.MAX_TOTAL_DD_PCT}
            )

        # Guardian
        if dd_pct >= self.GUARDIAN_TOTAL_DD_PCT:
            logger.warning(
                f"[{self.account_name}] Total DD guardian: {dd_pct:.2f}% - stop trading"
            )
            return ComplianceResult(
                action=ComplianceAction.STOP_TRADING,
                reason=f"Total DD guardian: {dd_pct:.2f}% (guardian 8%)",
                details={"total_dd_pct": dd_pct, "guardian": self.GUARDIAN_TOTAL_DD_PCT}
            )

        # Warning at 6%
        if dd_pct >= 6.0:
            return ComplianceResult(
                action=ComplianceAction.WARNING,
                reason=f"Total DD warning: {dd_pct:.2f}%",
                details={"total_dd_pct": dd_pct}
            )

        return ComplianceResult(action=ComplianceAction.OK, reason="Total DD within limits")

    def check_news_blackout(
        self,
        current_time: datetime,
        news_events: List[NewsEvent]
    ) -> ComplianceResult:
        """
        Block ALL trades within +/- 2 min of high-impact news.

        The5ers is stricter than GFT - blocks all execution, not just profit cap.

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
                    f"[{self.account_name}] News blackout (STRICT): {event.name} "
                    f"({event.currency}) - ALL execution blocked"
                )
                return ComplianceResult(
                    action=ComplianceAction.BLOCK_TRADE,
                    reason=f"News blackout (2min): {event.name} - ALL execution blocked",
                    details={
                        "event": event.name,
                        "currency": event.currency,
                        "time_to_event_sec": time_diff
                    }
                )

        return ComplianceResult(action=ComplianceAction.OK, reason="No news blackout")

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
            asset_class: "forex", "indices", "metals", "commodities", "crypto"

        Returns:
            Maximum leverage allowed
        """
        return self.LEVERAGE.get(asset_class.lower(), 15)

    def update_previous_day_max(self, balance: float, equity: float):
        """
        Update previous day's max(balance, equity) at end of day.

        Call this at daily reset time.

        Args:
            balance: Current balance
            equity: Current equity
        """
        self.previous_day_max = max(balance, equity)
        logger.info(
            f"[{self.account_name}] Daily reset - previous day max: ${self.previous_day_max:.2f}"
        )

    def run_all_checks(
        self,
        current_equity: float,
        current_balance: float,
        news_events: Optional[List[NewsEvent]] = None,
        days_since_last_trade: int = 0
    ) -> List[ComplianceResult]:
        """
        Run all compliance checks.

        Args:
            current_equity: Current account equity
            current_balance: Current account balance
            news_events: Optional list of news events
            days_since_last_trade: Days since last trade

        Returns:
            List of ComplianceResult objects
        """
        results = []

        # Check daily loss
        results.append(self.check_daily_loss(current_equity))

        # Check total drawdown (STATIC from initial)
        results.append(self.check_total_drawdown(current_equity))

        # Check inactivity
        results.append(self.check_inactivity(days_since_last_trade))

        # Check news blackout if events provided
        if news_events:
            results.append(self.check_news_blackout(datetime.utcnow(), news_events))

        return results

    def get_status(self, current_equity: float) -> Dict[str, Any]:
        """
        Get current compliance status.

        Args:
            current_equity: Current account equity

        Returns:
            Dict with compliance status
        """
        daily_loss = max(0, ((self.previous_day_max - current_equity) / self.previous_day_max * 100)) if self.previous_day_max > 0 else 0
        total_dd = max(0, ((self.initial_balance - current_equity) / self.initial_balance * 100)) if self.initial_balance > 0 else 0

        return {
            "account": self.account_name,
            "initial_balance": self.initial_balance,
            "current_equity": current_equity,
            "previous_day_max": self.previous_day_max,
            "daily_loss_pct": daily_loss,
            "daily_loss_limit": self.MAX_DAILY_LOSS_PCT,
            "daily_loss_guardian": self.GUARDIAN_DAILY_LOSS_PCT,
            "total_dd_pct": total_dd,
            "total_dd_limit": self.MAX_TOTAL_DD_PCT,
            "total_dd_guardian": self.GUARDIAN_TOTAL_DD_PCT,
            "dd_type": "STATIC (from initial balance)",
        }
