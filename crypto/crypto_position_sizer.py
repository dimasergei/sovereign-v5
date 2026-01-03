# crypto/crypto_position_sizer.py
"""
Crypto Position Sizer for The5ers Account.

Conservative sizing due to crypto volatility:
- Base risk: 0.5% per trade (half of forex)
- Volatility adjustment: Reduce size when ATR high
- Leverage limit: 30:1 for crypto on The5ers
- Max exposure: 50% of available leverage

The5ers Rules:
- Max daily drawdown: 4% (guardian: 3.5%)
- Max total drawdown: 6% (guardian: 5.5%)
- Leverage: 30:1 for crypto
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CryptoPositionResult:
    """Position sizing result for crypto."""
    size: float              # Position size in units
    size_usd: float          # Position size in USD
    risk_amount: float       # Dollar amount at risk
    risk_pct: float          # Percentage of account risked
    volatility_scalar: float # Volatility adjustment applied
    leverage_used: float     # Actual leverage used
    max_leverage: float      # Maximum allowed leverage
    approved: bool           # Whether trade is approved
    reason: str              # Approval/rejection reason


class CryptoPositionSizer:
    """
    Conservative position sizing for crypto trading on The5ers.

    Key principles:
    1. Never risk more than 0.5% per trade (half of forex)
    2. Reduce size when volatility is elevated
    3. Never use more than 50% of available leverage
    4. Hard stop if approaching drawdown limits
    """

    # Base parameters
    BASE_RISK_PCT = 0.005      # 0.5% base risk per trade
    MAX_RISK_PCT = 0.01        # 1.0% absolute max
    MIN_RISK_PCT = 0.002       # 0.2% minimum (avoid dust trades)

    # The5ers limits
    THE5ERS_CRYPTO_LEVERAGE = 30
    MAX_LEVERAGE_USAGE = 0.50  # Use max 50% of available leverage

    # Drawdown guardians
    DAILY_DD_GUARDIAN = 0.035     # 3.5% (actual limit 4%)
    TOTAL_DD_GUARDIAN = 0.055     # 5.5% (actual limit 6%)

    # Volatility scaling
    ATR_LOOKBACK = 14
    ATR_AVG_LOOKBACK = 100
    MIN_VOLATILITY_SCALAR = 0.5  # Never go below 50% of base size
    MAX_VOLATILITY_SCALAR = 1.0  # Never exceed base size

    def __init__(self, account_balance: float = 5000.0):
        """
        Initialize crypto position sizer.

        Args:
            account_balance: The5ers account balance (default $5000)
        """
        self.account_balance = account_balance
        self.peak_balance = account_balance
        self.daily_pnl = 0.0
        self.current_exposure = 0.0

    def calculate_position(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        current_equity: Optional[float] = None
    ) -> CryptoPositionResult:
        """
        Calculate safe position size for crypto trade.

        Args:
            entry_price: Intended entry price
            stop_loss: Stop loss price
            symbol: Trading symbol (BTCUSD, ETHUSD, etc.)
            df: OHLCV data for volatility calculation
            current_equity: Current account equity

        Returns:
            CryptoPositionResult with sizing details
        """
        if current_equity is not None:
            self.account_balance = current_equity

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pct = stop_distance / entry_price

        if stop_distance_pct == 0:
            return CryptoPositionResult(
                size=0, size_usd=0, risk_amount=0, risk_pct=0,
                volatility_scalar=0, leverage_used=0,
                max_leverage=self.THE5ERS_CRYPTO_LEVERAGE,
                approved=False, reason="Invalid stop loss (same as entry)"
            )

        # Check drawdown limits
        dd_check = self._check_drawdown_limits()
        if not dd_check[0]:
            return CryptoPositionResult(
                size=0, size_usd=0, risk_amount=0, risk_pct=0,
                volatility_scalar=0, leverage_used=0,
                max_leverage=self.THE5ERS_CRYPTO_LEVERAGE,
                approved=False, reason=dd_check[1]
            )

        # Calculate base risk
        risk_pct = self.BASE_RISK_PCT

        # Apply volatility adjustment
        volatility_scalar = 1.0
        if df is not None and len(df) >= self.ATR_AVG_LOOKBACK:
            volatility_scalar = self._calculate_volatility_scalar(df)
            risk_pct *= volatility_scalar

        # Apply drawdown proximity scaling
        dd_scalar = self._calculate_dd_scalar()
        risk_pct *= dd_scalar

        # Clamp to limits
        risk_pct = max(self.MIN_RISK_PCT, min(self.MAX_RISK_PCT, risk_pct))

        # Calculate dollar risk
        risk_amount = self.account_balance * risk_pct

        # Calculate position size
        position_value = risk_amount / stop_distance_pct

        # Check leverage limits
        max_position_value = self.account_balance * self.THE5ERS_CRYPTO_LEVERAGE * self.MAX_LEVERAGE_USAGE
        if position_value > max_position_value:
            position_value = max_position_value
            risk_amount = position_value * stop_distance_pct
            risk_pct = risk_amount / self.account_balance

        # Calculate actual leverage used
        leverage_used = position_value / self.account_balance

        # Calculate size in units (e.g., BTC for BTCUSD)
        size = position_value / entry_price

        # Round to reasonable precision
        if 'BTC' in symbol.upper():
            size = round(size, 4)  # 0.0001 BTC precision
        elif 'ETH' in symbol.upper():
            size = round(size, 3)  # 0.001 ETH precision
        else:
            size = round(size, 2)

        return CryptoPositionResult(
            size=size,
            size_usd=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            volatility_scalar=volatility_scalar,
            leverage_used=leverage_used,
            max_leverage=self.THE5ERS_CRYPTO_LEVERAGE,
            approved=True,
            reason=f"Approved: {risk_pct:.2%} risk, {leverage_used:.1f}x leverage, volatility scalar {volatility_scalar:.2f}"
        )

    def _calculate_volatility_scalar(self, df: pd.DataFrame) -> float:
        """
        Calculate volatility adjustment scalar.

        When volatility is high, reduce position size.
        When volatility is low, maintain base size.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Current ATR
        current_atr = self._calculate_atr(high, low, close, self.ATR_LOOKBACK)

        # Average ATR over longer period
        avg_atr = self._calculate_avg_atr(high, low, close)

        if avg_atr == 0:
            return 1.0

        # Ratio of average to current (higher = reduce size)
        volatility_ratio = avg_atr / current_atr

        # Clamp to reasonable range
        volatility_scalar = max(self.MIN_VOLATILITY_SCALAR, min(self.MAX_VOLATILITY_SCALAR, volatility_ratio))

        logger.debug(f"Volatility scalar: {volatility_scalar:.2f} (current ATR: {current_atr:.2f}, avg: {avg_atr:.2f})")

        return volatility_scalar

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate Average True Range."""
        if len(high) < period + 1:
            return np.mean(high - low)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        return np.mean(tr[-period:])

    def _calculate_avg_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate average ATR over longer lookback."""
        if len(high) < self.ATR_AVG_LOOKBACK:
            return self._calculate_atr(high, low, close, len(high) - 1)

        atr_values = []
        for i in range(self.ATR_LOOKBACK, min(len(high), self.ATR_AVG_LOOKBACK)):
            atr = self._calculate_atr(
                high[i-self.ATR_LOOKBACK:i+1],
                low[i-self.ATR_LOOKBACK:i+1],
                close[i-self.ATR_LOOKBACK:i+1],
                self.ATR_LOOKBACK
            )
            atr_values.append(atr)

        return np.mean(atr_values) if atr_values else 0

    def _check_drawdown_limits(self) -> Tuple[bool, str]:
        """
        Check if we're approaching drawdown limits.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Calculate current drawdowns
        total_dd = (self.peak_balance - self.account_balance) / self.peak_balance
        daily_dd = abs(self.daily_pnl) / self.peak_balance if self.daily_pnl < 0 else 0

        # Check total drawdown
        if total_dd >= self.TOTAL_DD_GUARDIAN:
            return False, f"Total DD {total_dd:.1%} >= guardian {self.TOTAL_DD_GUARDIAN:.1%}"

        # Check daily drawdown
        if daily_dd >= self.DAILY_DD_GUARDIAN:
            return False, f"Daily DD {daily_dd:.1%} >= guardian {self.DAILY_DD_GUARDIAN:.1%}"

        return True, "OK"

    def _calculate_dd_scalar(self) -> float:
        """
        Scale down size based on drawdown proximity.

        As we approach limits, reduce position size.
        """
        total_dd = (self.peak_balance - self.account_balance) / self.peak_balance
        daily_dd = abs(self.daily_pnl) / self.peak_balance if self.daily_pnl < 0 else 0

        # Calculate how much room we have
        total_room = self.TOTAL_DD_GUARDIAN - total_dd
        daily_room = self.DAILY_DD_GUARDIAN - daily_dd

        # Use the more restrictive
        min_room = min(total_room, daily_room)

        if min_room <= 0.01:  # Less than 1% room
            return 0.25  # 25% of normal size
        elif min_room <= 0.02:  # Less than 2% room
            return 0.5  # 50% of normal size
        elif min_room <= 0.03:  # Less than 3% room
            return 0.75  # 75% of normal size
        else:
            return 1.0  # Full size

    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking."""
        self.daily_pnl += pnl
        if pnl > 0:
            # New high water mark
            new_balance = self.account_balance + pnl
            if new_balance > self.peak_balance:
                self.peak_balance = new_balance

    def reset_daily(self):
        """Reset daily P&L (call at The5ers reset time: 5 PM EST)."""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")

    def update_balance(self, new_balance: float):
        """Update account balance."""
        self.account_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def get_stats(self) -> Dict:
        """Get current sizing statistics."""
        total_dd = (self.peak_balance - self.account_balance) / self.peak_balance
        daily_dd = abs(self.daily_pnl) / self.peak_balance if self.daily_pnl < 0 else 0

        return {
            "account_balance": self.account_balance,
            "peak_balance": self.peak_balance,
            "daily_pnl": self.daily_pnl,
            "total_dd_pct": total_dd,
            "daily_dd_pct": daily_dd,
            "total_dd_room": self.TOTAL_DD_GUARDIAN - total_dd,
            "daily_dd_room": self.DAILY_DD_GUARDIAN - daily_dd,
            "dd_scalar": self._calculate_dd_scalar()
        }
