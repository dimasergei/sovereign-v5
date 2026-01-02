# core/position_sizer.py
"""
Position Sizer for High-Frequency Multi-Alpha Trading.

Key insight: Many small positions, not few large ones.
Max risk per trade: 0.5% (allows for more positions simultaneously)

Target: 4-6 simultaneous positions with 0.5% risk each
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

from config.trading_params import get_params


logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing result."""
    size: float
    risk_amount: float
    risk_pct: float
    kelly_fraction: float
    regime_adjustment: float
    reason: str


class PositionSizer:
    """
    Position sizing for high-frequency multi-alpha trading.

    Key insight: Many small positions, not few large ones.
    This allows us to take more trades and let the edge compound.
    """

    def __init__(
        self,
        base_risk_pct: float = 0.5,  # 0.5% per trade (was 1.0%)
        max_risk_pct: float = 0.8,   # Max 0.8% even with high confidence
        min_risk_pct: float = 0.2,   # Min 0.2%
        max_positions: int = 6,      # Allow up to 6 simultaneous positions
        max_total_risk: float = 3.0  # Max 3% total exposure
    ):
        """
        Initialize position sizer.

        Args:
            base_risk_pct: Base risk per trade (0.5%)
            max_risk_pct: Maximum risk per trade (0.8%)
            min_risk_pct: Minimum risk per trade (0.2%)
            max_positions: Maximum simultaneous positions
            max_total_risk: Maximum total risk exposure
        """
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.max_positions = max_positions
        self.max_total_risk = max_total_risk

        # Performance tracking for Kelly calculation
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0.0
        self.total_losses = 0.0

    def calculate(
        self,
        account_balance: float,
        current_drawdown_pct: float,
        max_drawdown_pct: float,
        stop_loss_pct: float,
        signal_confidence: float,
        signal_position_size: float = 1.0,  # From multi-alpha (0-1)
        open_positions_count: int = 0,
        current_total_risk: float = 0.0,
        regime: str = "multi_alpha",
        trend_strength: float = 0.5,
        symbol: str = None
    ) -> PositionSize:
        """
        Calculate safe position size.

        Args:
            account_balance: Current account balance
            current_drawdown_pct: Current drawdown from peak
            max_drawdown_pct: Maximum allowed drawdown
            stop_loss_pct: Stop loss distance as percentage
            signal_confidence: Signal confidence 0-1
            signal_position_size: Position size scalar from multi-alpha (0-1)
            open_positions_count: Number of currently open positions
            current_total_risk: Current total risk exposure
            regime: Market regime (ignored in multi-alpha mode)
            trend_strength: Trend strength (ignored in multi-alpha mode)
            symbol: Trading symbol for asset-specific params

        Returns:
            PositionSize with all sizing details
        """
        # Get asset-specific parameters
        params = get_params(symbol) if symbol else get_params()

        # Position limit check
        if open_positions_count >= self.max_positions:
            return PositionSize(
                size=0,
                risk_amount=0,
                risk_pct=0,
                kelly_fraction=0,
                regime_adjustment=0,
                reason="max_positions_reached"
            )

        # Total risk limit check
        if current_total_risk >= self.max_total_risk:
            return PositionSize(
                size=0,
                risk_amount=0,
                risk_pct=0,
                kelly_fraction=0,
                regime_adjustment=0,
                reason="max_total_risk_reached"
            )

        # Guardian check - DD proximity
        dd_room = max_drawdown_pct - current_drawdown_pct
        guardian_buffer = params.get('guardian_buffer', 1.5)

        if dd_room <= guardian_buffer:
            return PositionSize(
                size=0,
                risk_amount=0,
                risk_pct=0,
                kelly_fraction=0,
                regime_adjustment=0,
                reason="guardian_proximity"
            )

        # Base risk from params
        risk_pct = params.get('base_risk_pct', self.base_risk_pct)

        # Adjust for DD proximity
        if dd_room < 2.5:
            risk_pct *= 0.5  # Half risk when close to guardian
        elif dd_room < 3.5:
            risk_pct *= 0.75  # 75% risk when approaching guardian

        # Apply signal confidence and position size from multi-alpha
        risk_pct *= signal_confidence * signal_position_size

        # Cap at limits
        max_risk = params.get('max_risk_pct', self.max_risk_pct)
        min_risk = params.get('min_risk_pct', self.min_risk_pct)
        risk_pct = np.clip(risk_pct, min_risk, max_risk)

        # Kelly adjustment (if we have enough data)
        kelly = self._calculate_kelly()
        if kelly > 0:
            # Use Kelly as a guide, but don't exceed our base sizing
            kelly_risk = kelly * 0.25 * 100  # Quarter Kelly
            risk_pct = min(risk_pct, kelly_risk) if kelly_risk > 0 else risk_pct

        # Calculate dollar risk
        risk_amount = account_balance * (risk_pct / 100)

        # Calculate position size
        if stop_loss_pct > 0:
            position_size = risk_amount / (stop_loss_pct / 100)
        else:
            position_size = 0

        return PositionSize(
            size=position_size,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            kelly_fraction=kelly,
            regime_adjustment=1.0,  # Not used in multi-alpha mode
            reason="approved"
        )

    def calculate_simple(
        self,
        account_balance: float,
        current_dd_pct: float,
        max_dd_pct: float,
        stop_distance_pct: float,
        signal_confidence: float,
        signal_position_size: float,
        open_positions_count: int = 0
    ) -> dict:
        """
        Simplified calculation for backtest compatibility.

        Args:
            account_balance: Current account balance
            current_dd_pct: Current drawdown percentage
            max_dd_pct: Maximum drawdown limit
            stop_distance_pct: Stop loss distance as percentage
            signal_confidence: Signal confidence 0-1
            signal_position_size: Position size from multi-alpha (0-1)
            open_positions_count: Number of open positions

        Returns:
            Dict with size, risk_amount, risk_pct, reason
        """
        # Guardian check
        dd_room = max_dd_pct - current_dd_pct
        if dd_room < 1.5:  # Less than 1.5% room
            return {"size": 0, "risk_amount": 0, "risk_pct": 0, "reason": "guardian_proximity"}

        # Position limit check
        if open_positions_count >= self.max_positions:
            return {"size": 0, "risk_amount": 0, "risk_pct": 0, "reason": "max_positions"}

        # Base risk adjusted for DD proximity
        if dd_room < 2.5:
            risk_pct = self.base_risk_pct * 0.5
        elif dd_room < 3.5:
            risk_pct = self.base_risk_pct * 0.75
        else:
            risk_pct = self.base_risk_pct

        # Apply signal confidence and position size
        risk_pct *= signal_confidence * signal_position_size

        # Cap at limits
        risk_pct = np.clip(risk_pct, self.min_risk_pct, self.max_risk_pct)

        # Calculate dollar risk
        risk_amount = account_balance * (risk_pct / 100)

        # Calculate position size
        if stop_distance_pct > 0:
            size = risk_amount / (stop_distance_pct / 100)
        else:
            size = 0

        return {
            "size": size,
            "risk_amount": risk_amount,
            "risk_pct": risk_pct,
            "reason": "approved"
        }

    def _calculate_kelly(self) -> float:
        """Calculate Kelly fraction from historical performance."""
        total_trades = self.win_count + self.loss_count

        if total_trades < 20:
            # Not enough data - use conservative default
            return 0.1

        win_rate = self.win_count / total_trades

        if self.loss_count > 0 and self.win_count > 0:
            avg_win = self.total_wins / self.win_count
            avg_loss = self.total_losses / self.loss_count

            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
                return max(0.05, min(0.5, kelly))  # Cap between 5% and 50%

        return 0.1

    def update_performance(self, pnl: float):
        """Update performance stats for Kelly calculation."""
        if pnl > 0:
            self.win_count += 1
            self.total_wins += pnl
        else:
            self.loss_count += 1
            self.total_losses += abs(pnl)

    def reset_performance(self):
        """Reset performance tracking."""
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0.0
        self.total_losses = 0.0

    def get_stats(self) -> Dict:
        """Get position sizing statistics."""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": self.total_wins / self.win_count if self.win_count > 0 else 0,
            "avg_loss": self.total_losses / self.loss_count if self.loss_count > 0 else 0,
            "kelly": self._calculate_kelly()
        }
