"""
Position Sizer - Kelly-based sizing with regime adjustment.

NEVER risk more than the guardian allows, regardless of Kelly calculation.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging


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
    Adaptive position sizing using Kelly criterion with safety caps.
    """

    def __init__(
        self,
        max_risk_pct: float = 1.0,  # Max 1% per trade (GFT allows 2%)
        min_risk_pct: float = 0.25,
        kelly_fraction: float = 0.25,  # Use 1/4 Kelly for safety
    ):
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.kelly_fraction = kelly_fraction

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
        regime: str,
        trend_strength: float = 0.5
    ) -> PositionSize:
        """
        Calculate safe position size.

        Args:
            account_balance: Current account balance
            current_drawdown_pct: Current drawdown from peak
            max_drawdown_pct: Maximum allowed drawdown
            stop_loss_pct: Stop loss distance as percentage
            signal_confidence: Signal confidence 0-1
            regime: Market regime
            trend_strength: Trend strength 0-1

        Returns:
            PositionSize with all sizing details
        """
        # Calculate Kelly-optimal fraction
        kelly = self._calculate_kelly()

        # Apply fractional Kelly for safety
        kelly_risk_pct = kelly * self.kelly_fraction * 100

        # Regime adjustment
        regime_adj = self._get_regime_adjustment(regime, trend_strength)
        adjusted_risk_pct = kelly_risk_pct * regime_adj

        # Confidence adjustment
        confidence_adj = 0.5 + (signal_confidence * 0.5)  # 0.5 to 1.0
        adjusted_risk_pct *= confidence_adj

        # CRITICAL: Drawdown-based reduction
        # As we approach max DD, reduce risk exponentially
        dd_room = max_drawdown_pct - current_drawdown_pct
        guardian_buffer = 1.5  # Stay 1.5% away from limit

        if dd_room <= guardian_buffer:
            # Too close to limit - minimal risk
            adjusted_risk_pct = self.min_risk_pct * 0.5
            reason = "guardian_proximity"
        elif dd_room <= 3.0:
            # Getting close - reduce proportionally
            reduction = (3.0 - dd_room) / (3.0 - guardian_buffer)
            adjusted_risk_pct *= (1 - reduction * 0.7)
            reason = "drawdown_reduction"
        else:
            reason = "normal"

        # Cap at max risk
        final_risk_pct = np.clip(adjusted_risk_pct, self.min_risk_pct, self.max_risk_pct)

        # Calculate actual size
        risk_amount = account_balance * (final_risk_pct / 100)

        if stop_loss_pct > 0:
            position_size = risk_amount / (stop_loss_pct / 100)
        else:
            position_size = 0

        return PositionSize(
            size=position_size,
            risk_amount=risk_amount,
            risk_pct=final_risk_pct,
            kelly_fraction=kelly,
            regime_adjustment=regime_adj,
            reason=reason
        )

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

    def _get_regime_adjustment(self, regime: str, trend_strength: float) -> float:
        """Get regime-based sizing adjustment."""
        adjustments = {
            "strong_trending": 1.0 + (trend_strength * 0.2),  # Up to 1.2x in strong trends
            "mild_trending": 0.9,
            "trending_up": 1.0 + (trend_strength * 0.2),
            "trending_down": 1.0 + (trend_strength * 0.2),
            "random_walk": 0.7,
            "unknown": 0.7,
            "mild_mean_reversion": 0.8,
            "strong_mean_reversion": 0.6,  # Smaller size in choppy markets
            "mean_reverting": 0.7,
        }
        return adjustments.get(regime, 0.7)

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
