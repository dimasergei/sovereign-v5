"""
Volatility Regime Filter.

CRITICAL: Block trades during extreme volatility conditions.
High volatility = wider stops = bigger losses when wrong.
Better to sit out until volatility normalizes.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class VolatilityState:
    """Current volatility regime state."""
    regime: str  # 'extreme_high', 'high', 'normal', 'low', 'extreme_low'
    can_trade: bool
    vol_percentile: float
    current_vol: float
    position_scalar: float  # 0.0 to 1.0


class VolatilityFilter:
    """
    Block trades during extreme volatility conditions.

    High volatility = wider stops = bigger losses when wrong.
    Better to sit out until volatility normalizes.

    Regimes:
    - extreme_high (>90th percentile): NO TRADING
    - high (75-90th): Trade with reduced size
    - normal (25-75th): Normal trading
    - low (10-25th): Normal trading
    - extreme_low (<10th): Normal (often precedes moves)
    """

    def __init__(self, lookback: int = 100, vol_window: int = 20):
        """
        Initialize volatility filter.

        Args:
            lookback: Bars for historical volatility distribution
            vol_window: Window for current volatility calculation
        """
        self.lookback = lookback
        self.vol_window = vol_window

        # Thresholds
        self.extreme_high_pct = 90
        self.high_pct = 75
        self.low_pct = 25
        self.extreme_low_pct = 10

    def analyze(self, df: pd.DataFrame) -> VolatilityState:
        """
        Analyze current volatility regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            VolatilityState with regime and trading permission
        """
        close = df['close'].values

        if len(close) < self.lookback:
            return VolatilityState(
                regime='normal',
                can_trade=True,
                vol_percentile=50.0,
                current_vol=0.0,
                position_scalar=1.0
            )

        # Calculate log returns
        returns = np.diff(np.log(close))

        if len(returns) < self.vol_window:
            return VolatilityState(
                regime='normal',
                can_trade=True,
                vol_percentile=50.0,
                current_vol=0.0,
                position_scalar=1.0
            )

        # Current volatility (annualized)
        current_vol = np.std(returns[-self.vol_window:]) * np.sqrt(252) * 100

        # Historical volatility distribution
        hist_vols = []
        step = max(1, self.vol_window // 4)

        for i in range(self.vol_window, len(returns), step):
            vol = np.std(returns[i - self.vol_window:i]) * np.sqrt(252) * 100
            hist_vols.append(vol)

        if not hist_vols:
            return VolatilityState(
                regime='normal',
                can_trade=True,
                vol_percentile=50.0,
                current_vol=current_vol,
                position_scalar=1.0
            )

        # Calculate percentile of current volatility
        vol_percentile = (np.sum(np.array(hist_vols) < current_vol) / len(hist_vols)) * 100

        # Determine regime
        if vol_percentile > self.extreme_high_pct:
            regime = 'extreme_high'
            can_trade = False  # NO TRADING in top 10% volatility
            position_scalar = 0.0
            logger.warning(
                f"VOLATILITY BLOCK: {vol_percentile:.1f}th percentile "
                f"(vol={current_vol:.2f}%)"
            )
        elif vol_percentile > self.high_pct:
            regime = 'high'
            can_trade = True  # Trade with reduced size
            position_scalar = 0.5
        elif vol_percentile < self.extreme_low_pct:
            regime = 'extreme_low'
            can_trade = True  # Low vol often precedes moves
            position_scalar = 1.0
        elif vol_percentile < self.low_pct:
            regime = 'low'
            can_trade = True
            position_scalar = 1.0
        else:
            regime = 'normal'
            can_trade = True
            position_scalar = self._calculate_position_scalar(vol_percentile)

        return VolatilityState(
            regime=regime,
            can_trade=can_trade,
            vol_percentile=vol_percentile,
            current_vol=current_vol,
            position_scalar=position_scalar
        )

    def _calculate_position_scalar(self, vol_percentile: float) -> float:
        """
        Scale position size based on volatility percentile.

        Higher volatility = smaller positions.
        """
        if vol_percentile > 90:
            return 0.0  # No trade
        elif vol_percentile > 75:
            return 0.5  # Half size
        elif vol_percentile > 60:
            return 0.75  # 75% size
        else:
            return 1.0  # Full size

    def get_volatility_adjusted_stop(
        self,
        base_stop_atr: float,
        vol_state: VolatilityState
    ) -> float:
        """
        Adjust stop loss multiplier based on volatility.

        In high volatility, widen stops to avoid noise.

        Args:
            base_stop_atr: Base stop distance in ATR
            vol_state: Current volatility state

        Returns:
            Adjusted stop ATR multiplier
        """
        if vol_state.regime == 'extreme_high':
            return base_stop_atr * 1.5
        elif vol_state.regime == 'high':
            return base_stop_atr * 1.3
        elif vol_state.regime == 'low':
            return base_stop_atr * 0.9  # Tighter in low vol
        else:
            return base_stop_atr


def analyze_volatility_trend(df: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> str:
    """
    Analyze if volatility is expanding or contracting.

    Returns: 'expanding', 'contracting', or 'stable'
    """
    close = df['close'].values

    if len(close) < long_window + 5:
        return 'stable'

    returns = np.diff(np.log(close))

    short_vol = np.std(returns[-short_window:])
    long_vol = np.std(returns[-long_window:])

    ratio = short_vol / long_vol if long_vol > 0 else 1.0

    if ratio > 1.3:
        return 'expanding'
    elif ratio < 0.7:
        return 'contracting'
    else:
        return 'stable'
