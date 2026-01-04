# crypto/liquidity_hunter.py
"""
Liquidity Hunt Detection for Crypto Markets.

Crypto regularly sweeps stops before reversing (liquidity grabs).
This module detects these patterns for counter-trade opportunities.

Pattern: Price spikes through key level, immediately reverses.
- Wick > 60% of candle range
- Close back inside previous range
- Volume spike > 2x average
- Occurs at known S/R level
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LiquidityGrabType(Enum):
    """Type of liquidity grab detected."""
    STOP_HUNT_LOWS = "stop_hunt_lows"      # Swept lows then reversed up
    STOP_HUNT_HIGHS = "stop_hunt_highs"    # Swept highs then reversed down
    NONE = "none"


@dataclass
class KeyLevel:
    """Key support/resistance level."""
    price: float
    level_type: str  # "support" or "resistance"
    strength: int    # Number of touches
    last_tested: int  # Bars ago


@dataclass
class LiquidityGrab:
    """Detected liquidity grab event."""
    grab_type: LiquidityGrabType
    level_price: float
    wick_low: float
    wick_high: float
    close_price: float
    volume_ratio: float
    wick_ratio: float
    confidence: float
    suggested_entry: float
    suggested_stop: float
    suggested_target: float
    reason: str


class LiquidityHuntDetector:
    """
    Detects stop hunts / liquidity grabs common in crypto.

    Key insight: Market makers and whales sweep liquidity pools
    (clusters of stop losses) before reversing. We can trade WITH
    this pattern after confirmation.
    """

    # Detection thresholds
    MIN_WICK_RATIO = 0.60          # Wick must be 60%+ of total range
    MIN_VOLUME_RATIO = 1.5         # Volume must be 1.5x+ average
    IDEAL_VOLUME_RATIO = 2.0       # Ideal is 2x+ volume spike
    LEVEL_LOOKBACK = 50            # Bars to look back for S/R levels
    LEVEL_TOLERANCE_PCT = 0.002    # 0.2% tolerance for level touches
    MIN_LEVEL_TOUCHES = 2          # Minimum touches to be valid level

    def __init__(self):
        self.detected_grabs: List[LiquidityGrab] = []
        self.key_levels: List[KeyLevel] = []

    def detect_liquidity_grab(
        self,
        df: pd.DataFrame,
        key_levels: Optional[List[KeyLevel]] = None
    ) -> Optional[LiquidityGrab]:
        """
        Detect if current candle is a liquidity grab.

        Args:
            df: OHLCV DataFrame (current candle is last row)
            key_levels: Pre-calculated key levels (optional)

        Returns:
            LiquidityGrab if detected, None otherwise
        """
        if len(df) < self.LEVEL_LOOKBACK:
            return None

        # Get current candle
        current = df.iloc[-1]
        open_price = current['open']
        high = current['high']
        low = current['low']
        close = current['close']
        volume = current.get('volume', 0)

        # Calculate candle metrics
        total_range = high - low
        if total_range == 0:
            return None

        body = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low

        # Calculate volume ratio
        avg_volume = df['volume'].iloc[-20:-1].mean() if 'volume' in df.columns else 0
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Find key levels if not provided
        if key_levels is None:
            key_levels = self._find_key_levels(df)

        # Check for stop hunt at lows (bullish reversal)
        lower_wick_ratio = lower_wick / total_range
        if lower_wick_ratio >= self.MIN_WICK_RATIO:
            # Check if swept a support level
            swept_level = self._check_level_sweep(low, key_levels, "support")
            if swept_level and close > swept_level.price:
                if volume_ratio >= self.MIN_VOLUME_RATIO:
                    return self._create_grab(
                        grab_type=LiquidityGrabType.STOP_HUNT_LOWS,
                        level=swept_level,
                        candle=current,
                        wick_ratio=lower_wick_ratio,
                        volume_ratio=volume_ratio
                    )

        # Check for stop hunt at highs (bearish reversal)
        upper_wick_ratio = upper_wick / total_range
        if upper_wick_ratio >= self.MIN_WICK_RATIO:
            # Check if swept a resistance level
            swept_level = self._check_level_sweep(high, key_levels, "resistance")
            if swept_level and close < swept_level.price:
                if volume_ratio >= self.MIN_VOLUME_RATIO:
                    return self._create_grab(
                        grab_type=LiquidityGrabType.STOP_HUNT_HIGHS,
                        level=swept_level,
                        candle=current,
                        wick_ratio=upper_wick_ratio,
                        volume_ratio=volume_ratio
                    )

        return None

    def _find_key_levels(self, df: pd.DataFrame) -> List[KeyLevel]:
        """
        Find key support and resistance levels.

        Uses swing high/low detection with touch counting.
        """
        levels = []
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        lookback = min(self.LEVEL_LOOKBACK, len(df) - 1)

        # Find swing highs (resistance)
        for i in range(5, lookback - 5):
            idx = -(lookback - i)
            if high[idx] > high[idx-1] and high[idx] > high[idx+1]:
                if high[idx] > high[idx-2] and high[idx] > high[idx+2]:
                    level_price = high[idx]
                    touches = self._count_touches(close, level_price)
                    if touches >= self.MIN_LEVEL_TOUCHES:
                        levels.append(KeyLevel(
                            price=level_price,
                            level_type="resistance",
                            strength=touches,
                            last_tested=lookback - i
                        ))

        # Find swing lows (support)
        for i in range(5, lookback - 5):
            idx = -(lookback - i)
            if low[idx] < low[idx-1] and low[idx] < low[idx+1]:
                if low[idx] < low[idx-2] and low[idx] < low[idx+2]:
                    level_price = low[idx]
                    touches = self._count_touches(close, level_price)
                    if touches >= self.MIN_LEVEL_TOUCHES:
                        levels.append(KeyLevel(
                            price=level_price,
                            level_type="support",
                            strength=touches,
                            last_tested=lookback - i
                        ))

        # Sort by strength
        levels.sort(key=lambda x: x.strength, reverse=True)

        self.key_levels = levels[:10]  # Keep top 10 levels
        return self.key_levels

    def _count_touches(self, prices: np.ndarray, level: float) -> int:
        """Count how many times price touched a level."""
        tolerance = level * self.LEVEL_TOLERANCE_PCT
        touches = 0

        for i in range(1, len(prices)):
            # Check if price crossed through level
            if (prices[i-1] < level - tolerance and prices[i] > level - tolerance) or \
               (prices[i-1] > level + tolerance and prices[i] < level + tolerance):
                touches += 1
            # Check if price bounced off level
            elif abs(prices[i] - level) < tolerance:
                if i < len(prices) - 1:
                    if (prices[i] < level and prices[i+1] > prices[i]) or \
                       (prices[i] > level and prices[i+1] < prices[i]):
                        touches += 1

        return touches

    def _check_level_sweep(
        self,
        price: float,
        levels: List[KeyLevel],
        level_type: str
    ) -> Optional[KeyLevel]:
        """Check if price swept through a key level."""
        tolerance = price * self.LEVEL_TOLERANCE_PCT * 2  # Slightly wider for sweep

        for level in levels:
            if level.level_type != level_type:
                continue

            if level_type == "support":
                # For support, wick low should be below level
                if price < level.price and abs(price - level.price) < tolerance * 5:
                    return level
            else:
                # For resistance, wick high should be above level
                if price > level.price and abs(price - level.price) < tolerance * 5:
                    return level

        return None

    def _create_grab(
        self,
        grab_type: LiquidityGrabType,
        level: KeyLevel,
        candle: pd.Series,
        wick_ratio: float,
        volume_ratio: float
    ) -> LiquidityGrab:
        """Create a LiquidityGrab object with entry suggestions."""
        close = candle['close']
        high = candle['high']
        low = candle['low']

        # Calculate confidence
        confidence = 0.5
        confidence += min(0.2, (wick_ratio - self.MIN_WICK_RATIO) * 2)
        confidence += min(0.2, (volume_ratio - self.MIN_VOLUME_RATIO) * 0.2)
        confidence += min(0.1, level.strength * 0.02)

        # Calculate entries based on grab type
        if grab_type == LiquidityGrabType.STOP_HUNT_LOWS:
            # Bullish reversal - buy after stop hunt
            suggested_entry = close  # Enter at close or slightly above
            suggested_stop = low * 0.998  # Stop below the wick
            atr_estimate = (high - low) * 2
            suggested_target = close + atr_estimate * 2  # 2:1 R:R minimum
            reason = f"Stop hunt at support {level.price:.2f} - wick ratio {wick_ratio:.1%}, volume {volume_ratio:.1f}x"
        else:
            # Bearish reversal - sell after stop hunt (but we don't short crypto!)
            suggested_entry = close
            suggested_stop = high * 1.002
            atr_estimate = (high - low) * 2
            suggested_target = close - atr_estimate * 2
            reason = f"Stop hunt at resistance {level.price:.2f} - wick ratio {wick_ratio:.1%}, volume {volume_ratio:.1f}x"

        grab = LiquidityGrab(
            grab_type=grab_type,
            level_price=level.price,
            wick_low=low,
            wick_high=high,
            close_price=close,
            volume_ratio=volume_ratio,
            wick_ratio=wick_ratio,
            confidence=confidence,
            suggested_entry=suggested_entry,
            suggested_stop=suggested_stop,
            suggested_target=suggested_target,
            reason=reason
        )

        self.detected_grabs.append(grab)
        logger.info(f"Liquidity grab detected: {grab.grab_type.value} at {level.price:.2f}")

        return grab

    def get_entry_after_grab(
        self,
        grab: LiquidityGrab,
        current_price: float
    ) -> Optional[dict]:
        """
        Get entry parameters after a liquidity grab confirms.

        For crypto, only return LONG entries (no shorting).
        """
        if grab.grab_type == LiquidityGrabType.STOP_HUNT_HIGHS:
            # We don't short crypto
            logger.debug("Stop hunt highs detected but not shorting crypto")
            return None

        if grab.grab_type == LiquidityGrabType.STOP_HUNT_LOWS:
            # Enter long after liquidity grab at support
            if current_price > grab.close_price:
                return {
                    "direction": "long",
                    "entry": current_price,
                    "stop_loss": grab.wick_low * 0.998,  # Below the wick
                    "take_profit": current_price + (current_price - grab.wick_low) * 2.5,
                    "confidence": grab.confidence,
                    "reason": f"liquidity_grab_reversal at {grab.level_price:.2f}"
                }

        return None

    def get_recent_grabs(self, last_n: int = 5) -> List[LiquidityGrab]:
        """Get most recent liquidity grabs."""
        return self.detected_grabs[-last_n:] if self.detected_grabs else []

    def clear_history(self):
        """Clear detection history."""
        self.detected_grabs = []
        self.key_levels = []
