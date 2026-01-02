"""
Trend Filter Module - Prevents counter-trend trades.

This is the #1 cause of losses. The system was shorting 65 times
in a bull market because there was no trend filter.

RULE: In trending regimes, ONLY trade with the trend.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Market trend direction."""
    STRONG_UP = "strong_up"
    MILD_UP = "mild_up"
    NEUTRAL = "neutral"
    MILD_DOWN = "mild_down"
    STRONG_DOWN = "strong_down"


@dataclass
class TrendState:
    """Complete trend state for filtering decisions."""
    direction: TrendDirection
    strength: float  # 0-1
    higher_tf_aligned: bool
    price_vs_ma50: float  # % above/below
    price_vs_ma200: float  # % above/below
    slope_ma50: float  # Slope of MA50
    hurst: float  # Hurst exponent
    is_trending: bool  # H > 0.55


class TrendFilter:
    """
    Multi-timeframe trend filter to prevent counter-trend trades.

    CRITICAL: This filter MUST be applied before any trade signal
    is acted upon. It is the primary defense against drawdowns.
    """

    def __init__(self):
        self.ma_short = 20
        self.ma_medium = 50
        self.ma_long = 200
        self.slope_period = 10

    def analyze(self, df: pd.DataFrame) -> TrendState:
        """
        Analyze trend state from price data.

        Args:
            df: DataFrame with OHLCV data, minimum 250 bars

        Returns:
            TrendState with all trend information
        """
        close = df['close']

        # Calculate MAs
        ma20 = close.rolling(self.ma_short).mean()
        ma50 = close.rolling(self.ma_medium).mean()
        ma200 = close.rolling(self.ma_long).mean()

        current_price = close.iloc[-1]

        # Price position vs MAs
        ma50_val = ma50.iloc[-1]
        ma200_val = ma200.iloc[-1]

        if pd.isna(ma50_val) or ma50_val == 0:
            price_vs_ma50 = 0.0
        else:
            price_vs_ma50 = (current_price - ma50_val) / ma50_val * 100

        if pd.isna(ma200_val) or ma200_val == 0:
            price_vs_ma200 = price_vs_ma50
        else:
            price_vs_ma200 = (current_price - ma200_val) / ma200_val * 100

        # MA50 slope (trend direction)
        slope_start_idx = max(0, len(ma50) - self.slope_period)
        if slope_start_idx < len(ma50) and not pd.isna(ma50.iloc[slope_start_idx]) and ma50.iloc[slope_start_idx] != 0:
            ma50_slope = (ma50.iloc[-1] - ma50.iloc[slope_start_idx]) / ma50.iloc[slope_start_idx] * 100
        else:
            ma50_slope = 0.0

        # Hurst exponent for regime
        hurst = self._calculate_hurst(close.values[-100:] if len(close) >= 100 else close.values)
        is_trending = hurst > 0.55

        # Determine direction
        direction, strength = self._classify_direction(
            price_vs_ma50, price_vs_ma200, ma50_slope,
            ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else current_price,
            ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else current_price
        )

        # Higher timeframe alignment (MA20 > MA50 > MA200 for uptrend)
        ma20_val = ma20.iloc[-1]
        if not pd.isna(ma200_val) and not pd.isna(ma50_val) and not pd.isna(ma20_val):
            higher_tf_aligned = (
                (direction in [TrendDirection.STRONG_UP, TrendDirection.MILD_UP] and
                 ma20_val > ma50_val > ma200_val) or
                (direction in [TrendDirection.STRONG_DOWN, TrendDirection.MILD_DOWN] and
                 ma20_val < ma50_val < ma200_val)
            )
        else:
            if not pd.isna(ma50_val) and not pd.isna(ma20_val):
                higher_tf_aligned = (
                    ma20_val > ma50_val if direction in [TrendDirection.STRONG_UP, TrendDirection.MILD_UP]
                    else ma20_val < ma50_val
                )
            else:
                higher_tf_aligned = False

        return TrendState(
            direction=direction,
            strength=strength,
            higher_tf_aligned=higher_tf_aligned,
            price_vs_ma50=price_vs_ma50,
            price_vs_ma200=price_vs_ma200,
            slope_ma50=ma50_slope,
            hurst=hurst,
            is_trending=is_trending
        )

    def _classify_direction(
        self,
        price_vs_ma50: float,
        price_vs_ma200: float,
        ma_slope: float,
        ma20: float,
        ma50: float
    ) -> Tuple[TrendDirection, float]:
        """Classify trend direction and strength."""

        # Strong uptrend: price well above MAs, slope positive, MA stack bullish
        if price_vs_ma50 > 3 and price_vs_ma200 > 5 and ma_slope > 1:
            return TrendDirection.STRONG_UP, min(1.0, (price_vs_ma50 + ma_slope) / 10)

        # Mild uptrend: price above MA50, positive slope
        elif price_vs_ma50 > 0 and ma_slope > 0:
            return TrendDirection.MILD_UP, min(0.7, price_vs_ma50 / 5)

        # Strong downtrend
        elif price_vs_ma50 < -3 and price_vs_ma200 < -5 and ma_slope < -1:
            return TrendDirection.STRONG_DOWN, min(1.0, abs(price_vs_ma50 + ma_slope) / 10)

        # Mild downtrend
        elif price_vs_ma50 < 0 and ma_slope < 0:
            return TrendDirection.MILD_DOWN, min(0.7, abs(price_vs_ma50) / 5)

        else:
            return TrendDirection.NEUTRAL, 0.0

    def _calculate_hurst(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent for regime detection."""
        if len(prices) < 20:
            return 0.5

        # Remove any NaN or inf values
        prices = prices[np.isfinite(prices)]
        if len(prices) < 20:
            return 0.5

        n = len(prices)
        max_k = min(int(np.log2(n)) - 1, 8)

        rs_values = []
        ns = []

        # Calculate returns safely
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        returns = returns[np.isfinite(returns)]

        if len(returns) < 10:
            return 0.5

        for k in range(2, max_k + 1):
            subset_size = int(2 ** k)
            num_subsets = len(returns) // subset_size

            if num_subsets == 0:
                continue

            rs_list = []

            for i in range(num_subsets):
                subset = returns[i * subset_size:(i + 1) * subset_size]
                mean_adj = subset - np.mean(subset)
                cumsum = np.cumsum(mean_adj)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(subset, ddof=1)

                if S > 0:
                    rs_list.append(R / S)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                ns.append(subset_size)

        if len(rs_values) > 2:
            log_n = np.log(ns)
            log_rs = np.log(rs_values)
            H, _ = np.polyfit(log_n, log_rs, 1)
            return np.clip(H, 0, 1)

        return 0.5

    def filter_signal(
        self,
        signal_direction: str,  # "long", "short", "neutral"
        trend_state: TrendState,
        regime: str  # from calibrator
    ) -> Tuple[str, float, str]:
        """
        Filter a signal based on trend state.

        Args:
            signal_direction: Raw signal from generator
            trend_state: Current trend analysis
            regime: Market regime from calibrator

        Returns:
            Tuple of (filtered_signal, confidence_multiplier, reason)
        """
        # Neutral signals pass through
        if signal_direction == "neutral":
            return "neutral", 1.0, "no_signal"

        # In trending regimes, only allow trend-following trades
        if trend_state.is_trending or regime in ["mild_trending", "strong_trending", "trending_up", "trending_down"]:

            # LONG signal in uptrend - ALLOW with boost
            if signal_direction == "long" and trend_state.direction in [TrendDirection.STRONG_UP, TrendDirection.MILD_UP]:
                multiplier = 1.2 if trend_state.higher_tf_aligned else 1.0
                return "long", multiplier, "trend_aligned"

            # SHORT signal in downtrend - ALLOW with boost
            elif signal_direction == "short" and trend_state.direction in [TrendDirection.STRONG_DOWN, TrendDirection.MILD_DOWN]:
                multiplier = 1.2 if trend_state.higher_tf_aligned else 1.0
                return "short", multiplier, "trend_aligned"

            # LONG signal in downtrend - BLOCK
            elif signal_direction == "long" and trend_state.direction in [TrendDirection.STRONG_DOWN, TrendDirection.MILD_DOWN]:
                return "neutral", 0.0, "counter_trend_blocked"

            # SHORT signal in uptrend - BLOCK (THIS IS THE KEY FIX)
            elif signal_direction == "short" and trend_state.direction in [TrendDirection.STRONG_UP, TrendDirection.MILD_UP]:
                return "neutral", 0.0, "counter_trend_blocked"

            # Neutral trend - allow with reduced confidence
            else:
                return signal_direction, 0.7, "trend_unclear"

        # In mean-reverting regimes, allow both directions
        elif regime in ["mild_mean_reversion", "strong_mean_reversion", "mean_reverting"]:
            return signal_direction, 1.0, "mean_reversion_allowed"

        # Default: allow with standard confidence
        return signal_direction, 0.8, "default"
