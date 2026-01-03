# crypto/regime_detector.py
"""
Crypto Market Regime Detection.

Identifies market regime to select appropriate strategy.
Crypto switches between trending and ranging more violently than forex.

Regimes:
- TRENDING_VOLATILE: Strong trend with high volatility - trend follow with wide stops
- TRENDING_QUIET: Strong trend with low volatility - trend follow with tight stops
- RANGING_VOLATILE: No trend + high volatility - NO TRADE (worst regime)
- RANGING_QUIET: No trend + low volatility - mean reversion plays
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CryptoRegime(Enum):
    """Market regime classification."""
    TRENDING_VOLATILE = "trending_volatile"
    TRENDING_QUIET = "trending_quiet"
    RANGING_VOLATILE = "ranging_volatile"
    RANGING_QUIET = "ranging_quiet"
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    regime: CryptoRegime
    adx: float
    atr_percentile: float
    trend_direction: str  # "bullish", "bearish", "neutral"
    trend_strength: float  # 0 to 1
    volatility_state: str  # "high", "normal", "low"
    recommended_strategy: Optional[str]
    confidence: float
    should_trade: bool
    reason: str


class CryptoRegimeDetector:
    """
    Identifies market regime to select appropriate strategy.

    Key insight: Crypto regime shifts happen faster and more violently.
    Must detect regime BEFORE entering, not after getting chopped up.
    """

    # ADX thresholds for trend detection
    ADX_TRENDING_THRESHOLD = 25
    ADX_STRONG_TREND_THRESHOLD = 40
    ADX_NO_TREND_THRESHOLD = 20

    # ATR percentile thresholds for volatility
    ATR_HIGH_VOLATILITY_PERCENTILE = 70
    ATR_LOW_VOLATILITY_PERCENTILE = 30

    # Lookback periods
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    ATR_PERCENTILE_LOOKBACK = 100
    EMA_FAST = 20
    EMA_SLOW = 50

    def __init__(self):
        self.regime_history: list = []
        self.last_regime: Optional[CryptoRegime] = None

    def detect_regime(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Detect current market regime from OHLCV data.

        Args:
            df: DataFrame with OHLCV data (minimum 100 bars recommended)

        Returns:
            RegimeAnalysis with full regime breakdown
        """
        if len(df) < self.ATR_PERCENTILE_LOOKBACK:
            return RegimeAnalysis(
                regime=CryptoRegime.UNKNOWN,
                adx=0,
                atr_percentile=50,
                trend_direction="neutral",
                trend_strength=0,
                volatility_state="normal",
                recommended_strategy=None,
                confidence=0,
                should_trade=False,
                reason="Insufficient data for regime detection"
            )

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Calculate ADX for trend strength
        adx = self._calculate_adx(high, low, close, self.ADX_PERIOD)

        # Calculate ATR and its percentile for volatility
        atr = self._calculate_atr(high, low, close, self.ATR_PERIOD)
        atr_history = self._calculate_atr_history(high, low, close, self.ATR_PERIOD)
        atr_percentile = self._calculate_percentile(atr, atr_history)

        # Calculate trend direction using EMAs
        ema_fast = self._ema(close, self.EMA_FAST)
        ema_slow = self._ema(close, self.EMA_SLOW)
        ema_slow_prev = self._ema(close[:-10], self.EMA_SLOW) if len(close) > 60 else ema_slow
        ema_slope = (ema_slow - ema_slow_prev) / ema_slow_prev * 100

        # Trend direction
        if close[-1] > ema_slow and ema_slope > 0.1:
            trend_direction = "bullish"
            trend_strength = min(1.0, abs(ema_slope) / 2)
        elif close[-1] < ema_slow and ema_slope < -0.1:
            trend_direction = "bearish"
            trend_strength = min(1.0, abs(ema_slope) / 2)
        else:
            trend_direction = "neutral"
            trend_strength = 0

        # Volatility state
        if atr_percentile >= self.ATR_HIGH_VOLATILITY_PERCENTILE:
            volatility_state = "high"
        elif atr_percentile <= self.ATR_LOW_VOLATILITY_PERCENTILE:
            volatility_state = "low"
        else:
            volatility_state = "normal"

        # Classify regime
        regime = self._classify_regime(adx, atr_percentile)

        # Get strategy recommendation
        recommended_strategy, should_trade, reason = self._get_strategy_recommendation(
            regime, trend_direction, adx, atr_percentile
        )

        # Calculate confidence
        confidence = self._calculate_confidence(adx, atr_percentile, regime)

        analysis = RegimeAnalysis(
            regime=regime,
            adx=adx,
            atr_percentile=atr_percentile,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            recommended_strategy=recommended_strategy,
            confidence=confidence,
            should_trade=should_trade,
            reason=reason
        )

        # Track regime changes
        if self.last_regime != regime:
            logger.info(f"Regime change: {self.last_regime} -> {regime}")
            self.regime_history.append((regime, confidence))
            self.last_regime = regime

        return analysis

    def _classify_regime(self, adx: float, atr_percentile: float) -> CryptoRegime:
        """Classify market regime based on ADX and ATR percentile."""
        is_trending = adx >= self.ADX_TRENDING_THRESHOLD
        is_volatile = atr_percentile >= self.ATR_HIGH_VOLATILITY_PERCENTILE
        is_quiet = atr_percentile <= self.ATR_LOW_VOLATILITY_PERCENTILE

        if is_trending and is_volatile:
            return CryptoRegime.TRENDING_VOLATILE
        elif is_trending and is_quiet:
            return CryptoRegime.TRENDING_QUIET
        elif not is_trending and is_volatile:
            return CryptoRegime.RANGING_VOLATILE
        elif not is_trending and is_quiet:
            return CryptoRegime.RANGING_QUIET
        else:
            # Normal volatility - determine by trend
            if is_trending:
                return CryptoRegime.TRENDING_QUIET
            else:
                return CryptoRegime.RANGING_QUIET

    def _get_strategy_recommendation(
        self,
        regime: CryptoRegime,
        trend_direction: str,
        adx: float,
        atr_percentile: float
    ) -> Tuple[Optional[str], bool, str]:
        """
        Get strategy recommendation for regime.

        Returns:
            Tuple of (strategy_name, should_trade, reason)
        """
        strategies = {
            CryptoRegime.TRENDING_VOLATILE: (
                "breakout_continuation",
                True,
                f"Strong trend (ADX={adx:.1f}) with high volatility - ride the momentum"
            ),
            CryptoRegime.TRENDING_QUIET: (
                "pullback_entry",
                True,
                f"Strong trend (ADX={adx:.1f}) with low volatility - enter on pullbacks"
            ),
            CryptoRegime.RANGING_VOLATILE: (
                None,
                False,
                f"No trend (ADX={adx:.1f}) + high volatility (ATR%={atr_percentile:.0f}) - AVOID"
            ),
            CryptoRegime.RANGING_QUIET: (
                "range_fade",
                True,
                f"Range-bound with low volatility - fade extremes"
            ),
            CryptoRegime.UNKNOWN: (
                None,
                False,
                "Regime unknown - insufficient data"
            )
        }

        return strategies.get(regime, (None, False, "Unknown regime"))

    def _calculate_confidence(
        self,
        adx: float,
        atr_percentile: float,
        regime: CryptoRegime
    ) -> float:
        """Calculate confidence in regime classification."""
        # Higher ADX = more confident in trend classification
        adx_confidence = min(1.0, adx / 50)

        # Extreme ATR percentiles = more confident in volatility classification
        if atr_percentile >= 80 or atr_percentile <= 20:
            vol_confidence = 0.9
        elif atr_percentile >= 70 or atr_percentile <= 30:
            vol_confidence = 0.7
        else:
            vol_confidence = 0.5

        # Weight differently by regime
        if regime in [CryptoRegime.TRENDING_VOLATILE, CryptoRegime.TRENDING_QUIET]:
            return adx_confidence * 0.7 + vol_confidence * 0.3
        else:
            return vol_confidence * 0.7 + adx_confidence * 0.3

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate Average Directional Index (ADX)."""
        if len(high) < period + 1:
            return 0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Directional Movement
        plus_dm = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0),
            0
        )
        minus_dm = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0),
            0
        )

        # Smoothed averages
        atr = self._wilder_smooth(tr, period)
        plus_di = 100 * self._wilder_smooth(plus_dm, period) / atr if atr > 0 else 0
        minus_di = 100 * self._wilder_smooth(minus_dm, period) / atr if atr > 0 else 0

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0

        return dx

    def _wilder_smooth(self, data: np.ndarray, period: int) -> float:
        """Wilder's smoothing method."""
        if len(data) < period:
            return np.mean(data) if len(data) > 0 else 0

        # Initial SMA
        result = np.mean(data[:period])

        # Wilder smoothing
        for i in range(period, len(data)):
            result = (result * (period - 1) + data[i]) / period

        return result

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate Average True Range."""
        if len(high) < period + 1:
            return np.mean(high - low) if len(high) > 0 else 0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        return np.mean(tr[-period:])

    def _calculate_atr_history(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate rolling ATR for percentile calculation."""
        if len(high) < period + 1:
            return np.array([self._calculate_atr(high, low, close, period)])

        atr_values = []
        for i in range(period, len(high)):
            atr = self._calculate_atr(
                high[i-period:i+1],
                low[i-period:i+1],
                close[i-period:i+1],
                period
            )
            atr_values.append(atr)

        return np.array(atr_values[-self.ATR_PERCENTILE_LOOKBACK:])

    def _calculate_percentile(self, value: float, history: np.ndarray) -> float:
        """Calculate percentile of value in history."""
        if len(history) == 0:
            return 50.0
        return (np.sum(history < value) / len(history)) * 100

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0

        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')[-1]

    def get_regime_stats(self) -> Dict:
        """Get statistics on regime history."""
        if not self.regime_history:
            return {"total": 0, "by_regime": {}}

        by_regime = {}
        for regime, conf in self.regime_history:
            if regime.value not in by_regime:
                by_regime[regime.value] = {"count": 0, "avg_confidence": 0}
            by_regime[regime.value]["count"] += 1
            by_regime[regime.value]["avg_confidence"] += conf

        for regime in by_regime:
            by_regime[regime]["avg_confidence"] /= by_regime[regime]["count"]

        return {
            "total": len(self.regime_history),
            "by_regime": by_regime,
            "current": self.last_regime.value if self.last_regime else None
        }
