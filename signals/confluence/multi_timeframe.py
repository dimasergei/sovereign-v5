"""
Multi-Timeframe Confluence Analyzer

Analyzes signal alignment across multiple timeframes to identify
high-probability trading opportunities. Confluence scoring provides
confidence based on agreement between timeframes.

Key concepts:
- Trend direction on each timeframe (M1, M5, M15, H1, H4, D1)
- Confluence score: 0-1 based on alignment
- Conflict detection: identifies when timeframes disagree
- Dominant timeframe: which timeframe is driving price action
- Signal quality: agreement-weighted confidence
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Trading timeframes in ascending order."""
    M1 = "M1"    # 1 minute
    M5 = "M5"    # 5 minutes
    M15 = "M15"  # 15 minutes
    M30 = "M30"  # 30 minutes
    H1 = "H1"    # 1 hour
    H4 = "H4"    # 4 hours
    D1 = "D1"    # 1 day
    W1 = "W1"    # 1 week

    @property
    def minutes(self) -> int:
        """Return timeframe in minutes."""
        mapping = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
        }
        return mapping[self.value]

    @property
    def weight(self) -> float:
        """Higher timeframes have more weight by default."""
        return np.log2(self.minutes + 1) / np.log2(10081)


@dataclass
class TimeframeTrend:
    """Trend analysis for a single timeframe."""
    timeframe: Timeframe
    direction: float  # -1 (bearish) to 1 (bullish)
    strength: float  # 0 to 1
    momentum: float  # -1 to 1 (accelerating/decelerating)

    # Technical indicators
    ma_cross: float = 0.0  # MA cross signal
    rsi: float = 50.0
    macd_hist: float = 0.0

    # Structure
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def trend_type(self) -> str:
        """Categorize the trend."""
        if abs(self.direction) < 0.2:
            return "sideways"
        elif self.direction > 0:
            if self.strength > 0.7:
                return "strong_uptrend"
            else:
                return "weak_uptrend"
        else:
            if self.strength > 0.7:
                return "strong_downtrend"
            else:
                return "weak_downtrend"


@dataclass
class TimeframeData:
    """OHLCV data for a specific timeframe."""
    timeframe: Timeframe
    data: pd.DataFrame
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class ConfluenceSignal:
    """Confluence-based trading signal."""
    direction: float  # -1 to 1
    confluence_score: float  # 0 to 1 (agreement between timeframes)
    confidence: float  # 0 to 1 (overall signal quality)

    # Individual timeframe analyses
    timeframe_trends: Dict[str, TimeframeTrend] = field(default_factory=dict)

    # Conflict analysis
    has_conflict: bool = False
    conflicting_timeframes: List[str] = field(default_factory=list)

    # Dominant timeframe
    dominant_timeframe: str = "H1"
    dominant_weight: float = 0.0

    # Signal characteristics
    signal_type: str = "none"  # "trend_following", "reversal", "breakout", "none"
    action: str = "neutral"  # "long", "short", "neutral"

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_position_scalar(self) -> float:
        """Calculate position size based on confluence."""
        # Higher confluence = larger position
        base_scalar = self.confluence_score * self.confidence

        # Reduce for conflicts
        if self.has_conflict:
            base_scalar *= 0.5

        return min(1.0, base_scalar)


class MultiTimeframeAnalyzer:
    """
    Analyzes price action across multiple timeframes to identify confluence.

    Confluence occurs when multiple timeframes agree on direction, increasing
    probability of successful trades. Conflicts (disagreement) should reduce
    position sizing or signal avoidance.

    Usage:
        analyzer = MultiTimeframeAnalyzer(
            timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        )

        # Add data for each timeframe
        analyzer.set_data(Timeframe.M15, df_m15)
        analyzer.set_data(Timeframe.H1, df_h1)
        analyzer.set_data(Timeframe.H4, df_h4)
        analyzer.set_data(Timeframe.D1, df_d1)

        # Get confluence signal
        signal = analyzer.analyze()

        if signal.confluence_score > 0.7:
            # High confluence - trade with confidence
            pass
    """

    DEFAULT_TIMEFRAMES = [
        Timeframe.M15,
        Timeframe.H1,
        Timeframe.H4,
        Timeframe.D1
    ]

    def __init__(
        self,
        timeframes: List[Timeframe] = None,
        ma_fast_period: int = 20,
        ma_slow_period: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        confluence_threshold: float = 0.6,
        conflict_threshold: float = 0.3
    ):
        """
        Initialize multi-timeframe analyzer.

        Args:
            timeframes: List of timeframes to analyze
            ma_fast_period: Fast moving average period
            ma_slow_period: Slow moving average period
            rsi_period: RSI period
            atr_period: ATR period
            confluence_threshold: Minimum score for confluence signal
            conflict_threshold: Direction difference indicating conflict
        """
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        self.ma_fast_period = ma_fast_period
        self.ma_slow_period = ma_slow_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.confluence_threshold = confluence_threshold
        self.conflict_threshold = conflict_threshold

        # Data storage
        self.data: Dict[Timeframe, TimeframeData] = {}
        self.trends: Dict[Timeframe, TimeframeTrend] = {}

        # Custom weights (override defaults)
        self.custom_weights: Dict[Timeframe, float] = {}

    def set_data(self, timeframe: Timeframe, df: pd.DataFrame):
        """
        Set OHLCV data for a timeframe.

        Args:
            timeframe: Timeframe enum
            df: DataFrame with columns: open, high, low, close, volume
        """
        # Validate columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.data[timeframe] = TimeframeData(
            timeframe=timeframe,
            data=df.copy(),
            last_update=datetime.now()
        )

    def set_weight(self, timeframe: Timeframe, weight: float):
        """Set custom weight for a timeframe."""
        self.custom_weights[timeframe] = weight

    def get_weight(self, timeframe: Timeframe) -> float:
        """Get weight for a timeframe."""
        if timeframe in self.custom_weights:
            return self.custom_weights[timeframe]
        return timeframe.weight

    def analyze(self) -> ConfluenceSignal:
        """
        Perform multi-timeframe confluence analysis.

        Returns:
            ConfluenceSignal with direction, confluence score, and confidence
        """
        if not self.data:
            return self._neutral_signal("No data available")

        # 1. Analyze each timeframe
        self.trends = {}
        for tf in self.timeframes:
            if tf in self.data:
                trend = self._analyze_timeframe(tf)
                self.trends[tf] = trend

        if not self.trends:
            return self._neutral_signal("No trends calculated")

        # 2. Calculate confluence
        confluence_score, weighted_direction = self._calculate_confluence()

        # 3. Detect conflicts
        has_conflict, conflicting = self._detect_conflicts()

        # 4. Identify dominant timeframe
        dominant_tf, dominant_weight = self._find_dominant_timeframe()

        # 5. Determine signal type
        signal_type = self._classify_signal_type()

        # 6. Calculate confidence
        confidence = self._calculate_confidence(confluence_score, has_conflict)

        # 7. Determine action
        if confluence_score < self.confluence_threshold:
            action = "neutral"
        elif abs(weighted_direction) < 0.2:
            action = "neutral"
        elif weighted_direction > 0:
            action = "long"
        else:
            action = "short"

        return ConfluenceSignal(
            direction=weighted_direction,
            confluence_score=confluence_score,
            confidence=confidence,
            timeframe_trends={tf.value: trend for tf, trend in self.trends.items()},
            has_conflict=has_conflict,
            conflicting_timeframes=conflicting,
            dominant_timeframe=dominant_tf.value if dominant_tf else "unknown",
            dominant_weight=dominant_weight,
            signal_type=signal_type,
            action=action,
            metadata={
                'timeframes_analyzed': len(self.trends),
                'total_weight': sum(self.get_weight(tf) for tf in self.trends.keys()),
            }
        )

    def _analyze_timeframe(self, timeframe: Timeframe) -> TimeframeTrend:
        """Analyze trend on a single timeframe."""
        tf_data = self.data[timeframe]
        df = tf_data.data.copy()

        if len(df) < self.ma_slow_period + 10:
            return TimeframeTrend(
                timeframe=timeframe,
                direction=0.0,
                strength=0.0,
                momentum=0.0
            )

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Calculate MAs
        ma_fast = self._sma(close, self.ma_fast_period)
        ma_slow = self._sma(close, self.ma_slow_period)

        # MA cross signal
        if ma_slow[-1] > 0:
            ma_cross = (ma_fast[-1] - ma_slow[-1]) / ma_slow[-1]
        else:
            ma_cross = 0.0

        # Calculate RSI
        rsi = self._calculate_rsi(close, self.rsi_period)

        # Calculate MACD histogram
        macd_hist = self._calculate_macd_hist(close)

        # Trend direction from multiple indicators
        direction = self._calculate_direction(close, ma_fast, ma_slow, rsi, macd_hist)

        # Trend strength (ADX-like)
        strength = self._calculate_strength(high, low, close)

        # Momentum (rate of change)
        momentum = self._calculate_momentum(close)

        # Market structure
        hh = self._is_higher_high(high)
        hl = self._is_higher_low(low)
        lh = self._is_lower_high(high)
        ll = self._is_lower_low(low)

        return TimeframeTrend(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            momentum=momentum,
            ma_cross=ma_cross,
            rsi=rsi,
            macd_hist=macd_hist,
            higher_high=hh,
            higher_low=hl,
            lower_high=lh,
            lower_low=ll
        )

    def _calculate_confluence(self) -> Tuple[float, float]:
        """
        Calculate confluence score and weighted direction.

        Returns:
            (confluence_score, weighted_direction)
        """
        if not self.trends:
            return 0.0, 0.0

        directions = []
        weights = []

        for tf, trend in self.trends.items():
            w = self.get_weight(tf) * trend.strength
            directions.append(trend.direction)
            weights.append(w)

        directions = np.array(directions)
        weights = np.array(weights)

        if weights.sum() == 0:
            return 0.0, 0.0

        # Weighted direction
        weighted_direction = np.sum(directions * weights) / weights.sum()

        # Confluence: how much do they agree?
        # Perfect confluence = all same sign and magnitude
        signs = np.sign(directions)
        dominant_sign = np.sign(weighted_direction)

        if dominant_sign == 0:
            confluence = 0.0
        else:
            # What percentage agree with dominant direction?
            agreeing_weight = weights[signs == dominant_sign].sum()
            total_weight = weights.sum()

            # How strongly do they agree?
            if total_weight > 0:
                confluence = agreeing_weight / total_weight
            else:
                confluence = 0.0

            # Penalize weak signals
            avg_strength = np.mean([t.strength for t in self.trends.values()])
            confluence *= (0.5 + 0.5 * avg_strength)

        return min(1.0, confluence), float(weighted_direction)

    def _detect_conflicts(self) -> Tuple[bool, List[str]]:
        """
        Detect conflicting timeframes.

        Returns:
            (has_conflict, list of conflicting timeframe names)
        """
        if len(self.trends) < 2:
            return False, []

        conflicting = []
        trends_list = list(self.trends.items())

        for i, (tf1, trend1) in enumerate(trends_list):
            for tf2, trend2 in trends_list[i+1:]:
                # Conflict if opposite directions with both having significant strength
                if (np.sign(trend1.direction) != np.sign(trend2.direction) and
                    trend1.strength > 0.3 and trend2.strength > 0.3 and
                    abs(trend1.direction) > 0.2 and abs(trend2.direction) > 0.2):

                    if tf1.value not in conflicting:
                        conflicting.append(tf1.value)
                    if tf2.value not in conflicting:
                        conflicting.append(tf2.value)

        return len(conflicting) > 0, conflicting

    def _find_dominant_timeframe(self) -> Tuple[Optional[Timeframe], float]:
        """
        Find the dominant (most influential) timeframe.

        Returns:
            (dominant timeframe, its effective weight)
        """
        if not self.trends:
            return None, 0.0

        max_weight = 0.0
        dominant = None

        for tf, trend in self.trends.items():
            # Effective weight = base weight * strength * abs(direction)
            effective_weight = self.get_weight(tf) * trend.strength * abs(trend.direction)

            if effective_weight > max_weight:
                max_weight = effective_weight
                dominant = tf

        return dominant, max_weight

    def _classify_signal_type(self) -> str:
        """Classify the type of trading signal."""
        if not self.trends:
            return "none"

        # Check for trend following (all timeframes aligned)
        directions = [t.direction for t in self.trends.values()]
        if all(d > 0.2 for d in directions) or all(d < -0.2 for d in directions):
            return "trend_following"

        # Check for potential reversal (lower TFs reversing vs higher)
        sorted_tfs = sorted(self.trends.items(), key=lambda x: x[0].minutes)

        if len(sorted_tfs) >= 2:
            lower_tf = sorted_tfs[0][1]
            higher_tf = sorted_tfs[-1][1]

            # Lower TF reversing against higher TF trend
            if (np.sign(lower_tf.direction) != np.sign(higher_tf.direction) and
                abs(lower_tf.direction) > 0.3):
                return "reversal"

        return "mixed"

    def _calculate_confidence(self, confluence: float, has_conflict: bool) -> float:
        """Calculate overall signal confidence."""
        if not self.trends:
            return 0.0

        # Base confidence from confluence
        confidence = confluence

        # Reduce for conflicts
        if has_conflict:
            confidence *= 0.6

        # Boost for strong individual trends
        avg_strength = np.mean([t.strength for t in self.trends.values()])
        confidence *= (0.7 + 0.3 * avg_strength)

        # Reduce if few timeframes
        if len(self.trends) < 3:
            confidence *= 0.8

        return min(1.0, confidence)

    # Technical indicator helpers

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        result = np.zeros_like(data)
        for i in range(len(data)):
            if i < period - 1:
                result[i] = np.mean(data[:i+1])
            else:
                result[i] = np.mean(data[i-period+1:i+1])
        return result

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd_hist(self, close: np.ndarray) -> float:
        """Calculate MACD histogram."""
        if len(close) < 26:
            return 0.0

        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(macd_line, 9)

        return float(macd_line[-1] - signal_line[-1])

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        result = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        result[0] = data[0]

        for i in range(1, len(data)):
            result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]

        return result

    def _calculate_direction(
        self,
        close: np.ndarray,
        ma_fast: np.ndarray,
        ma_slow: np.ndarray,
        rsi: float,
        macd_hist: float
    ) -> float:
        """Calculate overall trend direction."""
        signals = []

        # MA cross contribution
        if ma_slow[-1] > 0:
            ma_signal = (ma_fast[-1] - ma_slow[-1]) / ma_slow[-1] * 10
            signals.append(np.clip(ma_signal, -1, 1))

        # RSI contribution
        if rsi > 70:
            signals.append(-0.5 * (rsi - 70) / 30)  # Overbought = bearish
        elif rsi < 30:
            signals.append(0.5 * (30 - rsi) / 30)  # Oversold = bullish
        else:
            # Neutral zone with slight direction
            signals.append((rsi - 50) / 100)

        # MACD contribution
        if close[-1] > 0:
            macd_normalized = macd_hist / close[-1] * 100
            signals.append(np.clip(macd_normalized, -1, 1))

        # Recent price action
        if len(close) >= 20:
            recent_return = (close[-1] - close[-20]) / close[-20]
            signals.append(np.clip(recent_return * 5, -1, 1))

        if signals:
            return float(np.clip(np.mean(signals), -1, 1))
        return 0.0

    def _calculate_strength(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """Calculate trend strength (simplified ADX)."""
        if len(close) < 20:
            return 0.0

        # Directional movement
        plus_dm = np.maximum(np.diff(high), 0)
        minus_dm = np.maximum(-np.diff(low), 0)

        # True range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Smooth
        period = 14
        if len(tr) < period:
            return 0.0

        plus_di = np.mean(plus_dm[-period:]) / (np.mean(tr[-period:]) + 1e-10)
        minus_di = np.mean(minus_dm[-period:]) / (np.mean(tr[-period:]) + 1e-10)

        # DX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # Normalize to 0-1
        return float(min(1.0, dx * 2))

    def _calculate_momentum(self, close: np.ndarray) -> float:
        """Calculate momentum (rate of change acceleration)."""
        if len(close) < 20:
            return 0.0

        # Recent ROC
        roc_recent = (close[-1] - close[-5]) / close[-5] if close[-5] != 0 else 0

        # Previous ROC
        roc_prev = (close[-5] - close[-10]) / close[-10] if len(close) >= 10 and close[-10] != 0 else 0

        # Momentum = change in ROC
        momentum = roc_recent - roc_prev

        return float(np.clip(momentum * 10, -1, 1))

    def _is_higher_high(self, high: np.ndarray) -> bool:
        """Check if making higher highs."""
        if len(high) < 20:
            return False

        # Compare recent high to previous high
        recent_high = np.max(high[-5:])
        prev_high = np.max(high[-15:-5])

        return recent_high > prev_high

    def _is_higher_low(self, low: np.ndarray) -> bool:
        """Check if making higher lows."""
        if len(low) < 20:
            return False

        recent_low = np.min(low[-5:])
        prev_low = np.min(low[-15:-5])

        return recent_low > prev_low

    def _is_lower_high(self, high: np.ndarray) -> bool:
        """Check if making lower highs."""
        if len(high) < 20:
            return False

        recent_high = np.max(high[-5:])
        prev_high = np.max(high[-15:-5])

        return recent_high < prev_high

    def _is_lower_low(self, low: np.ndarray) -> bool:
        """Check if making lower lows."""
        if len(low) < 20:
            return False

        recent_low = np.min(low[-5:])
        prev_low = np.min(low[-15:-5])

        return recent_low < prev_low

    def _neutral_signal(self, reason: str) -> ConfluenceSignal:
        """Return neutral signal with reason."""
        logger.debug(f"Neutral confluence signal: {reason}")

        return ConfluenceSignal(
            direction=0.0,
            confluence_score=0.0,
            confidence=0.0,
            action="neutral",
            signal_type="none",
            metadata={'reason': reason}
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current multi-timeframe state."""
        if not self.trends:
            return {'status': 'no_data'}

        signal = self.analyze()

        return {
            'direction': signal.direction,
            'confluence_score': signal.confluence_score,
            'confidence': signal.confidence,
            'action': signal.action,
            'signal_type': signal.signal_type,
            'has_conflict': signal.has_conflict,
            'conflicting_timeframes': signal.conflicting_timeframes,
            'dominant_timeframe': signal.dominant_timeframe,
            'timeframe_directions': {
                tf.value: round(trend.direction, 3)
                for tf, trend in self.trends.items()
            },
            'interpretation': self._interpret_confluence(signal)
        }

    def _interpret_confluence(self, signal: ConfluenceSignal) -> str:
        """Generate human-readable interpretation."""
        if signal.confluence_score < 0.3:
            return "No clear confluence - stay neutral"
        elif signal.confluence_score < 0.6:
            if signal.has_conflict:
                return "Weak confluence with conflicts - use caution"
            else:
                return "Moderate confluence - small position"
        else:
            if signal.action == "long":
                return "Strong bullish confluence - favorable long setup"
            elif signal.action == "short":
                return "Strong bearish confluence - favorable short setup"
            else:
                return "High confluence but sideways - wait for breakout"
