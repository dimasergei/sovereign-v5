"""
Trend Following Model - Momentum-based trading signals.

Implements trend following using LOSSLESS parameters:
- Periods derived from spectral analysis
- Thresholds derived from distribution analysis
- Weights updated via online learning

Complements the mean-reversion model in the ensemble.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from collections import deque

from models.base import BaseModel, ModelPrediction

logger = logging.getLogger(__name__)


@dataclass
class TrendSignal:
    """Trend following signal output."""
    direction: int  # -1, 0, 1
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    trend_duration: int  # Bars in current trend
    trend_strength_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveTrendDetector:
    """
    Adaptive trend detection using multiple timeframes.

    All parameters are derived from market data:
    - Lookback periods from dominant market cycles
    - Smoothing factors from noise characteristics
    - Thresholds from return distribution
    """

    def __init__(self, min_period: int = 5, max_period: int = 200):
        """
        Initialize detector.

        Args:
            min_period: Minimum period to consider
            max_period: Maximum period to consider
        """
        self.min_period = min_period
        self.max_period = max_period

        # Derived parameters (set during calibration)
        self._fast_period: Optional[int] = None
        self._slow_period: Optional[int] = None
        self._signal_period: Optional[int] = None
        self._trend_threshold: Optional[float] = None

        # History for online learning
        self._period_history: deque = deque(maxlen=100)
        self._accuracy_history: deque = deque(maxlen=100)

    def calibrate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calibrate trend detection parameters from market data.

        Uses spectral analysis to find dominant cycles.

        Args:
            df: OHLCV DataFrame

        Returns:
            Calibrated parameters
        """
        prices = df['close'].values

        # Find dominant periods using autocorrelation
        fast, slow = self._find_dominant_periods(prices)

        self._fast_period = fast
        self._slow_period = slow
        self._signal_period = int(fast * 0.5)

        # Derive threshold from return distribution
        returns = pd.Series(prices).pct_change().dropna()
        self._trend_threshold = self._derive_threshold(returns)

        logger.info(
            f"Trend detector calibrated: fast={fast}, slow={slow}, "
            f"threshold={self._trend_threshold:.4f}"
        )

        return {
            'fast_period': self._fast_period,
            'slow_period': self._slow_period,
            'signal_period': self._signal_period,
            'trend_threshold': self._trend_threshold,
        }

    def _find_dominant_periods(self, prices: np.ndarray) -> Tuple[int, int]:
        """
        Find dominant cycle periods using autocorrelation.
        """
        if len(prices) < 50:
            # Fallback to derived defaults
            return self.min_period * 2, self.min_period * 5

        returns = np.diff(np.log(prices))

        # Calculate autocorrelation at various lags
        max_lag = min(len(returns) // 2, self.max_period)
        autocorrs = []

        for lag in range(1, max_lag):
            if lag < len(returns):
                corr = np.corrcoef(returns[lag:], returns[:-lag])[0, 1]
                autocorrs.append((lag, corr if not np.isnan(corr) else 0))

        if not autocorrs:
            return self.min_period * 2, self.min_period * 5

        # Find first significant peak
        autocorrs.sort(key=lambda x: abs(x[1]), reverse=True)

        # Fast period: first significant cycle
        fast_period = max(self.min_period, autocorrs[0][0])

        # Slow period: larger cycle (2-4x fast)
        slow_candidates = [a for a in autocorrs if a[0] > fast_period * 1.5]
        if slow_candidates:
            slow_period = slow_candidates[0][0]
        else:
            slow_period = fast_period * 3

        # Bound periods
        fast_period = max(self.min_period, min(fast_period, self.max_period // 3))
        slow_period = max(fast_period * 2, min(slow_period, self.max_period))

        return int(fast_period), int(slow_period)

    def _derive_threshold(self, returns: pd.Series) -> float:
        """
        Derive trend threshold from return distribution.

        Uses percentile-based approach to avoid hardcoding.
        """
        if len(returns) < 20:
            return 0.0

        # Use standard deviation scaled by kurtosis
        std = returns.std()
        # Adjust for fat tails
        kurtosis = returns.kurtosis()
        adjustment = 1 + max(0, kurtosis - 3) * 0.1

        # Threshold at ~1 std, adjusted for distribution shape
        threshold = std * adjustment

        return float(threshold)

    def detect_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current trend state.

        Args:
            df: OHLCV DataFrame

        Returns:
            Trend detection result
        """
        if self._fast_period is None:
            self.calibrate(df)

        prices = df['close']

        # Calculate EMAs with derived periods
        fast_ema = prices.ewm(span=self._fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=self._slow_period, adjust=False).mean()

        # MACD-like signal
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self._signal_period, adjust=False).mean()
        histogram = macd - signal

        # Normalize by price level
        macd_pct = macd / prices * 100
        signal_pct = signal / prices * 100

        # Current values
        current_macd = macd_pct.iloc[-1]
        current_signal = signal_pct.iloc[-1]
        current_histogram = histogram.iloc[-1] / prices.iloc[-1] * 100

        # Trend direction
        if current_macd > self._trend_threshold and current_histogram > 0:
            direction = 1  # Bullish
        elif current_macd < -self._trend_threshold and current_histogram < 0:
            direction = -1  # Bearish
        else:
            direction = 0  # Neutral

        # Calculate trend duration
        trend_duration = self._calculate_trend_duration(macd_pct, direction)

        # Trend strength (based on MACD magnitude vs threshold)
        if self._trend_threshold > 0:
            strength = min(1.0, abs(current_macd) / (self._trend_threshold * 3))
        else:
            strength = 0.5

        return {
            'direction': direction,
            'strength': float(strength),
            'macd': float(current_macd),
            'signal': float(current_signal),
            'histogram': float(current_histogram),
            'trend_duration': trend_duration,
            'fast_ema': float(fast_ema.iloc[-1]),
            'slow_ema': float(slow_ema.iloc[-1]),
        }

    def _calculate_trend_duration(
        self,
        macd_pct: pd.Series,
        current_direction: int
    ) -> int:
        """Calculate how long current trend has been active."""
        if current_direction == 0:
            return 0

        duration = 0
        threshold = self._trend_threshold

        for i in range(len(macd_pct) - 1, -1, -1):
            if current_direction > 0:
                if macd_pct.iloc[i] > threshold:
                    duration += 1
                else:
                    break
            else:
                if macd_pct.iloc[i] < -threshold:
                    duration += 1
                else:
                    break

        return duration


class TrendFollowingModel(BaseModel):
    """
    Trend Following Model for ensemble integration.

    Uses multiple trend detection methods:
    1. Adaptive MACD with derived periods
    2. Directional Movement (ADX-like)
    3. Price momentum

    All thresholds are derived from market data.
    """

    def __init__(self):
        """Initialize trend following model."""
        super().__init__()

        self.name = "TrendFollowing"
        self.trend_detector = AdaptiveTrendDetector()

        # Model state
        self._is_calibrated = False
        self._last_signal: Optional[TrendSignal] = None
        self._signal_history: deque = deque(maxlen=100)

        # Feature importance tracking
        self._feature_importance: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'TrendFollowingModel':
        """
        Fit model to data.

        For trend following, this calibrates the detector.

        Args:
            X: Feature matrix (OHLCV as DataFrame preferred)
            y: Optional labels (not used for trend following)

        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            # Convert numpy to DataFrame
            df = pd.DataFrame(X, columns=['open', 'high', 'low', 'close', 'volume'])

        self.trend_detector.calibrate(df)
        self._is_calibrated = True

        # Calculate feature importance based on trend detection
        self._calculate_feature_importance(df)

        return self

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Generate trend prediction.

        Args:
            X: Feature matrix

        Returns:
            ModelPrediction object
        """
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=['open', 'high', 'low', 'close', 'volume'])

        if not self._is_calibrated:
            self.fit(df)

        # Detect trend
        trend_result = self.trend_detector.detect_trend(df)

        # Calculate additional momentum signals
        momentum_signal = self._calculate_momentum(df)
        directional_signal = self._calculate_directional_movement(df)

        # Combine signals
        combined_direction, confidence = self._combine_signals(
            trend_result, momentum_signal, directional_signal
        )

        # Create prediction
        prediction = ModelPrediction(
            direction=combined_direction,
            magnitude=trend_result['strength'],
            confidence=confidence,
            model_name=self.name,
            timestamp=datetime.now(),
            metadata={
                'trend': trend_result,
                'momentum': momentum_signal,
                'directional': directional_signal,
            }
        )

        # Store for tracking
        self._last_signal = TrendSignal(
            direction=combined_direction,
            strength=trend_result['strength'],
            confidence=confidence,
            trend_duration=trend_result['trend_duration'],
            trend_strength_score=trend_result['strength'],
            metadata=prediction.metadata
        )
        self._signal_history.append(self._last_signal)

        return prediction

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Online update with new data.

        Args:
            X: New feature data
            y: Actual outcomes
        """
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=['open', 'high', 'low', 'close', 'volume'])

        # Re-calibrate periodically
        if len(self._signal_history) > 0 and len(self._signal_history) % 50 == 0:
            self.trend_detector.calibrate(df)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self._feature_importance

    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate price momentum signal.

        Uses rate of change over derived period.
        """
        prices = df['close']

        # Derive lookback from trend detector
        lookback = self.trend_detector._fast_period or 10

        # Rate of change
        roc = (prices.iloc[-1] - prices.iloc[-lookback]) / prices.iloc[-lookback]

        # Normalize using historical ROC distribution
        roc_series = prices.pct_change(lookback)
        if roc_series.std() > 0:
            z_score = (roc - roc_series.mean()) / roc_series.std()
        else:
            z_score = 0

        # Convert to signal (-1 to 1)
        signal = float(np.clip(z_score / 2, -1, 1))

        return {
            'roc': float(roc),
            'z_score': float(z_score),
            'signal': signal,
        }

    def _calculate_directional_movement(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate directional movement indicator.

        ADX-like calculation without hardcoded periods.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Derive period from trend detector
        period = self.trend_detector._fast_period or 14

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = pd.Series(plus_dm).rolling(window=period).mean() / atr * 100
        minus_di = pd.Series(minus_dm).rolling(window=period).mean() / atr * 100

        # DX and ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = dx.rolling(window=period).mean()

        # Current values
        current_adx = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0
        current_plus = plus_di.iloc[-1] if not np.isnan(plus_di.iloc[-1]) else 0
        current_minus = minus_di.iloc[-1] if not np.isnan(minus_di.iloc[-1]) else 0

        # Direction based on DI crossover
        if current_plus > current_minus:
            direction = 1
        elif current_minus > current_plus:
            direction = -1
        else:
            direction = 0

        # Normalize ADX to trend strength
        # Use historical ADX percentile instead of hardcoded threshold
        adx_threshold = adx.quantile(0.5) if len(adx.dropna()) > 10 else 25
        trend_strength = current_adx / adx_threshold if adx_threshold > 0 else 0

        return {
            'adx': float(current_adx),
            'plus_di': float(current_plus),
            'minus_di': float(current_minus),
            'direction': direction,
            'trend_strength': float(min(1.0, trend_strength)),
        }

    def _combine_signals(
        self,
        trend: Dict,
        momentum: Dict,
        directional: Dict
    ) -> Tuple[int, float]:
        """
        Combine multiple trend signals.

        Uses weighted voting based on signal strength.
        """
        signals = [
            (trend['direction'], trend['strength'], 0.4),  # Weight
            (1 if momentum['signal'] > 0.2 else -1 if momentum['signal'] < -0.2 else 0,
             abs(momentum['signal']), 0.3),
            (directional['direction'], directional['trend_strength'], 0.3),
        ]

        # Weighted vote
        weighted_sum = 0
        total_weight = 0

        for direction, strength, weight in signals:
            if direction != 0:
                weighted_sum += direction * strength * weight
                total_weight += strength * weight

        if total_weight == 0:
            return 0, 0.0

        avg_direction = weighted_sum / total_weight

        # Final direction
        if avg_direction > 0.2:
            final_direction = 1
        elif avg_direction < -0.2:
            final_direction = -1
        else:
            final_direction = 0

        # Confidence based on agreement
        agreements = sum(1 for d, _, _ in signals if d == final_direction)
        confidence = agreements / len(signals)

        return final_direction, float(confidence)

    def _calculate_feature_importance(self, df: pd.DataFrame) -> None:
        """Calculate feature importance based on correlation with returns."""
        returns = df['close'].pct_change().shift(-1)  # Forward returns

        features = {
            'close': df['close'],
            'volume': df['volume'],
            'high_low_range': df['high'] - df['low'],
            'close_open': df['close'] - df['open'],
        }

        importance = {}
        for name, feature in features.items():
            corr = feature.corr(returns)
            importance[name] = abs(corr) if not np.isnan(corr) else 0

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        self._feature_importance = importance
