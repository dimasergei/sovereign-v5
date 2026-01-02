"""
Signal Generator - Institutional-Grade Entry Logic.

CRITICAL PRINCIPLES:
1. Only trade WITH the trend (never counter-trend)
2. Only enter on PULLBACKS (never chase)
3. Require CONFIRMATION (reversal candle, RSI turning)
4. Asset-specific parameters
5. Volatility-aware sizing

Entry Logic:
- LONG in uptrend: Wait for pullback to MA20/MA50, require bullish candle + RSI turning up
- SHORT in downtrend: Wait for rally to MA20/MA50, require bearish candle + RSI turning down
- BLOCK if at recent highs/lows (no chasing)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from signals.trend_filter import TrendFilter, TrendState, TrendDirection
from signals.volatility_filter import VolatilityFilter, VolatilityState
from config.asset_profiles import get_profile, DEFAULT_PROFILE


logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Complete trading signal with all relevant information."""
    symbol: str = ""
    action: str = "neutral"  # "long", "short", "neutral"
    direction: float = 0.0  # -1 to 1
    confidence: float = 0.0  # 0 to 1
    position_scalar: float = 1.0  # 0 to 1

    # Risk parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0

    # Context
    regime: str = "unknown"
    entry_reason: str = ""
    model_agreement: float = 1.0

    # Trend info
    trend_direction: str = "neutral"
    trend_strength: float = 0.0
    higher_tf_aligned: bool = False

    # Filter info
    filter_reason: str = ""
    filters_applied: List[str] = field(default_factory=list)
    vol_regime: str = "normal"

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    calibration_confidence: float = 0.0
    contributing_models: List[str] = field(default_factory=list)

    # Market data
    current_price: float = 0.0
    atr: float = 0.0


class SignalGenerator:
    """
    Institutional-grade signal generator.

    Key principles:
    1. Only trade WITH the trend
    2. Only enter on PULLBACKS, never chase
    3. Require CONFIRMATION (reversal candle, RSI turn)
    4. Adaptive parameters per asset class
    5. Volatility-aware sizing
    """

    def __init__(self, min_confidence: float = 0.5):
        """Initialize signal generator."""
        self.trend_filter = TrendFilter()
        self.volatility_filter = VolatilityFilter()
        self.min_confidence = min_confidence

        # Track blocked signals for monitoring
        self.blocked_signals = {
            "counter_trend": 0,
            "volatility": 0,
            "no_pullback": 0,
            "no_confirmation": 0,
            "chasing": 0
        }

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        calibration: Any = None
    ) -> TradingSignal:
        """
        Generate institutional-quality trading signal.

        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol
            calibration: Optional calibration result

        Returns:
            TradingSignal with action, confidence, and risk parameters
        """
        # Get asset-specific parameters
        profile = get_profile(symbol)

        # Ensure we have enough data
        if len(df) < 60:
            return TradingSignal(
                symbol=symbol,
                action="neutral",
                filter_reason="insufficient_data"
            )

        # Step 1: Check volatility regime
        vol_state = self.volatility_filter.analyze(df)
        if not vol_state.can_trade:
            self.blocked_signals["volatility"] += 1
            return TradingSignal(
                symbol=symbol,
                action="neutral",
                filter_reason="extreme_volatility",
                vol_regime=vol_state.regime
            )

        # Step 2: Analyze trend
        trend_state = self.trend_filter.analyze(df)

        # Step 3: Generate signal based on trend
        if trend_state.is_trending:
            signal = self._generate_trend_signal(df, symbol, trend_state, profile)
        else:
            signal = self._generate_mean_reversion_signal(df, symbol, trend_state, profile)

        # Step 4: Apply volatility scalar to position size
        signal.position_scalar *= vol_state.position_scalar * profile['max_position_pct']
        signal.vol_regime = vol_state.regime
        signal.symbol = symbol

        return signal

    def _generate_trend_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        trend_state: TrendState,
        profile: Dict[str, Any]
    ) -> TradingSignal:
        """
        Generate trend-following signal with strict entry criteria.

        CRITICAL: Only enter on pullbacks with confirmation.
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_prices = df['open'].values

        current_price = close[-1]

        # Calculate indicators using asset-specific params
        ma_fast = pd.Series(close).rolling(profile['trend_ma_fast']).mean()
        ma_slow = pd.Series(close).rolling(profile['trend_ma_slow']).mean()

        # ATR
        atr = self._calculate_atr(high, low, close, 14)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 1.0

        # RSI with lookback
        rsi = self._calculate_rsi(close, 14)
        rsi_prev = self._calculate_rsi(close[:-1], 14) if len(close) > 15 else rsi

        # Price distances from MAs
        dist_from_ma_fast = ((current_price - ma_fast.iloc[-1]) / ma_fast.iloc[-1] * 100
                            if ma_fast.iloc[-1] > 0 else 0)
        dist_from_ma_slow = ((current_price - ma_slow.iloc[-1]) / ma_slow.iloc[-1] * 100
                            if ma_slow.iloc[-1] > 0 else 0)

        # Recent extremes (for blocking chasing)
        recent_high = max(high[-10:])
        recent_low = min(low[-10:])
        at_recent_high = current_price >= recent_high * 0.995
        at_recent_low = current_price <= recent_low * 1.005

        # Pullback calculations
        pullback_from_high = (recent_high - current_price) / recent_high * 100 if recent_high > 0 else 0
        pullback_from_low = (current_price - recent_low) / recent_low * 100 if recent_low > 0 else 0

        # Candle analysis
        is_bullish_candle = close[-1] > open_prices[-1]
        is_bearish_candle = close[-1] < open_prices[-1]
        is_hammer = self._is_hammer(open_prices[-1], high[-1], low[-1], close[-1])
        is_shooting_star = self._is_shooting_star(open_prices[-1], high[-1], low[-1], close[-1])

        # RSI turning
        rsi_turning_up = rsi > rsi_prev and rsi < 50
        rsi_turning_down = rsi < rsi_prev and rsi > 50

        # Minimum pullback requirement (asset-specific)
        min_pullback = profile['min_pullback_pct']
        ma_proximity = profile['ma_proximity_atr']

        # ==================== LONG ENTRIES ====================
        if trend_state.direction in [TrendDirection.STRONG_UP, TrendDirection.MILD_UP]:

            # BLOCK: Don't chase - require pullback from recent high
            if at_recent_high:
                self.blocked_signals["chasing"] += 1
                return TradingSignal(
                    symbol=symbol,
                    action="neutral",
                    filter_reason="chasing_blocked",
                    trend_direction=trend_state.direction.value,
                    current_price=current_price,
                    atr=atr
                )

            # BLOCK: Insufficient pullback
            if pullback_from_high < min_pullback:
                self.blocked_signals["no_pullback"] += 1
                return TradingSignal(
                    symbol=symbol,
                    action="neutral",
                    filter_reason="insufficient_pullback",
                    trend_direction=trend_state.direction.value,
                    current_price=current_price,
                    atr=atr
                )

            # SETUP 1: Pullback to fast MA with bullish confirmation
            if abs(dist_from_ma_fast) < atr_pct * ma_proximity:
                if is_bullish_candle and rsi_turning_up:
                    confidence = 0.70

                    if trend_state.higher_tf_aligned:
                        confidence += 0.10
                    if is_hammer:
                        confidence += 0.08

                    stop = min(low[-3:]) - atr * profile['atr_multiplier_stop']
                    target = current_price + atr * profile['atr_multiplier_target']

                    return TradingSignal(
                        symbol=symbol,
                        action="long",
                        direction=0.7,
                        confidence=min(0.92, confidence),
                        entry_reason="fast_ma_pullback",
                        trend_direction=trend_state.direction.value,
                        trend_strength=trend_state.strength,
                        higher_tf_aligned=trend_state.higher_tf_aligned,
                        stop_loss=stop,
                        take_profit=target,
                        stop_loss_atr_mult=profile['atr_multiplier_stop'],
                        take_profit_atr_mult=profile['atr_multiplier_target'],
                        current_price=current_price,
                        atr=atr,
                        contributing_models=["trend_following"]
                    )

            # SETUP 2: Pullback to slow MA (deeper, higher confidence)
            if abs(dist_from_ma_slow) < atr_pct * ma_proximity and dist_from_ma_fast < 0:
                if is_bullish_candle:
                    confidence = 0.75

                    if trend_state.higher_tf_aligned:
                        confidence += 0.10

                    stop = min(low[-5:]) - atr * profile['atr_multiplier_stop']
                    target = current_price + atr * profile['atr_multiplier_target'] * 1.2

                    return TradingSignal(
                        symbol=symbol,
                        action="long",
                        direction=0.8,
                        confidence=min(0.92, confidence),
                        entry_reason="slow_ma_pullback",
                        trend_direction=trend_state.direction.value,
                        trend_strength=trend_state.strength,
                        higher_tf_aligned=trend_state.higher_tf_aligned,
                        stop_loss=stop,
                        take_profit=target,
                        stop_loss_atr_mult=profile['atr_multiplier_stop'],
                        take_profit_atr_mult=profile['atr_multiplier_target'] * 1.2,
                        current_price=current_price,
                        atr=atr,
                        contributing_models=["trend_following"]
                    )

            # SETUP 3: RSI oversold in uptrend
            if rsi < profile['rsi_oversold'] and trend_state.price_vs_ma50 > 0:
                if is_bullish_candle:
                    confidence = 0.65
                    stop = low[-1] - atr * profile['atr_multiplier_stop']
                    target = current_price + atr * profile['atr_multiplier_target']

                    return TradingSignal(
                        symbol=symbol,
                        action="long",
                        direction=0.6,
                        confidence=confidence,
                        entry_reason="rsi_oversold_uptrend",
                        trend_direction=trend_state.direction.value,
                        stop_loss=stop,
                        take_profit=target,
                        stop_loss_atr_mult=profile['atr_multiplier_stop'],
                        take_profit_atr_mult=profile['atr_multiplier_target'],
                        current_price=current_price,
                        atr=atr,
                        contributing_models=["trend_following"]
                    )

        # ==================== SHORT ENTRIES ====================
        elif trend_state.direction in [TrendDirection.STRONG_DOWN, TrendDirection.MILD_DOWN]:

            # BLOCK: Don't chase breakdown
            if at_recent_low:
                self.blocked_signals["chasing"] += 1
                return TradingSignal(
                    symbol=symbol,
                    action="neutral",
                    filter_reason="chasing_blocked",
                    trend_direction=trend_state.direction.value,
                    current_price=current_price,
                    atr=atr
                )

            # BLOCK: Insufficient rally
            if pullback_from_low < min_pullback:
                self.blocked_signals["no_pullback"] += 1
                return TradingSignal(
                    symbol=symbol,
                    action="neutral",
                    filter_reason="insufficient_rally",
                    trend_direction=trend_state.direction.value,
                    current_price=current_price,
                    atr=atr
                )

            # SETUP 1: Rally to fast MA with bearish confirmation
            if abs(dist_from_ma_fast) < atr_pct * ma_proximity:
                if is_bearish_candle and rsi_turning_down:
                    confidence = 0.70

                    if trend_state.higher_tf_aligned:
                        confidence += 0.10
                    if is_shooting_star:
                        confidence += 0.08

                    stop = max(high[-3:]) + atr * profile['atr_multiplier_stop']
                    target = current_price - atr * profile['atr_multiplier_target']

                    return TradingSignal(
                        symbol=symbol,
                        action="short",
                        direction=-0.7,
                        confidence=min(0.92, confidence),
                        entry_reason="fast_ma_rally",
                        trend_direction=trend_state.direction.value,
                        trend_strength=trend_state.strength,
                        higher_tf_aligned=trend_state.higher_tf_aligned,
                        stop_loss=stop,
                        take_profit=target,
                        stop_loss_atr_mult=profile['atr_multiplier_stop'],
                        take_profit_atr_mult=profile['atr_multiplier_target'],
                        current_price=current_price,
                        atr=atr,
                        contributing_models=["trend_following"]
                    )

            # SETUP 2: Rally to slow MA
            if abs(dist_from_ma_slow) < atr_pct * ma_proximity and dist_from_ma_fast > 0:
                if is_bearish_candle:
                    confidence = 0.75

                    if trend_state.higher_tf_aligned:
                        confidence += 0.10

                    stop = max(high[-5:]) + atr * profile['atr_multiplier_stop']
                    target = current_price - atr * profile['atr_multiplier_target'] * 1.2

                    return TradingSignal(
                        symbol=symbol,
                        action="short",
                        direction=-0.8,
                        confidence=min(0.92, confidence),
                        entry_reason="slow_ma_rally",
                        trend_direction=trend_state.direction.value,
                        trend_strength=trend_state.strength,
                        higher_tf_aligned=trend_state.higher_tf_aligned,
                        stop_loss=stop,
                        take_profit=target,
                        stop_loss_atr_mult=profile['atr_multiplier_stop'],
                        take_profit_atr_mult=profile['atr_multiplier_target'] * 1.2,
                        current_price=current_price,
                        atr=atr,
                        contributing_models=["trend_following"]
                    )

            # SETUP 3: RSI overbought in downtrend
            if rsi > profile['rsi_overbought'] and trend_state.price_vs_ma50 < 0:
                if is_bearish_candle:
                    confidence = 0.65
                    stop = high[-1] + atr * profile['atr_multiplier_stop']
                    target = current_price - atr * profile['atr_multiplier_target']

                    return TradingSignal(
                        symbol=symbol,
                        action="short",
                        direction=-0.6,
                        confidence=confidence,
                        entry_reason="rsi_overbought_downtrend",
                        trend_direction=trend_state.direction.value,
                        stop_loss=stop,
                        take_profit=target,
                        stop_loss_atr_mult=profile['atr_multiplier_stop'],
                        take_profit_atr_mult=profile['atr_multiplier_target'],
                        current_price=current_price,
                        atr=atr,
                        contributing_models=["trend_following"]
                    )

        # No valid setup found - waiting for better entry
        self.blocked_signals["no_confirmation"] += 1
        return TradingSignal(
            symbol=symbol,
            action="neutral",
            filter_reason="no_entry_setup",
            trend_direction=trend_state.direction.value if trend_state else "neutral",
            current_price=current_price,
            atr=atr
        )

    def _generate_mean_reversion_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        trend_state: TrendState,
        profile: Dict[str, Any]
    ) -> TradingSignal:
        """
        Mean reversion for ranging markets.

        Only used when trend is NEUTRAL.
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_prices = df['open'].values

        current_price = close[-1]

        # Z-score
        lookback = 50
        if len(close) < lookback:
            lookback = len(close)

        mean = np.mean(close[-lookback:])
        std = np.std(close[-lookback:])

        if std == 0:
            return TradingSignal(
                symbol=symbol,
                action="neutral",
                filter_reason="no_volatility",
                regime="mean_reverting"
            )

        zscore = (close[-1] - mean) / std

        # RSI
        rsi = self._calculate_rsi(close, 14)

        # ATR
        atr = self._calculate_atr(high, low, close, 14)

        # Candle
        is_bullish = close[-1] > open_prices[-1]
        is_bearish = close[-1] < open_prices[-1]

        # LONG: Oversold bounce
        if zscore < -2.0 and rsi < 30 and is_bullish:
            confidence = min(0.70, 0.50 + abs(zscore) * 0.08)
            stop = low[-1] - atr * profile['atr_multiplier_stop']
            target = mean  # Target the mean

            return TradingSignal(
                symbol=symbol,
                action="long",
                direction=0.5,
                confidence=confidence,
                entry_reason="mean_reversion_long",
                regime="mean_reverting",
                stop_loss=stop,
                take_profit=target,
                current_price=current_price,
                atr=atr,
                contributing_models=["mean_reversion"]
            )

        # SHORT: Overbought fade
        if zscore > 2.0 and rsi > 70 and is_bearish:
            confidence = min(0.70, 0.50 + abs(zscore) * 0.08)
            stop = high[-1] + atr * profile['atr_multiplier_stop']
            target = mean

            return TradingSignal(
                symbol=symbol,
                action="short",
                direction=-0.5,
                confidence=confidence,
                entry_reason="mean_reversion_short",
                regime="mean_reverting",
                stop_loss=stop,
                take_profit=target,
                current_price=current_price,
                atr=atr,
                contributing_models=["mean_reversion"]
            )

        return TradingSignal(
            symbol=symbol,
            action="neutral",
            filter_reason="no_extreme",
            regime="mean_reverting",
            current_price=current_price,
            atr=atr
        )

    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        if len(high) < period + 1:
            return high[-1] - low[-1] if len(high) > 0 else 0.0

        # True Range components
        hl = high[-period:] - low[-period:]
        hc = np.abs(high[-period:] - np.concatenate([[close[-period - 1]], close[-period:-1]]))
        lc = np.abs(low[-period:] - np.concatenate([[close[-period - 1]], close[-period:-1]]))

        tr = np.maximum(hl, np.maximum(hc, lc))
        return np.mean(tr)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0  # Neutral default

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _is_hammer(
        self,
        open_p: float,
        high: float,
        low: float,
        close: float
    ) -> bool:
        """
        Detect hammer candle (bullish reversal).

        Hammer: small body, long lower wick (2x+ body), small upper wick
        """
        body = abs(close - open_p)
        if body == 0:
            return False

        lower_wick = min(open_p, close) - low
        upper_wick = high - max(open_p, close)

        return lower_wick > body * 2 and upper_wick < body * 0.5

    def _is_shooting_star(
        self,
        open_p: float,
        high: float,
        low: float,
        close: float
    ) -> bool:
        """
        Detect shooting star candle (bearish reversal).

        Shooting star: small body, long upper wick (2x+ body), small lower wick
        """
        body = abs(close - open_p)
        if body == 0:
            return False

        lower_wick = min(open_p, close) - low
        upper_wick = high - max(open_p, close)

        return upper_wick > body * 2 and lower_wick < body * 0.5

    def get_blocked_stats(self) -> Dict[str, int]:
        """Get statistics on blocked signals."""
        return self.blocked_signals.copy()

    def reset_stats(self):
        """Reset blocked signal counters."""
        for key in self.blocked_signals:
            self.blocked_signals[key] = 0
