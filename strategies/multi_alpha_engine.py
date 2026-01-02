# strategies/multi_alpha_engine.py
"""
Renaissance-style Multi-Alpha Signal Generation Engine.

Key principles:
1. Multiple strategies run independently (Trend, Mean Reversion, Breakout, Lead-Lag)
2. Each has different edge source and timeframe
3. Signals are combined with dynamic weighting
4. More trades = more opportunities for edge to compound

Target: 150-200 trades/year per symbol with 52% win rate and 2:1 R:R
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    LEAD_LAG = "lead_lag"


# Crypto assets - NEVER SHORT these
CRYPTO_PATTERNS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'AVAX', 'MATIC', 'LTC', 'BNB']


@dataclass
class AlphaSignal:
    """Individual alpha signal from one strategy."""
    strategy: StrategyType
    direction: float  # -1 to 1
    confidence: float  # 0 to 1
    expected_move: float  # Expected % move
    holding_period: int  # Expected bars to hold
    stop_distance: float  # ATR multiplier
    entry_reason: str


@dataclass
class CombinedSignal:
    """Combined signal from all strategies."""
    action: str  # "long", "short", "neutral"
    direction: float
    confidence: float
    position_size: float  # 0 to 1
    strategies_agreeing: List[str] = field(default_factory=list)
    primary_strategy: str = "none"
    stop_loss: float = 0.0
    take_profit: float = 0.0
    expected_holding: int = 0
    entry_reason: str = ""


class MultiAlphaEngine:
    """
    Renaissance-style multi-alpha signal generation.

    Key insight: Renaissance makes thousands of small bets.
    Stop waiting for perfection - take many trades with slight edge.
    """

    def __init__(self):
        # Strategy weights (can be updated based on recent performance)
        self.strategy_weights = {
            StrategyType.TREND: 0.30,
            StrategyType.MEAN_REVERSION: 0.35,
            StrategyType.BREAKOUT: 0.15,
            StrategyType.LEAD_LAG: 0.20
        }

        # Trade counting for debugging
        self.signal_counts = {s: 0 for s in StrategyType}

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is cryptocurrency - NEVER SHORT CRYPTO."""
        if not symbol:
            return False
        upper = symbol.upper()
        return any(pattern in upper for pattern in CRYPTO_PATTERNS)

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        related_data: Dict[str, pd.DataFrame] = None
    ) -> CombinedSignal:
        """
        Generate combined signal from all strategies.

        Args:
            df: OHLCV data for primary symbol
            symbol: Symbol being traded
            related_data: Data for correlated symbols (for lead-lag)
        """
        signals = []

        # Run all strategies - pass symbol for crypto detection
        trend_signal = self._trend_strategy(df, symbol=symbol)
        if trend_signal:
            signals.append(trend_signal)
            self.signal_counts[StrategyType.TREND] += 1

        mr_signal = self._mean_reversion_strategy(df, symbol=symbol)
        if mr_signal:
            signals.append(mr_signal)
            self.signal_counts[StrategyType.MEAN_REVERSION] += 1

        breakout_signal = self._breakout_strategy(df, symbol=symbol)
        if breakout_signal:
            signals.append(breakout_signal)
            self.signal_counts[StrategyType.BREAKOUT] += 1

        if related_data:
            leadlag_signal = self._lead_lag_strategy(df, symbol, related_data)
            if leadlag_signal:
                signals.append(leadlag_signal)
                self.signal_counts[StrategyType.LEAD_LAG] += 1

        # Combine signals
        return self._combine_signals(signals, df)

    def _trend_strategy(self, df: pd.DataFrame, symbol: str = None) -> Optional[AlphaSignal]:
        """
        Trend following strategy with MACRO TREND BIAS.

        CRITICAL FIX: Check higher timeframe trend before allowing any signal.
        In a bull market (price > EMA50 with rising slope), only generate LONG signals.
        In a bear market (price < EMA50 with falling slope), only generate SHORT signals.

        CRYPTO: NEVER generate short signals for crypto assets.
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if len(close) < 50:
            return None

        # Check if crypto - NEVER SHORT CRYPTO
        is_crypto = self._is_crypto(symbol)

        # EMAs
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)

        # MACRO TREND DETECTION - THE KEY FIX
        # Use 50-day slope and price position to determine macro trend
        ema50_10_bars_ago = self._ema(close[:-10], 50) if len(close) > 60 else ema50
        ema50_slope = (ema50 - ema50_10_bars_ago) / ema50_10_bars_ago * 100

        macro_bullish = close[-1] > ema50 and ema50_slope > 0
        macro_bearish = close[-1] < ema50 and ema50_slope < 0

        # Momentum (rate of change)
        roc_10 = (close[-1] - close[-10]) / close[-10] * 100
        roc_5 = (close[-1] - close[-5]) / close[-5] * 100

        # ATR for stops
        atr = self._atr(high, low, close, 14)

        # LONG: Only in macro bullish environment
        if macro_bullish:
            # Standard trend following: price > EMA20 > EMA50 with momentum
            if close[-1] > ema20 > ema50 and roc_10 > 0.3:
                recent_high = max(close[-5:])
                if close[-1] < recent_high * 1.02:
                    confidence = min(0.75, 0.5 + roc_10 / 8)
                    return AlphaSignal(
                        strategy=StrategyType.TREND,
                        direction=1.0,
                        confidence=confidence,
                        expected_move=atr / close[-1] * 100 * 2.5,
                        holding_period=10,
                        stop_distance=1.5,
                        entry_reason="trend_momentum_long"
                    )

            # Pullback entry in uptrend: Price pulled back to EMA20 but macro still bullish
            if close[-1] < ema20 and close[-1] > ema50 and roc_5 > -2:
                if close[-1] > close[-2]:  # Bouncing off support
                    confidence = 0.65
                    return AlphaSignal(
                        strategy=StrategyType.TREND,
                        direction=1.0,
                        confidence=confidence,
                        expected_move=atr / close[-1] * 100 * 2,
                        holding_period=8,
                        stop_distance=1.2,
                        entry_reason="trend_pullback_long"
                    )

        # SHORT: Only in macro bearish environment
        # CRYPTO: NEVER SHORT - the edge is riding trends, not fading them
        elif macro_bearish and not is_crypto:
            if close[-1] < ema20 < ema50 and roc_10 < -0.3:
                recent_low = min(close[-5:])
                if close[-1] > recent_low * 0.98:
                    confidence = min(0.75, 0.5 + abs(roc_10) / 8)
                    return AlphaSignal(
                        strategy=StrategyType.TREND,
                        direction=-1.0,
                        confidence=confidence,
                        expected_move=atr / close[-1] * 100 * 2.5,
                        holding_period=10,
                        stop_distance=1.5,
                        entry_reason="trend_momentum_short"
                    )

            # Rally entry in downtrend: Price rallied to EMA20 but macro still bearish
            if close[-1] > ema20 and close[-1] < ema50 and roc_5 < 2:
                if close[-1] < close[-2]:  # Rejecting resistance
                    confidence = 0.65
                    return AlphaSignal(
                        strategy=StrategyType.TREND,
                        direction=-1.0,
                        confidence=confidence,
                        expected_move=atr / close[-1] * 100 * 2,
                        holding_period=8,
                        stop_distance=1.2,
                        entry_reason="trend_rally_short"
                    )

        # NEUTRAL environment - no trend signal
        # Don't trade trend strategy in choppy/transitioning markets
        return None

    def _mean_reversion_strategy(self, df: pd.DataFrame, symbol: str = None) -> Optional[AlphaSignal]:
        """
        Mean reversion strategy with ASSET-AWARE BIAS.

        CRYPTO: NEVER short mean reversion. Only buy oversold dips.
        The edge in crypto is riding the trend, not fading it.
        Mean reversion shorts in crypto are consistent losers.

        FOREX/COMMODITIES: Allow both directions with trend check.
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if len(close) < 30:
            return None

        # Check if this is crypto - NEVER SHORT CRYPTO MEAN REVERSION
        is_crypto = symbol and any(c in symbol.upper() for c in ['BTC', 'ETH', 'SOL', 'XRP'])

        # Trend detection with slope-based confirmation
        ema50 = self._ema(close, 50)
        ema50_prev = self._ema(close[:-10], 50) if len(close) > 60 else ema50
        ema50_slope = (ema50 - ema50_prev) / ema50_prev * 100
        trend_up = close[-1] > ema50 and ema50_slope > -0.5
        trend_down = close[-1] < ema50 and ema50_slope < 0.5

        # Z-scores
        ma10 = np.mean(close[-10:])
        std10 = np.std(close[-10:])
        zscore_10 = (close[-1] - ma10) / std10 if std10 > 0 else 0

        # RSI
        rsi = self._rsi(close, 7)

        # ATR
        atr = self._atr(high, low, close, 14)

        # LONG: Oversold bounce - ALWAYS ALLOWED (buying dips works)
        if zscore_10 < -1.5 and rsi < 35:
            confidence = min(0.72, 0.45 + abs(zscore_10) * 0.1)

            # Boost confidence in uptrend (buying dips in bull market)
            if trend_up:
                confidence = min(0.82, confidence + 0.12)

            return AlphaSignal(
                strategy=StrategyType.MEAN_REVERSION,
                direction=1.0,
                confidence=confidence,
                expected_move=abs(zscore_10) * std10 / close[-1] * 100 * 0.5,
                holding_period=5,
                stop_distance=1.0,
                entry_reason="mean_reversion_oversold"
            )

        # SHORT: Overbought fade
        if zscore_10 > 1.5 and rsi > 65:
            # CRYPTO: NEVER SHORT MEAN REVERSION - this was killing BTCUSD
            if is_crypto:
                return None

            # NON-CRYPTO: Only short in confirmed downtrend
            if not trend_down:
                return None

            confidence = min(0.68, 0.42 + abs(zscore_10) * 0.08)

            return AlphaSignal(
                strategy=StrategyType.MEAN_REVERSION,
                direction=-1.0,
                confidence=confidence,
                expected_move=abs(zscore_10) * std10 / close[-1] * 100 * 0.5,
                holding_period=5,
                stop_distance=1.0,
                entry_reason="mean_reversion_overbought"
            )

        return None

    def _breakout_strategy(self, df: pd.DataFrame, symbol: str = None) -> Optional[AlphaSignal]:
        """
        Breakout strategy with MACRO TREND BIAS.

        CRITICAL FIX: Block counter-trend breakouts!
        - Don't buy breakouts in macro bearish environment
        - Don't short breakdowns in macro bullish environment

        CRYPTO: NEVER short breakdowns for crypto assets.
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if len(close) < 20:
            return None

        # Check if crypto - NEVER SHORT CRYPTO
        is_crypto = self._is_crypto(symbol)

        # MACRO TREND DETECTION - same as trend strategy
        ema50 = self._ema(close, 50)
        ema50_10_bars_ago = self._ema(close[:-10], 50) if len(close) > 60 else ema50
        ema50_slope = (ema50 - ema50_10_bars_ago) / ema50_10_bars_ago * 100

        macro_bullish = close[-1] > ema50 and ema50_slope > 0
        macro_bearish = close[-1] < ema50 and ema50_slope < 0

        # CHANGED: 10-bar range instead of 20 for more triggers
        lookback = 10
        range_high = max(high[-lookback:])
        range_low = min(low[-lookback:])
        range_size = (range_high - range_low) / range_low * 100

        # Volatility compression (ATR contraction) - slightly looser threshold
        atr_current = self._atr(high[-10:], low[-10:], close[-10:], 10)
        atr_prior = self._atr(high[-20:-10], low[-20:-10], close[-20:-10], 10) if len(high) >= 20 else atr_current

        vol_compression = atr_current < atr_prior * 0.85  # Was 0.8

        # Current price position
        current = close[-1]
        prev = close[-2]

        # Momentum confirmation
        momentum = (current - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0

        # LONG breakout - only if macro bullish OR neutral (not bearish)
        if current > range_high and prev <= range_high:
            # DON'T buy breakouts in downtrend - they often fail
            if macro_bearish:
                return None

            confidence = 0.55
            if vol_compression:
                confidence += 0.12  # Boost for compression breakout

            # Additional boost if breaking with momentum
            if momentum > 1.0:
                confidence += 0.08

            # Extra confidence in uptrend
            if macro_bullish:
                confidence += 0.05

            return AlphaSignal(
                strategy=StrategyType.BREAKOUT,
                direction=1.0,
                confidence=min(0.80, confidence),
                expected_move=range_size * 0.6,  # Target: 60% of range added
                holding_period=12,
                stop_distance=1.5,  # Tighter stop than before
                entry_reason="range_breakout_long"
            )

        # SHORT breakdown - only if macro bearish OR neutral (not bullish)
        if current < range_low and prev >= range_low:
            # CRYPTO: NEVER SHORT BREAKDOWNS
            if is_crypto:
                return None

            # DON'T short breakdowns in uptrend - they often bounce
            if macro_bullish:
                return None

            confidence = 0.55
            if vol_compression:
                confidence += 0.12

            # Boost for momentum breakdown
            if momentum < -1.0:
                confidence += 0.08

            # Extra confidence in downtrend
            if macro_bearish:
                confidence += 0.05

            return AlphaSignal(
                strategy=StrategyType.BREAKOUT,
                direction=-1.0,
                confidence=min(0.80, confidence),
                expected_move=range_size * 0.6,
                holding_period=12,
                stop_distance=1.5,
                entry_reason="range_breakdown_short"
            )

        return None

    def _lead_lag_strategy(
        self,
        df: pd.DataFrame,
        symbol: str,
        related_data: Dict[str, pd.DataFrame]
    ) -> Optional[AlphaSignal]:
        """
        Lead-lag strategy.

        Edge: Some assets lead others (BTC leads ETH, EUR leads GBP).
        Entry: When leader moves but follower hasn't caught up.
        """
        # Define lead-lag relationships
        relationships = {
            'BTCUSD': ['ETHUSD'],  # BTC leads ETH
            'ETHUSD': [],
            'EURUSD': ['GBPUSD'],  # EUR leads GBP
            'GBPUSD': [],
            'XAUUSD': [],
            'NAS100': ['SPX500', 'US30'],  # NAS leads other indices
            'SPX500': [],
            'US30': []
        }

        # Clean symbol
        clean_symbol = symbol.upper().replace('.X', '').replace('.x', '')

        # Get leaders for this symbol (symbols that lead this one)
        leaders = []
        for leader, followers in relationships.items():
            if clean_symbol in followers:
                leaders.append(leader)

        if not leaders:
            return None

        close = df['close'].values
        if len(close) < 20:
            return None

        # Check each leader
        for leader in leaders:
            # Try different symbol formats
            leader_keys = [leader, leader + '.x', leader + '.X', leader.lower()]
            leader_df = None
            for key in leader_keys:
                if key in related_data:
                    leader_df = related_data[key]
                    break

            if leader_df is None:
                continue

            leader_close = leader_df['close'].values

            if len(leader_close) < 20:
                continue

            # Calculate returns
            leader_ret_5 = (leader_close[-1] - leader_close[-5]) / leader_close[-5] * 100
            follower_ret_5 = (close[-1] - close[-5]) / close[-5] * 100

            # Leader moved significantly, follower lagging
            lag = leader_ret_5 - follower_ret_5

            # If leader up >1% more than follower, follower should catch up
            if lag > 1.0:
                return AlphaSignal(
                    strategy=StrategyType.LEAD_LAG,
                    direction=1.0,
                    confidence=min(0.65, 0.5 + lag * 0.05),
                    expected_move=lag * 0.5,  # Expect to capture half the lag
                    holding_period=5,
                    stop_distance=1.0,
                    entry_reason=f"lead_lag_catch_up_{leader}"
                )

            # If leader down >1% more than follower
            if lag < -1.0:
                return AlphaSignal(
                    strategy=StrategyType.LEAD_LAG,
                    direction=-1.0,
                    confidence=min(0.65, 0.5 + abs(lag) * 0.05),
                    expected_move=abs(lag) * 0.5,
                    holding_period=5,
                    stop_distance=1.0,
                    entry_reason=f"lead_lag_catch_down_{leader}"
                )

        return None

    def _combine_signals(self, signals: List[AlphaSignal], df: pd.DataFrame) -> CombinedSignal:
        """
        Combine multiple alpha signals into one trading decision.

        Key insight: Don't require all to agree.
        If one strong signal exists, take it with appropriate sizing.
        Minimum confidence threshold: 0.40 (lowered from 0.50)
        """
        if not signals:
            return CombinedSignal(
                action="neutral",
                direction=0,
                confidence=0,
                position_size=0,
                strategies_agreeing=[],
                primary_strategy="none",
                stop_loss=0,
                take_profit=0,
                expected_holding=0,
                entry_reason=""
            )

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        atr = self._atr(high, low, close, 14)

        # Calculate weighted direction
        total_weight = 0
        weighted_direction = 0

        for signal in signals:
            weight = self.strategy_weights[signal.strategy] * signal.confidence
            weighted_direction += signal.direction * weight
            total_weight += weight

        if total_weight > 0:
            final_direction = weighted_direction / total_weight
        else:
            final_direction = 0

        # Count agreeing strategies
        agreeing = []
        for signal in signals:
            if (signal.direction > 0 and final_direction > 0) or \
               (signal.direction < 0 and final_direction < 0):
                agreeing.append(signal.strategy.value)

        # Agreement affects position size but NOT whether we trade
        agreement_ratio = len(agreeing) / len(signals) if signals else 0

        # Find primary (highest confidence) signal
        primary = max(signals, key=lambda s: s.confidence)

        # Final confidence
        avg_confidence = np.mean([s.confidence for s in signals])
        # Less penalty for disagreement - we want more trades
        final_confidence = avg_confidence * (0.8 + 0.2 * agreement_ratio)

        # Position sizing
        # Base: 1.0 = full position
        # Reduce for low confidence or disagreement, but less aggressively
        position_size = final_confidence * (0.5 + 0.5 * agreement_ratio)
        position_size = max(0.4, min(1.0, position_size))  # Min 40%, max 100%

        # Determine action - LOWER threshold: 0.40 instead of 0.50
        if abs(final_direction) < 0.2 or final_confidence < 0.40:
            action = "neutral"
            position_size = 0
        elif final_direction > 0:
            action = "long"
        else:
            action = "short"

        # Calculate stops based on primary strategy
        if action == "long":
            stop_loss = close[-1] - atr * primary.stop_distance
            take_profit = close[-1] + atr * primary.stop_distance * 2.5
        elif action == "short":
            stop_loss = close[-1] + atr * primary.stop_distance
            take_profit = close[-1] - atr * primary.stop_distance * 2.5
        else:
            stop_loss = 0
            take_profit = 0

        return CombinedSignal(
            action=action,
            direction=final_direction,
            confidence=final_confidence,
            position_size=position_size,
            strategies_agreeing=agreeing,
            primary_strategy=primary.strategy.value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_holding=primary.holding_period,
            entry_reason=primary.entry_reason
        )

    def get_signal_stats(self) -> Dict[str, int]:
        """Get signal counts by strategy for debugging."""
        return {s.value: self.signal_counts[s] for s in StrategyType}

    def reset_counts(self):
        """Reset signal counts."""
        self.signal_counts = {s: 0 for s in StrategyType}

    # Helper functions
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1]
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')[-1]

    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate ATR."""
        if len(high) < period + 1:
            return high[-1] - low[-1] if len(high) > 0 else 0

        tr_list = []
        for i in range(-period, 0):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i-1]) if i > -period else h_l
            l_pc = abs(low[i] - close[i-1]) if i > -period else h_l
            tr_list.append(max(h_l, h_pc, l_pc))

        return np.mean(tr_list)

    def _rsi(self, close: np.ndarray, period: int) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50

        deltas = np.diff(close[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
