# crypto/crypto_strategy.py
"""
Crypto Strategy Engine for The5ers Account.

High-conviction entries only. Quality over quantity.
Combines regime detection, liquidity hunting, and multi-timeframe analysis.

Key Principles:
- Only LONG positions (no shorting crypto)
- Regime filtering (avoid RANGING_VOLATILE)
- Multi-timeframe confluence required
- Conservative position sizing
- The5ers compliance integrated
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging

from .regime_detector import CryptoRegimeDetector, CryptoRegime
from .liquidity_hunter import LiquidityHuntDetector, LiquidityGrabType
from .crypto_position_sizer import CryptoPositionSizer

logger = logging.getLogger(__name__)


class CryptoSignalType(Enum):
    """Type of crypto signal generated."""
    TREND_CONTINUATION = "trend_continuation"
    PULLBACK_ENTRY = "pullback_entry"
    LIQUIDITY_REVERSAL = "liquidity_reversal"
    RANGE_FADE = "range_fade"
    BREAKOUT = "breakout"
    NONE = "none"


@dataclass
class CryptoSignal:
    """Complete crypto trading signal."""
    symbol: str
    action: str                    # "long" or "neutral"
    signal_type: CryptoSignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float              # 0 to 1
    regime: CryptoRegime
    position_size: float           # In units (BTC, ETH)
    position_usd: float            # In USD
    risk_amount: float             # Dollar risk
    risk_pct: float                # Account risk %
    confirmations: List[str]       # Checks passed
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class CryptoStrategyEngine:
    """
    Sophisticated crypto strategy engine for The5ers.

    Combines:
    - Regime detection (avoid bad regimes)
    - Multi-timeframe analysis (4H + 1H alignment)
    - Liquidity hunt detection
    - Trend following in good regimes
    - Mean reversion in quiet ranges
    """

    # Entry requirements
    MIN_CONFLUENCE_SCORE = 0.70    # 70% of checks must pass
    MIN_RR_RATIO = 2.0             # Minimum reward:risk
    MIN_CONFIDENCE = 0.60          # Minimum signal confidence

    # Required confirmations for entry
    REQUIRED_CONFIRMATIONS = [
        "regime_favorable",           # Not RANGING_VOLATILE
        "mtf_alignment",              # 4H + 1H agree on direction
        "drawdown_headroom",          # >50% of daily DD remaining
        "volume_confirmation",        # Above average volume
        "not_extended",               # RSI not at extreme (unless divergence)
    ]

    # Technical parameters
    EMA_FAST = 20
    EMA_SLOW = 50
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    ATR_PERIOD = 14
    ATR_STOP_MULT = 2.0            # Wider stops for crypto
    ATR_TARGET_MULT = 4.0          # Larger targets

    def __init__(self, account_balance: float = 5000.0):
        """
        Initialize crypto strategy engine.

        Args:
            account_balance: The5ers account balance
        """
        self.regime_detector = CryptoRegimeDetector()
        self.liquidity_hunter = LiquidityHuntDetector()
        self.position_sizer = CryptoPositionSizer(account_balance)

        # Performance tracking
        self.signals_generated = 0
        self.signals_taken = 0
        self.last_signal_time: Dict[str, datetime] = {}

    def generate_signal(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        symbol: str,
        current_equity: Optional[float] = None
    ) -> CryptoSignal:
        """
        Generate trading signal for crypto symbol.

        Args:
            df_1h: 1-hour OHLCV data (primary timeframe)
            df_4h: 4-hour OHLCV data (trend confirmation)
            symbol: Trading symbol (BTCUSD, ETHUSD)
            current_equity: Current account equity

        Returns:
            CryptoSignal with full trade details
        """
        self.signals_generated += 1

        if current_equity:
            self.position_sizer.update_balance(current_equity)

        # Get current price
        current_price = df_1h['close'].iloc[-1]

        # Create neutral signal as default
        neutral_signal = CryptoSignal(
            symbol=symbol,
            action="neutral",
            signal_type=CryptoSignalType.NONE,
            entry_price=current_price,
            stop_loss=0,
            take_profit=0,
            confidence=0,
            regime=CryptoRegime.UNKNOWN,
            position_size=0,
            position_usd=0,
            risk_amount=0,
            risk_pct=0,
            confirmations=[],
            reason="No valid signal"
        )

        # Step 1: Regime Detection
        regime_analysis = self.regime_detector.detect_regime(df_4h)

        if not regime_analysis.should_trade:
            neutral_signal.regime = regime_analysis.regime
            neutral_signal.reason = regime_analysis.reason
            return neutral_signal

        # Step 2: Multi-Timeframe Analysis
        mtf_result = self._check_mtf_alignment(df_1h, df_4h)
        if not mtf_result["aligned"]:
            neutral_signal.regime = regime_analysis.regime
            neutral_signal.reason = f"MTF not aligned: {mtf_result['reason']}"
            return neutral_signal

        # Step 3: Run all confirmation checks
        confirmations = self._run_confirmations(df_1h, df_4h, regime_analysis)
        confluence_score = len(confirmations) / len(self.REQUIRED_CONFIRMATIONS)

        if confluence_score < self.MIN_CONFLUENCE_SCORE:
            neutral_signal.regime = regime_analysis.regime
            neutral_signal.confirmations = confirmations
            neutral_signal.reason = f"Low confluence: {confluence_score:.0%} (need {self.MIN_CONFLUENCE_SCORE:.0%})"
            return neutral_signal

        # Step 4: Generate Entry Based on Regime
        entry_result = self._generate_entry(df_1h, regime_analysis, mtf_result)

        if entry_result is None:
            neutral_signal.regime = regime_analysis.regime
            neutral_signal.confirmations = confirmations
            neutral_signal.reason = "No valid entry pattern"
            return neutral_signal

        # Step 5: Check R:R ratio
        rr_ratio = abs(entry_result["target"] - entry_result["entry"]) / abs(entry_result["entry"] - entry_result["stop"])
        if rr_ratio < self.MIN_RR_RATIO:
            neutral_signal.regime = regime_analysis.regime
            neutral_signal.confirmations = confirmations
            neutral_signal.reason = f"R:R too low: {rr_ratio:.1f} (need {self.MIN_RR_RATIO})"
            return neutral_signal

        # Step 6: Position Sizing
        position_result = self.position_sizer.calculate_position(
            entry_price=entry_result["entry"],
            stop_loss=entry_result["stop"],
            symbol=symbol,
            df=df_1h,
            current_equity=current_equity
        )

        if not position_result.approved:
            neutral_signal.regime = regime_analysis.regime
            neutral_signal.confirmations = confirmations
            neutral_signal.reason = f"Position rejected: {position_result.reason}"
            return neutral_signal

        # Step 7: Final confidence calculation
        confidence = self._calculate_confidence(
            regime_analysis,
            confluence_score,
            entry_result,
            rr_ratio
        )

        if confidence < self.MIN_CONFIDENCE:
            neutral_signal.regime = regime_analysis.regime
            neutral_signal.confirmations = confirmations
            neutral_signal.confidence = confidence
            neutral_signal.reason = f"Low confidence: {confidence:.0%}"
            return neutral_signal

        # Create final signal
        self.signals_taken += 1
        self.last_signal_time[symbol] = datetime.now()

        return CryptoSignal(
            symbol=symbol,
            action="long",
            signal_type=entry_result["type"],
            entry_price=entry_result["entry"],
            stop_loss=entry_result["stop"],
            take_profit=entry_result["target"],
            confidence=confidence,
            regime=regime_analysis.regime,
            position_size=position_result.size,
            position_usd=position_result.size_usd,
            risk_amount=position_result.risk_amount,
            risk_pct=position_result.risk_pct,
            confirmations=confirmations,
            reason=entry_result["reason"]
        )

    def _check_mtf_alignment(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Dict:
        """
        Check multi-timeframe alignment.

        4H trend must agree with 1H for entry.
        """
        # 4H trend
        close_4h = df_4h['close'].values
        ema_fast_4h = self._ema(close_4h, self.EMA_FAST)
        ema_slow_4h = self._ema(close_4h, self.EMA_SLOW)
        trend_4h = "bullish" if ema_fast_4h > ema_slow_4h else "bearish"

        # 1H trend
        close_1h = df_1h['close'].values
        ema_fast_1h = self._ema(close_1h, self.EMA_FAST)
        ema_slow_1h = self._ema(close_1h, self.EMA_SLOW)
        trend_1h = "bullish" if ema_fast_1h > ema_slow_1h else "bearish"

        aligned = trend_4h == trend_1h

        return {
            "aligned": aligned,
            "trend_4h": trend_4h,
            "trend_1h": trend_1h,
            "reason": f"4H: {trend_4h}, 1H: {trend_1h}" + (" - ALIGNED" if aligned else " - NOT ALIGNED")
        }

    def _run_confirmations(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        regime_analysis
    ) -> List[str]:
        """Run all confirmation checks."""
        confirmations = []

        # Check 1: Regime favorable
        if regime_analysis.regime != CryptoRegime.RANGING_VOLATILE:
            confirmations.append("regime_favorable")

        # Check 2: MTF alignment (already checked, but include for completeness)
        close_4h = df_4h['close'].values
        close_1h = df_1h['close'].values
        ema_4h = self._ema(close_4h, self.EMA_SLOW)
        ema_1h = self._ema(close_1h, self.EMA_SLOW)
        if (close_4h[-1] > ema_4h) == (close_1h[-1] > ema_1h):
            confirmations.append("mtf_alignment")

        # Check 3: Drawdown headroom
        stats = self.position_sizer.get_stats()
        if stats["daily_dd_room"] > 0.02:  # More than 2% room
            confirmations.append("drawdown_headroom")

        # Check 4: Volume confirmation
        if 'volume' in df_1h.columns:
            avg_volume = df_1h['volume'].iloc[-20:-1].mean()
            current_volume = df_1h['volume'].iloc[-1]
            if current_volume > avg_volume:
                confirmations.append("volume_confirmation")
        else:
            # If no volume data, pass this check
            confirmations.append("volume_confirmation")

        # Check 5: Not extended (RSI)
        rsi = self._calculate_rsi(close_1h, self.RSI_PERIOD)
        if self.RSI_OVERSOLD < rsi < self.RSI_OVERBOUGHT:
            confirmations.append("not_extended")
        elif rsi <= self.RSI_OVERSOLD:
            # Oversold is OK for long entry
            confirmations.append("not_extended")

        return confirmations

    def _generate_entry(
        self,
        df_1h: pd.DataFrame,
        regime_analysis,
        mtf_result: Dict
    ) -> Optional[Dict]:
        """
        Generate entry parameters based on regime and conditions.

        Only generates LONG entries (no shorting crypto).
        """
        close = df_1h['close'].values
        high = df_1h['high'].values
        low = df_1h['low'].values

        current_price = close[-1]
        atr = self._calculate_atr(high, low, close, self.ATR_PERIOD)

        # Only consider long entries in bullish alignment
        if mtf_result["trend_4h"] != "bullish":
            return None

        # Strategy selection based on regime
        if regime_analysis.regime == CryptoRegime.TRENDING_VOLATILE:
            # Breakout continuation
            return self._breakout_entry(df_1h, atr)

        elif regime_analysis.regime == CryptoRegime.TRENDING_QUIET:
            # Pullback entry
            return self._pullback_entry(df_1h, atr)

        elif regime_analysis.regime == CryptoRegime.RANGING_QUIET:
            # Range fade at support
            return self._range_fade_entry(df_1h, atr)

        # Check for liquidity grab opportunity
        liquidity_grab = self.liquidity_hunter.detect_liquidity_grab(df_1h)
        if liquidity_grab and liquidity_grab.grab_type == LiquidityGrabType.STOP_HUNT_LOWS:
            return self._liquidity_reversal_entry(df_1h, liquidity_grab, atr)

        return None

    def _breakout_entry(self, df: pd.DataFrame, atr: float) -> Optional[Dict]:
        """Breakout continuation entry."""
        close = df['close'].values
        high = df['high'].values

        current_price = close[-1]
        recent_high = max(high[-20:])

        # Only enter if price just broke above recent high
        if current_price > recent_high and close[-2] <= high[-21]:
            entry = current_price
            stop = entry - atr * self.ATR_STOP_MULT
            target = entry + atr * self.ATR_TARGET_MULT

            return {
                "type": CryptoSignalType.BREAKOUT,
                "entry": entry,
                "stop": stop,
                "target": target,
                "reason": f"Breakout above {recent_high:.2f} in trending market"
            }

        return None

    def _pullback_entry(self, df: pd.DataFrame, atr: float) -> Optional[Dict]:
        """Pullback entry in trend."""
        close = df['close'].values

        current_price = close[-1]
        ema_fast = self._ema(close, self.EMA_FAST)
        ema_slow = self._ema(close, self.EMA_SLOW)

        # Price pulled back to EMA20 but above EMA50
        if current_price > ema_slow and current_price <= ema_fast * 1.01:
            # Bouncing off EMA20
            if close[-1] > close[-2]:
                entry = current_price
                stop = entry - atr * self.ATR_STOP_MULT
                target = entry + atr * self.ATR_TARGET_MULT

                return {
                    "type": CryptoSignalType.PULLBACK_ENTRY,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "reason": f"Pullback to EMA20 ({ema_fast:.2f}) with bounce"
                }

        return None

    def _range_fade_entry(self, df: pd.DataFrame, atr: float) -> Optional[Dict]:
        """Range fade at support."""
        close = df['close'].values
        low = df['low'].values

        current_price = close[-1]
        range_low = min(low[-20:])
        range_high = max(df['high'].values[-20:])

        # Near range low (within 1 ATR)
        if current_price - range_low < atr:
            rsi = self._calculate_rsi(close, self.RSI_PERIOD)

            # Oversold bounce
            if rsi < 35 and close[-1] > close[-2]:
                entry = current_price
                stop = range_low - atr * 0.5  # Tight stop below range
                target = range_low + (range_high - range_low) * 0.7  # Target 70% of range

                return {
                    "type": CryptoSignalType.RANGE_FADE,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "reason": f"Range fade at support {range_low:.2f}, RSI={rsi:.1f}"
                }

        return None

    def _liquidity_reversal_entry(self, df: pd.DataFrame, grab, atr: float) -> Optional[Dict]:
        """Entry after liquidity grab."""
        entry = grab.close_price
        stop = grab.wick_low * 0.998  # Below the wick
        target = entry + atr * self.ATR_TARGET_MULT

        return {
            "type": CryptoSignalType.LIQUIDITY_REVERSAL,
            "entry": entry,
            "stop": stop,
            "target": target,
            "reason": grab.reason
        }

    def _calculate_confidence(
        self,
        regime_analysis,
        confluence_score: float,
        entry_result: Dict,
        rr_ratio: float
    ) -> float:
        """Calculate overall signal confidence."""
        # Base from regime
        confidence = regime_analysis.confidence * 0.3

        # Confluence score
        confidence += confluence_score * 0.3

        # R:R bonus
        rr_bonus = min(0.2, (rr_ratio - self.MIN_RR_RATIO) * 0.1)
        confidence += rr_bonus

        # Signal type bonus
        type_bonuses = {
            CryptoSignalType.PULLBACK_ENTRY: 0.1,
            CryptoSignalType.BREAKOUT: 0.1,
            CryptoSignalType.LIQUIDITY_REVERSAL: 0.15,
            CryptoSignalType.RANGE_FADE: 0.05,
        }
        confidence += type_bonuses.get(entry_result["type"], 0)

        return min(1.0, confidence)

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1]
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')[-1]

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate ATR."""
        if len(high) < period + 1:
            return np.mean(high - low)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        return np.mean(tr[-period:])

    def _calculate_rsi(self, close: np.ndarray, period: int) -> float:
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

    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            "signals_generated": self.signals_generated,
            "signals_taken": self.signals_taken,
            "take_rate": self.signals_taken / self.signals_generated if self.signals_generated > 0 else 0,
            "regime_stats": self.regime_detector.get_regime_stats(),
            "position_stats": self.position_sizer.get_stats()
        }
