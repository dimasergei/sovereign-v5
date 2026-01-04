# signals/generator.py
"""
High-Frequency Multi-Alpha Signal Generator.

Key insight: Generate MORE signals, not fewer.
Each signal is smaller, but many compound to high returns.

Uses MultiAlphaEngine with 4 strategies:
- Trend Following
- Mean Reversion
- Breakout
- Lead-Lag

Target: 150-200 trades/year per symbol with 52% win rate and 2:1 R:R
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from strategies.multi_alpha_engine import MultiAlphaEngine, CombinedSignal
from config.trading_params import get_params

# ML Signal Adapter - optional, gracefully degrades if models not trained
try:
    from signals.ml_signal_adapter import MLSignalAdapter
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Complete trading signal with all relevant information."""
    symbol: str = ""
    action: str = "neutral"  # "long", "short", "neutral"
    direction: float = 0.0  # -1 to 1
    confidence: float = 0.0  # 0 to 1
    position_scalar: float = 1.0  # 0 to 1 (from multi-alpha)

    # Risk parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.75

    # Context
    regime: str = "multi_alpha"
    entry_reason: str = ""
    model_agreement: float = 1.0

    # Strategy info
    strategies_agreeing: List[str] = field(default_factory=list)
    primary_strategy: str = "none"
    expected_holding: int = 0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    contributing_models: List[str] = field(default_factory=list)

    # Market data
    current_price: float = 0.0
    atr: float = 0.0


class SignalGenerator:
    """
    High-frequency multi-alpha signal generator.

    Key change: Generate MORE signals, not fewer.
    Each signal is smaller, but many compound to high returns.
    """

    def __init__(
        self,
        min_confidence: float = 0.40,
        use_ml: bool = True,
        ml_mode: str = 'filter',
        ml_confidence_threshold: float = 0.5
    ):
        """
        Initialize signal generator.

        Args:
            min_confidence: Minimum confidence for a signal (default 0.40)
            use_ml: Whether to use ML models if available
            ml_mode: ML operating mode ('filter', 'confirm', 'replace')
            ml_confidence_threshold: Minimum ML confidence to act
        """
        self.multi_alpha = MultiAlphaEngine()
        self.related_data_cache: Dict[str, pd.DataFrame] = {}
        self.min_confidence = min_confidence

        # Track signal counts for monitoring
        self.signal_counts = {
            "long": 0,
            "short": 0,
            "neutral": 0
        }

        # ML Signal Adapter - optional enhancement
        self.ml_adapter = None
        self.use_ml = use_ml

        if use_ml and ML_AVAILABLE:
            try:
                self.ml_adapter = MLSignalAdapter(
                    mode=ml_mode,
                    confidence_threshold=ml_confidence_threshold,
                    load_models=True
                )
                if self.ml_adapter.is_ready():
                    logger.info(f"ML adapter initialized (mode={ml_mode})")
                else:
                    logger.info("ML adapter loaded but no models available")
            except Exception as e:
                logger.warning(f"Failed to initialize ML adapter: {e}")
                self.ml_adapter = None

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        related_data: Dict[str, pd.DataFrame] = None,
        calibration: Any = None
    ) -> TradingSignal:
        """
        Generate multi-alpha signal.

        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol
            related_data: Data for related symbols (for lead-lag strategy)
            calibration: Optional calibration (ignored, for compatibility)

        Returns:
            TradingSignal with action, confidence, and risk parameters
        """
        # Get asset-specific parameters
        params = get_params(symbol)

        if len(df) < 50:
            return TradingSignal(
                symbol=symbol,
                action="neutral",
                entry_reason="insufficient_data"
            )

        # Use cached related data if none provided
        data_for_leadlag = related_data or self.related_data_cache

        # Generate combined signal from all strategies
        combined = self.multi_alpha.generate_signals(
            df,
            symbol,
            data_for_leadlag
        )

        # Calculate ATR and current price for the signal
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        current_price = close[-1]
        atr = self._calculate_atr(high, low, close, 14)

        # Track signal counts
        self.signal_counts[combined.action] += 1

        if combined.action == "neutral":
            return TradingSignal(
                symbol=symbol,
                action="neutral",
                confidence=0.0,
                entry_reason="no_signal",
                current_price=current_price,
                atr=atr
            )

        # Create base signal from rule-based strategies
        signal = TradingSignal(
            symbol=symbol,
            action=combined.action,
            direction=combined.direction,
            confidence=combined.confidence,
            position_scalar=combined.position_size,
            stop_loss=combined.stop_loss,
            take_profit=combined.take_profit,
            entry_reason=combined.entry_reason,
            strategies_agreeing=combined.strategies_agreeing,
            primary_strategy=combined.primary_strategy,
            expected_holding=combined.expected_holding,
            current_price=current_price,
            atr=atr,
            contributing_models=combined.strategies_agreeing
        )

        # Apply ML filtering if available and enabled
        if self.ml_adapter is not None and self.ml_adapter.is_ready():
            try:
                # Prepare features for ML models
                features = self._prepare_ml_features(df)
                if features is not None:
                    signal = self.ml_adapter.filter_signal(signal, features, symbol)

                    # Update signal counts if ML vetoed
                    if signal.action == "neutral" and combined.action != "neutral":
                        self.signal_counts["neutral"] += 1
                        self.signal_counts[combined.action] -= 1
            except Exception as e:
                logger.debug(f"ML filtering failed: {e}")

        return signal

    def _prepare_ml_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare features for ML models from OHLCV data.

        Returns:
            Feature array suitable for LSTM/Transformer (sequence_length, n_features)
        """
        if len(df) < 50:
            return None

        try:
            # Use last 50 bars
            df_recent = df.tail(50).copy()

            # Calculate features
            close = df_recent['close'].values
            high = df_recent['high'].values
            low = df_recent['low'].values

            features = []

            # Returns
            returns = np.diff(close) / close[:-1]
            returns = np.concatenate([[0], returns])
            features.append(returns)

            # Volatility (rolling std of returns)
            vol = pd.Series(returns).rolling(10, min_periods=1).std().values
            features.append(vol)

            # RSI
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
            avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi = np.concatenate([[50], rsi])  # Pad first value
            features.append(rsi / 100)  # Normalize to 0-1

            # Price momentum
            mom_10 = close / np.roll(close, 10) - 1
            mom_10[:10] = 0
            features.append(mom_10)

            # ATR percentage
            tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
            tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
            atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
            atr_pct = atr / close
            features.append(atr_pct)

            # Stack features
            feature_matrix = np.column_stack(features)

            # Handle NaN/Inf
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            return feature_matrix

        except Exception as e:
            logger.debug(f"Feature preparation failed: {e}")
            return None

    def update_related_data(self, symbol: str, df: pd.DataFrame):
        """Cache related symbol data for lead-lag strategy."""
        self.related_data_cache[symbol] = df

    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        strategy_stats = self.multi_alpha.get_signal_stats()
        return {
            "signal_counts": self.signal_counts.copy(),
            "strategy_signals": strategy_stats
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.signal_counts = {
            "long": 0,
            "short": 0,
            "neutral": 0
        }
        self.multi_alpha.reset_counts()

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

        tr_list = []
        for i in range(-period, 0):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i-1]) if i > -period else h_l
            l_pc = abs(low[i] - close[i-1]) if i > -period else h_l
            tr_list.append(max(h_l, h_pc, l_pc))

        return np.mean(tr_list)

    # Legacy compatibility methods
    def get_blocked_stats(self) -> Dict[str, int]:
        """Get statistics on blocked signals (legacy compatibility)."""
        return {"no_signal": self.signal_counts["neutral"]}

    # Additional property for compatibility with old code
    @property
    def blocked_signals(self) -> Dict[str, int]:
        """Legacy compatibility for blocked_signals dict."""
        return {"no_signal": self.signal_counts["neutral"]}
