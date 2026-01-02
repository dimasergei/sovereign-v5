"""
Signal Generator - Combines all signal sources into trading decisions.

CRITICAL FIX: Now generates signals IN THE DIRECTION OF THE TREND.

Old logic: Generate signal from indicators → Filter by trend
New logic: Detect trend → Generate signals aligned with trend → Filter confirms
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from core.lossless import MarketCalibrator, FullCalibrationResult
from data import FeatureEngineer
from models import EnsembleMetaLearner, RegimeDetector, EnsemblePrediction
from signals.trend_filter import TrendFilter, TrendState, TrendDirection


logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Complete trading signal with all relevant information."""
    symbol: str
    action: str  # "long", "short", "neutral"
    direction: float  # -1 to 1
    confidence: float  # 0 to 1
    position_scalar: float  # 0 to 1

    # Risk parameters (derived from calibrator)
    stop_loss_atr_mult: float
    take_profit_atr_mult: float

    # Context
    regime: str
    model_agreement: float

    # Trend filter fields (CRITICAL FOR COUNTER-TREND PREVENTION)
    trend_direction: str = "neutral"  # strong_up, mild_up, neutral, mild_down, strong_down
    trend_strength: float = 0.0  # 0-1
    higher_tf_aligned: bool = False
    filter_reason: str = ""  # Why signal was filtered
    filters_applied: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    calibration_confidence: float = 0.0
    contributing_models: List[str] = field(default_factory=list)

    # Raw values for order placement
    current_price: float = 0.0
    atr: float = 0.0
    
    def get_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss price."""
        distance = self.atr * self.stop_loss_atr_mult
        if is_long:
            return entry_price - distance
        else:
            return entry_price + distance
    
    def get_take_profit(self, entry_price: float, is_long: bool) -> float:
        """Calculate take profit price."""
        distance = self.atr * self.take_profit_atr_mult
        if is_long:
            return entry_price + distance
        else:
            return entry_price - distance


class SignalGenerator:
    """
    Generates trading signals by combining:
    - Market calibration (lossless parameters)
    - Feature engineering
    - Ensemble model predictions
    - Regime detection
    - Alternative data (optional)
    
    Usage:
        generator = SignalGenerator(
            calibrator=calibrator,
            feature_engineer=engineer,
            ensemble=ensemble,
            regime_detector=regime
        )
        
        signal = generator.generate_signal("BTCUSD.x", df)
    """
    
    def __init__(
        self,
        calibrator: MarketCalibrator,
        feature_engineer: FeatureEngineer = None,
        ensemble: EnsembleMetaLearner = None,
        regime_detector: RegimeDetector = None,
        min_confidence: float = 0.5,
        recalibrate_hours: float = 4.0
    ):
        """
        Initialize signal generator.

        Args:
            calibrator: MarketCalibrator for parameter derivation
            feature_engineer: FeatureEngineer for indicator calculation
            ensemble: EnsembleMetaLearner for predictions
            regime_detector: RegimeDetector for regime classification
            min_confidence: Minimum confidence for non-neutral signals
            recalibrate_hours: Hours between recalibrations
        """
        self.calibrator = calibrator
        self.feature_engineer = feature_engineer
        self.ensemble = ensemble
        self.regime_detector = regime_detector or RegimeDetector()
        self.min_confidence = min_confidence
        self.recalibrate_hours = recalibrate_hours

        # CRITICAL: Add trend filter to prevent counter-trend trades
        self.trend_filter = TrendFilter()
        self.min_confidence_trending = 0.4  # Lower threshold in trends
        self.min_confidence_ranging = 0.6   # Higher threshold in ranges

        # Track blocked signals for monitoring
        self.blocked_signals = {"counter_trend": 0, "low_confidence": 0}

        # Cache
        self._calibration_cache: Dict[str, FullCalibrationResult] = {}
        self._last_calibration: Dict[str, datetime] = {}
    
    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        alt_data: Dict[str, float] = None
    ) -> TradingSignal:
        """
        Generate complete trading signal for a symbol.

        CRITICAL FIX: Generates signals IN THE DIRECTION OF THE TREND.

        Flow:
        1. Analyze trend FIRST
        2. Generate signals that align with trend direction
        3. Apply filter for confirmation (should mostly approve now)

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            alt_data: Optional alternative data signals

        Returns:
            TradingSignal with action, confidence, and risk parameters
        """
        if len(df) < 100:
            return self._neutral_signal(symbol, "Insufficient data")

        # 1. Calibrate parameters if needed
        calibration = self._get_calibration(symbol, df)

        # 2. Add features
        if self.feature_engineer:
            df = self.feature_engineer.add_all_features(df)

        # 3. Detect regime
        regime, regime_prob = self.regime_detector.detect_regime(df['close'].values)

        # 4. CRITICAL: Analyze trend FIRST - this drives signal direction
        trend_state = self.trend_filter.analyze(df)

        # 5. Get current market data
        current_price = float(df['close'].iloc[-1])
        atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else current_price * 0.02

        # 6. GENERATE TREND-ALIGNED SIGNALS (KEY FIX)
        if trend_state.is_trending:
            # In trending market: generate signals in trend direction
            raw_signal = self._generate_trend_following_signal(
                df, trend_state, regime, calibration, current_price, atr
            )
        else:
            # In ranging market: generate mean-reversion signals
            raw_signal = self._generate_mean_reversion_signal(
                df, trend_state, regime, calibration, current_price, atr
            )

        # 7. Apply trend filter for confirmation (should mostly approve now)
        filtered_action, conf_mult, filter_reason = self.trend_filter.filter_signal(
            raw_signal.action,
            trend_state,
            regime
        )

        # Track blocked signals
        if filter_reason == "counter_trend_blocked":
            self.blocked_signals["counter_trend"] += 1
            logger.info(f"BLOCKED {raw_signal.action.upper()} on {symbol} - "
                       f"counter-trend in {trend_state.direction.value}")

        # Adjust confidence threshold based on regime
        if trend_state.is_trending:
            min_conf = self.min_confidence_trending
        else:
            min_conf = self.min_confidence_ranging

        # Apply confidence adjustment
        adjusted_confidence = raw_signal.confidence * conf_mult

        # Final decision
        if filtered_action == "neutral" or adjusted_confidence < min_conf:
            if raw_signal.action != "neutral" and adjusted_confidence < min_conf:
                self.blocked_signals["low_confidence"] += 1

            return TradingSignal(
                symbol=symbol,
                action="neutral",
                direction=0.0,
                confidence=0.0,
                position_scalar=0.0,
                stop_loss_atr_mult=calibration.stop_loss_atr_multiple,
                take_profit_atr_mult=calibration.take_profit_atr_multiple,
                regime=regime,
                model_agreement=raw_signal.model_agreement,
                trend_direction=trend_state.direction.value,
                trend_strength=trend_state.strength,
                higher_tf_aligned=trend_state.higher_tf_aligned,
                filter_reason=filter_reason,
                filters_applied=["trend_aware_generation", "trend_filter"],
                calibration_confidence=calibration.confidence,
                contributing_models=raw_signal.contributing_models,
                current_price=current_price,
                atr=atr,
            )

        # Return approved signal with trend info
        return TradingSignal(
            symbol=symbol,
            action=filtered_action,
            direction=raw_signal.direction,
            confidence=adjusted_confidence,
            position_scalar=raw_signal.position_scalar * conf_mult,
            stop_loss_atr_mult=calibration.stop_loss_atr_multiple,
            take_profit_atr_mult=calibration.take_profit_atr_multiple,
            regime=regime,
            model_agreement=raw_signal.model_agreement,
            trend_direction=trend_state.direction.value,
            trend_strength=trend_state.strength,
            higher_tf_aligned=trend_state.higher_tf_aligned,
            filter_reason=filter_reason if filter_reason else raw_signal.filter_reason,
            filters_applied=["trend_aware_generation", "trend_filter"],
            calibration_confidence=calibration.confidence,
            contributing_models=raw_signal.contributing_models,
            current_price=current_price,
            atr=atr,
        )
    
    def _get_calibration(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> FullCalibrationResult:
        """Get or update calibration for symbol."""
        now = datetime.now()
        
        # Check if recalibration needed
        if symbol in self._last_calibration:
            hours_since = (now - self._last_calibration[symbol]).total_seconds() / 3600
            if hours_since < self.recalibrate_hours and symbol in self._calibration_cache:
                return self._calibration_cache[symbol]
        
        # Recalibrate
        try:
            calibration = self.calibrator.calibrate_all(df)
            self._calibration_cache[symbol] = calibration
            self._last_calibration[symbol] = now
            
            # Update feature engineer if needed
            if self.feature_engineer:
                self.feature_engineer = FeatureEngineer.from_calibration(calibration)
            
            logger.info(f"Calibrated {symbol}: regime={calibration.current_regime}")
            return calibration
            
        except Exception as e:
            logger.error(f"Calibration failed for {symbol}: {e}")
            
            # Return cached if available
            if symbol in self._calibration_cache:
                return self._calibration_cache[symbol]
            
            # Return default calibration
            return self.calibrator._calibrate_with_limited_data(df)
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        alt_data: Dict[str, float] = None
    ) -> Dict[str, np.ndarray]:
        """Prepare features for model prediction."""
        features = {}
        
        # General features for models
        feature_cols = [
            'returns', 'log_returns', 'volatility', 'atr', 'rsi',
            'macd', 'macd_hist', 'bb_pct', 'volume_ratio',
            'adx', 'trend_strength'
        ]
        
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if available_cols:
            # Use last N rows
            lookback = 50
            feature_df = df[available_cols].iloc[-lookback:].copy()
            feature_df = feature_df.fillna(0)
            
            features['general'] = feature_df.values
            
            # Specific features for different model types
            features['mean_reversion'] = df['close'].values
            features['regime'] = df['close'].values
        
        # Add alternative data
        if alt_data:
            features['alternative'] = np.array(list(alt_data.values()))
        
        return features
    
    def _combine_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        calibration: FullCalibrationResult,
        regime: str,
        ensemble_pred: Optional[EnsemblePrediction],
        alt_data: Dict[str, float] = None
    ) -> TradingSignal:
        """Combine all signal sources into final signal."""
        
        # Get current market data
        current_price = float(df['close'].iloc[-1])
        atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else current_price * 0.02
        
        # Start with ensemble prediction
        if ensemble_pred:
            direction = ensemble_pred.direction
            confidence = ensemble_pred.confidence
            position_scalar = ensemble_pred.position_size_scalar
            model_agreement = ensemble_pred.model_agreement
            contributing = ensemble_pred.contributing_models
        else:
            # Fallback to simple signals from features
            direction, confidence = self._simple_signal(df, calibration)
            position_scalar = confidence
            model_agreement = 1.0
            contributing = ['technical']
        
        # Adjust with alternative data
        if alt_data:
            alt_direction = self._alt_data_signal(alt_data)
            
            # If alt data disagrees, reduce confidence
            if np.sign(alt_direction) != np.sign(direction) and abs(alt_direction) > 0.3:
                confidence *= 0.7
                position_scalar *= 0.7
            elif np.sign(alt_direction) == np.sign(direction):
                # Agreement boosts confidence slightly
                confidence = min(1.0, confidence * 1.1)
        
        # Determine action
        if confidence < self.min_confidence or abs(direction) < 0.2:
            action = "neutral"
            position_scalar = 0.0
        else:
            action = "long" if direction > 0 else "short"
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            direction=direction,
            confidence=confidence,
            position_scalar=position_scalar,
            stop_loss_atr_mult=calibration.stop_loss_atr_multiple,
            take_profit_atr_mult=calibration.take_profit_atr_multiple,
            regime=regime,
            model_agreement=model_agreement,
            calibration_confidence=calibration.confidence,
            contributing_models=contributing,
            current_price=current_price,
            atr=atr,
        )
    
    def _simple_signal(
        self,
        df: pd.DataFrame,
        calibration: FullCalibrationResult
    ) -> tuple:
        """Generate simple signal from technical indicators."""
        direction = 0.0
        confidence = 0.0
        signals = []
        
        # RSI signal
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if rsi < calibration.oversold_threshold:
                signals.append(('rsi', 0.5, 0.6))
            elif rsi > calibration.overbought_threshold:
                signals.append(('rsi', -0.5, 0.6))
        
        # MACD signal
        if 'macd_hist' in df.columns:
            macd_hist = df['macd_hist'].iloc[-1]
            if macd_hist > 0:
                signals.append(('macd', 0.3, 0.5))
            else:
                signals.append(('macd', -0.3, 0.5))
        
        # Trend signal
        if 'ma_cross' in df.columns:
            ma_cross = df['ma_cross'].iloc[-1]
            signals.append(('trend', ma_cross * 0.4, 0.4))
        
        if signals:
            total_weight = sum(s[2] for s in signals)
            direction = sum(s[1] * s[2] for s in signals) / total_weight
            confidence = sum(s[2] for s in signals) / len(signals)
        
        return direction, confidence
    
    def _alt_data_signal(self, alt_data: Dict[str, float]) -> float:
        """Extract signal from alternative data."""
        direction = 0.0
        
        # Funding rate (contrarian)
        if 'funding_rate' in alt_data:
            funding = alt_data['funding_rate']
            direction -= funding * 2  # Negative funding = long signal
        
        # Fear & Greed (contrarian at extremes)
        if 'fear_greed' in alt_data:
            fg = alt_data['fear_greed']
            if fg < 25:  # Extreme fear = buy
                direction += 0.3
            elif fg > 75:  # Extreme greed = sell
                direction -= 0.3
        
        return np.clip(direction, -1, 1)
    
    def _apply_regime_filter(
        self,
        signal: TradingSignal,
        regime: str
    ) -> TradingSignal:
        """
        Apply regime-based filters to the signal.
        
        - In trending regime: reduce mean-reversion signals
        - In mean-reverting regime: reduce momentum signals
        """
        # Don't modify neutral signals
        if signal.action == "neutral":
            return signal
        
        # Reduce confidence in mismatched regime-strategy combinations
        if regime in ['trending_up', 'trending_down']:
            # In trending market, momentum signals are stronger
            if 'mean_reversion' in signal.contributing_models:
                signal.confidence *= 0.7
                signal.position_scalar *= 0.7
        
        elif regime == 'mean_reverting':
            # In MR market, trend signals are weaker
            if 'trend' in signal.contributing_models:
                signal.confidence *= 0.7
                signal.position_scalar *= 0.7
        
        elif regime == 'high_volatility':
            # Reduce all signals in high vol
            signal.position_scalar *= 0.5
        
        # Re-check action threshold
        if signal.confidence < self.min_confidence:
            signal.action = "neutral"
            signal.position_scalar = 0.0
        
        return signal
    
    def _neutral_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Return neutral signal."""
        logger.debug(f"Neutral signal for {symbol}: {reason}")

        return TradingSignal(
            symbol=symbol,
            action="neutral",
            direction=0.0,
            confidence=0.0,
            position_scalar=0.0,
            stop_loss_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            regime="unknown",
            model_agreement=0.0,
        )

    def _generate_trend_following_signal(
        self,
        df: pd.DataFrame,
        trend_state: TrendState,
        regime: str,
        calibration: FullCalibrationResult,
        current_price: float,
        atr: float
    ) -> TradingSignal:
        """
        Generate signals that FOLLOW the trend.

        CRITICAL: In an uptrend, we ONLY look for long entries.
        In a downtrend, we ONLY look for short entries.
        """
        close = df['close'].values

        # Determine which direction we're looking for
        if trend_state.direction in [TrendDirection.STRONG_UP, TrendDirection.MILD_UP]:
            looking_for = "long"
        elif trend_state.direction in [TrendDirection.STRONG_DOWN, TrendDirection.MILD_DOWN]:
            looking_for = "short"
        else:
            return self._create_signal_result(
                action="neutral", confidence=0.0, direction=0.0,
                regime=regime, calibration=calibration,
                current_price=current_price, atr=atr,
                entry_reason="no_clear_trend"
            )

        # Calculate entry timing indicators
        ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        ma50 = pd.Series(close).rolling(50).mean().iloc[-1]

        pullback_to_ma20 = (current_price - ma20) / ma20 * 100 if ma20 > 0 else 0
        pullback_to_ma50 = (current_price - ma50) / ma50 * 100 if ma50 > 0 else 0

        # RSI for oversold/overbought
        rsi = self._calculate_rsi(close, 14)

        # MACD momentum confirmation
        fast_ma = pd.Series(close).ewm(span=12).mean().values
        slow_ma = pd.Series(close).ewm(span=26).mean().values
        macd = fast_ma - slow_ma
        macd_signal = pd.Series(macd).ewm(span=9).mean().values
        macd_histogram = macd[-1] - macd_signal[-1]

        # LONG ENTRY CONDITIONS (in uptrend)
        if looking_for == "long":
            confidence = 0.0
            entry_reason = None

            # Condition A: Pullback to MA20 in uptrend (high probability)
            if -3 < pullback_to_ma20 < 0 and trend_state.slope_ma50 > 0:
                confidence = 0.7
                entry_reason = "pullback_to_ma20"

            # Condition B: RSI oversold in uptrend (mean reversion within trend)
            elif rsi < 40 and trend_state.price_vs_ma50 > 0:
                confidence = 0.65
                entry_reason = "rsi_oversold_in_uptrend"

            # Condition C: MACD bullish crossover
            elif len(macd) > 1 and macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
                confidence = 0.6
                entry_reason = "macd_bullish_crossover"

            # Condition D: Strong trend continuation (price making new highs)
            elif current_price >= max(close[-20:]) and trend_state.strength > 0.5:
                confidence = 0.55
                entry_reason = "breakout_continuation"

            # Condition E: Price above all MAs with positive momentum
            elif current_price > ma20 > ma50 and macd_histogram > 0:
                confidence = 0.5
                entry_reason = "aligned_momentum"

            if confidence > 0:
                # Boost confidence if higher TF aligned
                if trend_state.higher_tf_aligned:
                    confidence = min(0.95, confidence + 0.15)

                return self._create_signal_result(
                    action="long", confidence=confidence, direction=0.7,
                    regime=regime, calibration=calibration,
                    current_price=current_price, atr=atr,
                    entry_reason=entry_reason,
                    trend_direction=trend_state.direction.value,
                    trend_strength=trend_state.strength
                )

        # SHORT ENTRY CONDITIONS (in downtrend)
        elif looking_for == "short":
            confidence = 0.0
            entry_reason = None

            # Condition A: Rally to MA20 in downtrend
            if 0 < pullback_to_ma20 < 3 and trend_state.slope_ma50 < 0:
                confidence = 0.7
                entry_reason = "rally_to_ma20"

            # Condition B: RSI overbought in downtrend
            elif rsi > 60 and trend_state.price_vs_ma50 < 0:
                confidence = 0.65
                entry_reason = "rsi_overbought_in_downtrend"

            # Condition C: MACD bearish crossover
            elif len(macd) > 1 and macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                confidence = 0.6
                entry_reason = "macd_bearish_crossover"

            # Condition D: Breakdown continuation
            elif current_price <= min(close[-20:]) and trend_state.strength > 0.5:
                confidence = 0.55
                entry_reason = "breakdown_continuation"

            # Condition E: Price below all MAs with negative momentum
            elif current_price < ma20 < ma50 and macd_histogram < 0:
                confidence = 0.5
                entry_reason = "aligned_momentum"

            if confidence > 0:
                if trend_state.higher_tf_aligned:
                    confidence = min(0.95, confidence + 0.15)

                return self._create_signal_result(
                    action="short", confidence=confidence, direction=-0.7,
                    regime=regime, calibration=calibration,
                    current_price=current_price, atr=atr,
                    entry_reason=entry_reason,
                    trend_direction=trend_state.direction.value,
                    trend_strength=trend_state.strength
                )

        # No entry condition met
        return self._create_signal_result(
            action="neutral", confidence=0.0, direction=0.0,
            regime=regime, calibration=calibration,
            current_price=current_price, atr=atr,
            entry_reason="no_entry_condition"
        )

    def _generate_mean_reversion_signal(
        self,
        df: pd.DataFrame,
        trend_state: TrendState,
        regime: str,
        calibration: FullCalibrationResult,
        current_price: float,
        atr: float
    ) -> TradingSignal:
        """
        Generate mean-reversion signals for ranging markets.

        Only used when trend is NEUTRAL and market is mean-reverting.
        """
        close = df['close'].values

        # Z-score calculation
        lookback = 50
        if len(close) < lookback:
            lookback = len(close)

        mean = np.mean(close[-lookback:])
        std = np.std(close[-lookback:])

        if std == 0:
            return self._create_signal_result(
                action="neutral", confidence=0.0, direction=0.0,
                regime=regime, calibration=calibration,
                current_price=current_price, atr=atr,
                entry_reason="no_volatility"
            )

        zscore = (close[-1] - mean) / std

        # RSI
        rsi = self._calculate_rsi(close, 14)

        # Mean reversion entries - LONG on oversold
        if zscore < -2.0 and rsi < 30:
            confidence = min(0.75, 0.5 + abs(zscore) * 0.1)
            return self._create_signal_result(
                action="long", confidence=confidence, direction=0.5,
                regime=regime, calibration=calibration,
                current_price=current_price, atr=atr,
                entry_reason="oversold_mean_reversion"
            )

        # Mean reversion entries - SHORT on overbought
        elif zscore > 2.0 and rsi > 70:
            confidence = min(0.75, 0.5 + abs(zscore) * 0.1)
            return self._create_signal_result(
                action="short", confidence=confidence, direction=-0.5,
                regime=regime, calibration=calibration,
                current_price=current_price, atr=atr,
                entry_reason="overbought_mean_reversion"
            )

        return self._create_signal_result(
            action="neutral", confidence=0.0, direction=0.0,
            regime=regime, calibration=calibration,
            current_price=current_price, atr=atr,
            entry_reason="no_extreme"
        )

    def _create_signal_result(
        self,
        action: str,
        confidence: float,
        direction: float,
        regime: str,
        calibration: FullCalibrationResult,
        current_price: float,
        atr: float,
        entry_reason: str = "",
        trend_direction: str = "neutral",
        trend_strength: float = 0.0
    ) -> TradingSignal:
        """Helper to create TradingSignal with common fields."""
        return TradingSignal(
            symbol="",  # Will be set by caller
            action=action,
            direction=direction,
            confidence=confidence,
            position_scalar=confidence,
            stop_loss_atr_mult=calibration.stop_loss_atr_multiple,
            take_profit_atr_mult=calibration.take_profit_atr_multiple,
            regime=regime,
            model_agreement=1.0,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            filter_reason=entry_reason,
            contributing_models=["trend_following"] if action != "neutral" else [],
            current_price=current_price,
            atr=atr,
        )

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
