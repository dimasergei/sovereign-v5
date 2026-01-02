"""
Signal Generator - Combines all signal sources into trading decisions.
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
        
        # 4. Prepare features for models
        features = self._prepare_features(df, alt_data)
        
        # 5. Get ensemble prediction (if available)
        if self.ensemble:
            try:
                ensemble_pred = self.ensemble.predict(features)
            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")
                ensemble_pred = None
        else:
            ensemble_pred = None
        
        # 6. Combine signals
        signal = self._combine_signals(
            symbol=symbol,
            df=df,
            calibration=calibration,
            regime=regime,
            ensemble_pred=ensemble_pred,
            alt_data=alt_data
        )
        
        # 7. Apply regime filter
        signal = self._apply_regime_filter(signal, regime)
        
        return signal
    
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
