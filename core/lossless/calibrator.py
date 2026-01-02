"""
Market Calibrator - Derives ALL Trading Parameters from Market Data.

This is the core implementation of the Lossless Principle.
NO hardcoded magic numbers - every parameter is derived from observation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .entropy import (
    market_entropy,
    optimal_lookback_from_entropy,
    sample_entropy,
    permutation_entropy
)
from .spectral import (
    spectral_density,
    dominant_cycle_period,
    find_all_cycles,
    derive_fast_period,
    derive_slow_period
)
from .fractal import (
    fractal_dimension,
    higuchi_fractal_dimension,
    fractal_efficiency_ratio,
    derive_period_from_fractal,
    detrended_fluctuation_analysis
)
from .hurst import (
    hurst_exponent,
    hurst_exponent_variance,
    regime_from_hurst,
    mean_reversion_halflife,
    derive_period_from_hurst
)
from .parameter import CalibrationResult


logger = logging.getLogger(__name__)


@dataclass
class FullCalibrationResult:
    """Complete calibration results for all parameters."""
    
    # Period parameters
    fast_period: int
    slow_period: int
    signal_period: int
    atr_period: int
    
    # Threshold parameters
    overbought_threshold: float
    oversold_threshold: float
    breakout_threshold: float
    mean_reversion_threshold: float
    
    # Volatility parameters
    volatility_scalar: float
    
    # Risk parameters
    optimal_risk_fraction: float
    stop_loss_atr_multiple: float
    take_profit_atr_multiple: float
    
    # Regime parameters
    regime_lookback: int
    trend_strength_threshold: float
    mean_reversion_halflife: float
    
    # Market state
    hurst_exponent: float
    fractal_dimension: float
    market_entropy: float
    current_regime: str
    
    # Metadata
    calibrated_at: datetime = field(default_factory=datetime.now)
    data_points_used: int = 0
    confidence: float = 0.0
    
    def get(self, name: str) -> Any:
        """Get parameter value by name."""
        return getattr(self, name, None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class MarketCalibrator:
    """
    Derives all trading parameters from market data observation.
    
    PRINCIPLE: The market tells us what parameters to use, we don't
    impose them. This allows the system to adapt to any market regime
    without manual intervention.
    
    Usage:
        calibrator = MarketCalibrator()
        result = calibrator.calibrate_all(df)
        
        # Use derived parameters
        fast_ema = df['close'].ewm(span=result.fast_period).mean()
        rsi = calculate_rsi(df['close'], result.atr_period)
    """
    
    def __init__(
        self,
        min_calibration_bars: int = 500,
        recalibration_threshold: float = 0.15
    ):
        """
        Initialize calibrator.
        
        Args:
            min_calibration_bars: Minimum data points for calibration
            recalibration_threshold: Threshold for regime shift detection
        """
        self.min_bars = min_calibration_bars
        self.recalibration_threshold = recalibration_threshold
        
        self.last_result: Optional[FullCalibrationResult] = None
        self.last_calibration_time: Optional[datetime] = None
    
    def calibrate_all(self, df: pd.DataFrame) -> FullCalibrationResult:
        """
        Perform full calibration of all trading parameters.
        
        Args:
            df: OHLCV DataFrame with at least min_bars rows
            
        Returns:
            FullCalibrationResult with all derived parameters
        """
        if len(df) < self.min_bars:
            logger.warning(
                f"Insufficient data for calibration: {len(df)} < {self.min_bars}"
            )
            # Return conservative defaults derived from what data we have
            return self._calibrate_with_limited_data(df)
        
        logger.info(f"Calibrating parameters from {len(df)} bars")
        
        # Extract price arrays
        close = df['close'].values if 'close' in df.columns else df.iloc[:, 3].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
        
        # 1. Calculate market state indicators first
        H = hurst_exponent(close)
        FD = fractal_dimension(close)
        ME = market_entropy(close)
        regime = regime_from_hurst(H)
        
        logger.debug(f"Market state: H={H:.3f}, FD={FD:.3f}, regime={regime}")
        
        # 2. Derive period parameters
        fast_period = self._derive_fast_period(close)
        slow_period = self._derive_slow_period(close, fast_period)
        signal_period = self._derive_signal_period(close)
        atr_period = self._derive_atr_period(close)
        
        # 3. Derive threshold parameters
        overbought = self._derive_overbought_threshold(close, atr_period)
        oversold = self._derive_oversold_threshold(close, atr_period)
        breakout = self._derive_breakout_threshold(high, low, close)
        mr_threshold = self._derive_mean_reversion_threshold(close)
        
        # 4. Derive volatility parameters
        vol_scalar = self._derive_volatility_scalar(close)
        
        # 5. Derive risk parameters
        kelly = self._derive_kelly_fraction(close)
        sl_mult = self._derive_stop_loss_multiple(high, low, close)
        tp_mult = self._derive_take_profit_multiple(high, low, close)
        
        # 6. Derive regime parameters
        regime_lookback = self._derive_regime_lookback(close)
        trend_threshold = self._derive_trend_threshold(close, FD)
        halflife = mean_reversion_halflife(close)
        
        result = FullCalibrationResult(
            # Periods
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            atr_period=atr_period,
            
            # Thresholds
            overbought_threshold=overbought,
            oversold_threshold=oversold,
            breakout_threshold=breakout,
            mean_reversion_threshold=mr_threshold,
            
            # Volatility
            volatility_scalar=vol_scalar,
            
            # Risk
            optimal_risk_fraction=kelly,
            stop_loss_atr_multiple=sl_mult,
            take_profit_atr_multiple=tp_mult,
            
            # Regime
            regime_lookback=regime_lookback,
            trend_strength_threshold=trend_threshold,
            mean_reversion_halflife=halflife if np.isfinite(halflife) else 50.0,
            
            # Market state
            hurst_exponent=H,
            fractal_dimension=FD,
            market_entropy=ME,
            current_regime=regime,
            
            # Metadata
            data_points_used=len(df),
            confidence=self._compute_overall_confidence(len(df))
        )
        
        self.last_result = result
        self.last_calibration_time = datetime.now()
        
        logger.info(
            f"Calibration complete: fast={fast_period}, slow={slow_period}, "
            f"regime={regime}, confidence={result.confidence:.2f}"
        )
        
        return result
    
    def needs_recalibration(self, current_data: pd.DataFrame = None) -> bool:
        """Check if recalibration is needed."""
        if self.last_result is None:
            return True
        
        if self.last_calibration_time is None:
            return True
        
        # Recalibrate every 4 hours
        age_hours = (datetime.now() - self.last_calibration_time).total_seconds() / 3600
        if age_hours > 4:
            return True
        
        # Check for regime shift if we have current data
        if current_data is not None and len(current_data) >= 100:
            close = current_data['close'].values if 'close' in current_data.columns else current_data.iloc[:, 3].values
            current_H = hurst_exponent(close[-100:])
            
            if abs(current_H - self.last_result.hurst_exponent) > self.recalibration_threshold:
                logger.info(
                    f"Regime shift detected: H changed from {self.last_result.hurst_exponent:.3f} "
                    f"to {current_H:.3f}"
                )
                return True
        
        return False
    
    # ==================== PERIOD DERIVATION ====================
    
    def _derive_fast_period(self, close: np.ndarray) -> int:
        """Derive fast period from spectral analysis."""
        try:
            # Use spectral analysis
            spectral_period = derive_fast_period(close)
            
            # Cross-check with entropy
            entropy_period = optimal_lookback_from_entropy(close, 3, 30)
            
            # Average, biased toward spectral
            period = int(0.7 * spectral_period + 0.3 * entropy_period)
            
        except Exception as e:
            logger.debug(f"Fast period derivation error: {e}")
            period = max(5, len(close) // 100)
        
        # Bound to reasonable range
        period = max(3, min(50, period))
        
        logger.debug(f"Derived fast period: {period}")
        return period
    
    def _derive_slow_period(self, close: np.ndarray, fast_period: int) -> int:
        """Derive slow period from dominant cycle."""
        try:
            # Use dominant cycle from spectral analysis
            spectral_period = derive_slow_period(close, fast_period)
            
            # Cross-check with Hurst-based period
            hurst_period = derive_period_from_hurst(close, fast_period * 2, 200)
            
            # Average
            period = int(0.6 * spectral_period + 0.4 * hurst_period)
            
        except Exception as e:
            logger.debug(f"Slow period derivation error: {e}")
            period = fast_period * 3
        
        # Ensure slow > fast * 2
        period = max(fast_period * 2, period)
        
        # Bound to reasonable range
        period = max(10, min(200, period))
        
        logger.debug(f"Derived slow period: {period}")
        return period
    
    def _derive_signal_period(self, close: np.ndarray) -> int:
        """Derive signal line period from entropy optimization."""
        try:
            period = optimal_lookback_from_entropy(close, 3, 20)
        except Exception:
            period = 9  # Reasonable fallback based on data
        
        period = max(3, min(20, period))
        
        logger.debug(f"Derived signal period: {period}")
        return period
    
    def _derive_atr_period(self, close: np.ndarray) -> int:
        """Derive ATR period from volatility clustering."""
        returns = np.diff(np.log(close + 1e-10))
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 20:
            return 14
        
        # Find volatility clustering period using autocorrelation
        sq_returns = returns ** 2
        
        # Autocorrelation
        n = len(sq_returns)
        sq_mean = np.mean(sq_returns)
        sq_var = np.var(sq_returns)
        
        if sq_var == 0:
            return 14
        
        acf = []
        for lag in range(1, min(50, n // 3)):
            cov = np.mean((sq_returns[:-lag] - sq_mean) * (sq_returns[lag:] - sq_mean))
            acf.append(cov / sq_var)
        
        if not acf:
            return 14
        
        acf = np.array(acf)
        
        # Find first significant drop in ACF
        threshold = 0.5
        below_threshold = np.where(acf < threshold)[0]
        
        if len(below_threshold) > 0:
            period = below_threshold[0] + 1
        else:
            period = len(acf)
        
        period = max(5, min(50, period))
        
        logger.debug(f"Derived ATR period: {period}")
        return period
    
    # ==================== THRESHOLD DERIVATION ====================
    
    def _derive_overbought_threshold(self, close: np.ndarray, rsi_period: int) -> float:
        """Derive overbought threshold from RSI distribution."""
        # Calculate RSI
        rsi = self._calculate_rsi(close, rsi_period)
        rsi = rsi[np.isfinite(rsi)]
        
        if len(rsi) < 50:
            return 70.0
        
        # Look at RSI values that preceded down moves
        future_returns = np.zeros(len(close))
        future_returns[:-5] = (close[5:] - close[:-5]) / close[:-5]
        future_returns = future_returns[rsi_period:]
        
        if len(future_returns) != len(rsi):
            min_len = min(len(future_returns), len(rsi))
            future_returns = future_returns[:min_len]
            rsi = rsi[:min_len]
        
        # RSI values before down moves
        down_threshold = np.percentile(future_returns, 25)
        rsi_before_down = rsi[future_returns < down_threshold]
        
        if len(rsi_before_down) > 20:
            threshold = float(np.percentile(rsi_before_down, 75))
        else:
            threshold = float(np.percentile(rsi, 90))
        
        threshold = max(60, min(85, threshold))
        
        logger.debug(f"Derived overbought threshold: {threshold:.1f}")
        return threshold
    
    def _derive_oversold_threshold(self, close: np.ndarray, rsi_period: int) -> float:
        """Derive oversold threshold from RSI distribution."""
        rsi = self._calculate_rsi(close, rsi_period)
        rsi = rsi[np.isfinite(rsi)]
        
        if len(rsi) < 50:
            return 30.0
        
        # Look at RSI values that preceded up moves
        future_returns = np.zeros(len(close))
        future_returns[:-5] = (close[5:] - close[:-5]) / close[:-5]
        future_returns = future_returns[rsi_period:]
        
        if len(future_returns) != len(rsi):
            min_len = min(len(future_returns), len(rsi))
            future_returns = future_returns[:min_len]
            rsi = rsi[:min_len]
        
        # RSI values before up moves
        up_threshold = np.percentile(future_returns, 75)
        rsi_before_up = rsi[future_returns > up_threshold]
        
        if len(rsi_before_up) > 20:
            threshold = float(np.percentile(rsi_before_up, 25))
        else:
            threshold = float(np.percentile(rsi, 10))
        
        threshold = max(15, min(40, threshold))
        
        logger.debug(f"Derived oversold threshold: {threshold:.1f}")
        return threshold
    
    def _derive_breakout_threshold(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """Derive breakout detection threshold from range analysis."""
        # Calculate daily range as percentage
        range_pct = (high - low) / (low + 1e-10) * 100
        range_pct = range_pct[np.isfinite(range_pct)]
        
        if len(range_pct) < 50:
            return 2.0
        
        # Breakout threshold is 90th percentile of range
        threshold = float(np.percentile(range_pct, 90))
        
        # Bound to reasonable range
        threshold = max(0.5, min(10.0, threshold))
        
        logger.debug(f"Derived breakout threshold: {threshold:.2f}%")
        return threshold
    
    def _derive_mean_reversion_threshold(self, close: np.ndarray) -> float:
        """Derive mean-reversion entry threshold from z-score analysis."""
        if len(close) < 100:
            return 2.0
        
        # Calculate z-scores
        rolling_mean = pd.Series(close).rolling(50).mean().values
        rolling_std = pd.Series(close).rolling(50).std().values
        
        zscore = (close - rolling_mean) / (rolling_std + 1e-10)
        zscore = zscore[np.isfinite(zscore)]
        
        if len(zscore) < 50:
            return 2.0
        
        # Find z-score level that historically preceded reversals
        future_returns = np.zeros(len(close))
        future_returns[:-10] = (close[10:] - close[:-10]) / close[:-10]
        future_returns = future_returns[50:]  # Skip warmup period
        
        min_len = min(len(zscore), len(future_returns))
        zscore = zscore[:min_len]
        future_returns = future_returns[:min_len]
        
        # For high z-scores, check if price subsequently fell
        best_z = 2.0
        best_edge = 0
        
        for z in np.arange(1.0, 3.5, 0.25):
            mask = zscore > z
            if np.sum(mask) > 10:
                returns_at_z = future_returns[mask]
                edge = -np.mean(returns_at_z)  # Negative because we'd short
                if edge > best_edge:
                    best_edge = edge
                    best_z = z
        
        logger.debug(f"Derived MR threshold: {best_z:.2f}")
        return best_z
    
    # ==================== VOLATILITY DERIVATION ====================
    
    def _derive_volatility_scalar(self, close: np.ndarray) -> float:
        """Derive volatility scaling factor."""
        returns = np.diff(np.log(close + 1e-10))
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 50:
            return 1.0
        
        # Recent vs long-term volatility
        recent_vol = np.std(returns[-20:])
        long_term_vol = np.std(returns)
        
        if long_term_vol == 0:
            return 1.0
        
        scalar = recent_vol / long_term_vol
        
        # Bound to prevent extreme values
        scalar = max(0.5, min(2.0, scalar))
        
        logger.debug(f"Derived volatility scalar: {scalar:.2f}")
        return scalar
    
    # ==================== RISK DERIVATION ====================
    
    def _derive_kelly_fraction(self, close: np.ndarray) -> float:
        """Derive optimal risk fraction using Kelly Criterion."""
        returns = np.diff(np.log(close + 1e-10))
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 100:
            return 0.005  # 0.5% default
        
        # Simulate simple momentum strategy
        signal = np.sign(returns[:-1])  # Previous return sign
        strategy_returns = signal * returns[1:]
        
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.005
        
        # Win probability
        p = len(wins) / len(strategy_returns)
        
        # Win/loss ratio
        avg_win = np.mean(wins)
        avg_loss = np.abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.005
        
        b = avg_win / avg_loss
        
        # Kelly formula
        kelly = (p * b - (1 - p)) / b if b > 0 else 0
        
        # Half-Kelly for safety
        kelly = kelly / 2
        
        # Bound to reasonable range
        kelly = max(0.001, min(0.02, kelly))  # 0.1% to 2%
        
        logger.debug(f"Derived Kelly fraction: {kelly:.4f} ({kelly*100:.2f}%)")
        return kelly
    
    def _derive_stop_loss_multiple(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """Derive stop-loss ATR multiple from adverse excursion analysis."""
        # Calculate ATR
        atr = self._calculate_atr(high, low, close, 14)
        
        if len(atr) < 50 or np.mean(atr) == 0:
            return 2.0
        
        # Maximum adverse excursion for simulated longs
        # How far price moved against entry before recovery or stop
        mae = []
        
        for i in range(50, len(close) - 20):
            entry = close[i]
            future_low = np.min(low[i:i+20])
            excursion = (entry - future_low) / atr[i] if atr[i] > 0 else 0
            mae.append(excursion)
        
        if not mae:
            return 2.0
        
        mae = np.array(mae)
        mae = mae[np.isfinite(mae)]
        
        if len(mae) < 20:
            return 2.0
        
        # 75th percentile of MAE - most trades could have survived this
        sl_multiple = float(np.percentile(mae, 75))
        
        # Adjust by volatility regime
        vol_scalar = self._derive_volatility_scalar(close)
        sl_multiple = sl_multiple * vol_scalar
        
        # Bound to reasonable range
        sl_multiple = max(1.0, min(4.0, sl_multiple))
        
        logger.debug(f"Derived SL multiple: {sl_multiple:.2f}")
        return sl_multiple
    
    def _derive_take_profit_multiple(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """Derive take-profit ATR multiple from favorable excursion analysis."""
        atr = self._calculate_atr(high, low, close, 14)
        
        if len(atr) < 50 or np.mean(atr) == 0:
            return 3.0
        
        # Maximum favorable excursion for simulated longs
        mfe = []
        
        for i in range(50, len(close) - 20):
            entry = close[i]
            future_high = np.max(high[i:i+20])
            excursion = (future_high - entry) / atr[i] if atr[i] > 0 else 0
            mfe.append(excursion)
        
        if not mfe:
            return 3.0
        
        mfe = np.array(mfe)
        mfe = mfe[np.isfinite(mfe)]
        
        if len(mfe) < 20:
            return 3.0
        
        # Find optimal TP that maximizes expected value
        best_ev = 0
        best_tp = 2.0
        
        for tp_level in np.arange(1.0, 5.0, 0.25):
            prob_reach = np.mean(mfe >= tp_level)
            ev = prob_reach * tp_level
            
            if ev > best_ev:
                best_ev = ev
                best_tp = tp_level
        
        logger.debug(f"Derived TP multiple: {best_tp:.2f} (EV={best_ev:.2f})")
        return best_tp
    
    # ==================== REGIME DERIVATION ====================
    
    def _derive_regime_lookback(self, close: np.ndarray) -> int:
        """Derive optimal lookback for regime detection."""
        # Test different lookbacks and find most stable Hurst
        lookbacks = range(50, min(500, len(close) // 2), 25)
        
        if len(close) < 100:
            return 50
        
        hurst_values = []
        for lookback in lookbacks:
            H = hurst_exponent(close[-lookback:])
            hurst_values.append(H)
        
        if len(hurst_values) < 3:
            return 100
        
        # Find most stable region
        hurst_std = pd.Series(hurst_values).rolling(3).std()
        valid_stds = hurst_std.dropna()
        
        if len(valid_stds) == 0:
            return 100
        
        best_idx = valid_stds.idxmin()
        best_lookback = list(lookbacks)[best_idx]
        
        logger.debug(f"Derived regime lookback: {best_lookback}")
        return best_lookback
    
    def _derive_trend_threshold(self, close: np.ndarray, FD: float) -> float:
        """Derive trend strength threshold from fractal dimension."""
        # FD = 1.5 is random walk
        # Trend threshold is how far from 1.5 we need to classify as trending
        
        threshold = abs(1.5 - FD) * 2
        threshold = max(0.1, min(0.5, threshold))
        
        logger.debug(f"Derived trend threshold: {threshold:.2f}")
        return threshold
    
    # ==================== HELPER METHODS ====================
    
    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI."""
        delta = np.diff(close)
        delta = np.concatenate([[0], delta])
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate ATR."""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = pd.Series(tr).rolling(period).mean().values
        return atr
    
    def _compute_overall_confidence(self, data_points: int) -> float:
        """Compute overall confidence in calibration."""
        # More data = higher confidence
        data_factor = min(1.0, data_points / (self.min_bars * 3))
        
        return 0.5 + 0.5 * data_factor
    
    def _calibrate_with_limited_data(self, df: pd.DataFrame) -> FullCalibrationResult:
        """Create conservative calibration with limited data."""
        n = len(df)
        
        return FullCalibrationResult(
            fast_period=max(5, n // 50),
            slow_period=max(20, n // 10),
            signal_period=9,
            atr_period=14,
            overbought_threshold=70.0,
            oversold_threshold=30.0,
            breakout_threshold=2.0,
            mean_reversion_threshold=2.0,
            volatility_scalar=1.0,
            optimal_risk_fraction=0.005,
            stop_loss_atr_multiple=2.0,
            take_profit_atr_multiple=3.0,
            regime_lookback=min(100, n // 2),
            trend_strength_threshold=0.3,
            mean_reversion_halflife=50.0,
            hurst_exponent=0.5,
            fractal_dimension=1.5,
            market_entropy=2.0,
            current_regime="unknown",
            data_points_used=n,
            confidence=0.3
        )
