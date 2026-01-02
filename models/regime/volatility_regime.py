"""
GARCH Volatility Regime Model - Volatility clustering and forecasting.

GARCH models capture the volatility clustering observed in financial markets
where high volatility tends to follow high volatility.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, FIGARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch package not installed")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class GARCHModel(BaseModel):
    """
    GARCH model for volatility forecasting and regime detection.
    
    Variants:
    - GARCH(1,1): Standard model
    - EGARCH: Asymmetric effects (leverage)
    - GJR-GARCH: Threshold effects
    
    Usage:
        model = GARCHModel(variant='egarch')
        model.fit(returns)
        
        forecast = model.forecast_volatility(horizon=5)
        regime = model.detect_volatility_regime()
    """
    
    def __init__(
        self,
        name: str = "garch",
        p: int = 1,
        q: int = 1,
        variant: str = 'garch',  # 'garch', 'egarch', 'gjr'
        dist: str = 'normal'  # 'normal', 't', 'skewt'
    ):
        """
        Initialize GARCH model.
        
        Args:
            name: Model name
            p: GARCH order (volatility lags)
            q: ARCH order (return lags)
            variant: Model variant
            dist: Error distribution
        """
        super().__init__(name)
        
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required")
        
        self.p = p
        self.q = q
        self.variant = variant
        self.dist = dist
        
        self.model = None
        self.fitted_model = None
        self.returns_std: float = 1.0
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'GARCHModel':
        """
        Fit GARCH model to return series.
        
        Args:
            X: Return series (not prices!)
        """
        returns = X.flatten() if len(X.shape) > 1 else X
        returns = returns * 100  # Scale for numerical stability
        
        self.returns_std = np.std(returns)
        
        # Create model based on variant
        if self.variant == 'egarch':
            vol = EGARCH(p=self.p, q=self.q)
        elif self.variant == 'gjr':
            vol = GARCH(p=self.p, o=1, q=self.q)
        else:
            vol = GARCH(p=self.p, q=self.q)
        
        self.model = arch_model(
            returns,
            vol=vol,
            dist=self.dist,
            rescale=False
        )
        
        # Fit with increased iterations
        self.fitted_model = self.model.fit(
            disp='off',
            options={'maxiter': 500}
        )
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        self.metadata = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood
        }
        
        logger.info(f"GARCH fitted: AIC={self.metadata['aic']:.2f}")
        
        return self
    
    def predict(self, X: np.ndarray = None) -> ModelPrediction:
        """
        Generate trading signal based on volatility regime.
        
        High vol = reduce position
        Low vol = increase position
        """
        if not self.is_trained:
            raise RuntimeError("Model not fitted")
        
        # Get current volatility
        cond_vol = self.fitted_model.conditional_volatility[-1]
        
        # Forecast volatility
        forecast = self.fitted_model.forecast(horizon=1)
        forecast_vol = np.sqrt(forecast.variance.values[-1, 0])
        
        # Volatility regime
        long_term_vol = self.fitted_model.conditional_volatility.mean()
        vol_ratio = cond_vol / long_term_vol
        
        # Signal based on vol regime
        if vol_ratio > 1.5:
            # High vol - reduce exposure
            direction = 0.0
            confidence = 0.8
            regime = 'high_volatility'
        elif vol_ratio > 1.2:
            # Elevated vol
            direction = 0.0
            confidence = 0.5
            regime = 'elevated_volatility'
        elif vol_ratio < 0.7:
            # Low vol - can increase exposure
            direction = 0.0  # Direction from other signals
            confidence = 0.8
            regime = 'low_volatility'
        else:
            direction = 0.0
            confidence = 0.3
            regime = 'normal_volatility'
        
        return ModelPrediction(
            model_name=self.name,
            direction=direction,
            magnitude=forecast_vol / 100,  # Expected daily move
            confidence=confidence,
            metadata={
                'current_vol': float(cond_vol / 100),
                'forecast_vol': float(forecast_vol / 100),
                'vol_ratio': float(vol_ratio),
                'regime': regime,
                'position_scalar': 1.0 / vol_ratio if vol_ratio > 1 else 1.0
            }
        )
    
    def forecast_volatility(self, horizon: int = 5) -> Dict[str, np.ndarray]:
        """
        Forecast volatility over horizon.
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            Dict with mean, variance forecasts
        """
        if not self.is_trained:
            raise RuntimeError("Model not fitted")
        
        forecast = self.fitted_model.forecast(horizon=horizon)
        
        return {
            'mean': forecast.mean.values[-1, :] / 100,
            'variance': forecast.variance.values[-1, :] / 10000,
            'volatility': np.sqrt(forecast.variance.values[-1, :]) / 100
        }
    
    def detect_volatility_regime(self) -> Dict[str, Any]:
        """
        Detect current volatility regime.
        
        Returns:
            Dict with regime classification and metrics
        """
        if not self.is_trained:
            return {'regime': 'unknown'}
        
        cond_vol = self.fitted_model.conditional_volatility
        
        # Current vs historical
        current = cond_vol.iloc[-1]
        mean_vol = cond_vol.mean()
        std_vol = cond_vol.std()
        
        # Z-score
        vol_zscore = (current - mean_vol) / std_vol
        
        # Percentile
        vol_percentile = (cond_vol < current).mean() * 100
        
        # Regime classification
        if vol_zscore > 2:
            regime = 'extreme_high'
        elif vol_zscore > 1:
            regime = 'high'
        elif vol_zscore < -1:
            regime = 'low'
        elif vol_zscore < -2:
            regime = 'extreme_low'
        else:
            regime = 'normal'
        
        return {
            'regime': regime,
            'current_vol': float(current / 100),
            'mean_vol': float(mean_vol / 100),
            'vol_zscore': float(vol_zscore),
            'vol_percentile': float(vol_percentile),
            'position_scalar': max(0.25, min(1.5, 1.0 / (1 + vol_zscore * 0.3)))
        }
    
    def get_vol_of_vol(self) -> float:
        """Get volatility of volatility (vol clustering strength)."""
        if not self.is_trained:
            return 0.0
        
        cond_vol = self.fitted_model.conditional_volatility
        vol_returns = cond_vol.pct_change().dropna()
        
        return float(vol_returns.std())
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'p': self.p,
            'q': self.q,
            'variant': self.variant,
            'dist': self.dist,
            'returns_std': self.returns_std,
            'params': dict(self.fitted_model.params) if self.fitted_model else None
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.p = state['p']
        self.q = state['q']
        self.variant = state['variant']
        self.dist = state['dist']
        self.returns_std = state['returns_std']
        # Note: Full model restoration requires refitting


class VolatilityRegimeFilter:
    """
    Filter trading signals based on volatility regime.
    
    Usage:
        filter = VolatilityRegimeFilter(garch_model)
        
        # Scale position based on vol
        scaled_size = filter.scale_position(base_size, returns)
        
        # Filter signal
        filtered_signal = filter.filter_signal(signal, returns)
    """
    
    def __init__(
        self,
        garch_model: GARCHModel = None,
        lookback: int = 100
    ):
        """Initialize filter."""
        self.garch = garch_model
        self.lookback = lookback
        
        # Fallback simple vol calculation
        self.use_simple = garch_model is None
    
    def scale_position(
        self,
        base_size: float,
        returns: np.ndarray = None
    ) -> float:
        """
        Scale position size based on volatility.
        
        High vol → smaller position
        Low vol → larger position
        """
        if self.use_simple and returns is not None:
            # Simple scaling
            recent_vol = np.std(returns[-20:])
            long_term_vol = np.std(returns[-self.lookback:])
            
            vol_ratio = recent_vol / (long_term_vol + 1e-8)
            scalar = 1.0 / vol_ratio if vol_ratio > 1 else 1.0
            
        elif self.garch and self.garch.is_trained:
            regime = self.garch.detect_volatility_regime()
            scalar = regime['position_scalar']
        else:
            scalar = 1.0
        
        # Bound scalar
        scalar = max(0.25, min(1.5, scalar))
        
        return base_size * scalar
    
    def filter_signal(
        self,
        signal: float,
        returns: np.ndarray = None
    ) -> Tuple[float, float]:
        """
        Filter signal based on volatility regime.
        
        Returns:
            Tuple of (filtered_signal, confidence_adjustment)
        """
        if self.garch and self.garch.is_trained:
            regime = self.garch.detect_volatility_regime()
            vol_regime = regime['regime']
            
            if vol_regime in ['extreme_high', 'high']:
                # Reduce signal strength in high vol
                return signal * 0.5, 0.7
            elif vol_regime in ['extreme_low', 'low']:
                # Can be more aggressive in low vol
                return signal, 1.0
            else:
                return signal, 0.9
        
        return signal, 1.0
