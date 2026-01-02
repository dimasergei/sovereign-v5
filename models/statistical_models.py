"""
Statistical Models - Mean Reversion and Regime Detection.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseModel, ModelPrediction


logger = logging.getLogger(__name__)


class MeanReversionModel(BaseModel):
    """
    Ornstein-Uhlenbeck based mean reversion model.
    
    Generates signals when price deviates significantly from mean,
    expecting reversion.
    """
    
    def __init__(
        self,
        name: str = "mean_reversion",
        lookback: int = 50,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5
    ):
        super().__init__(name)
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        
        # Fitted parameters
        self.mean = None
        self.std = None
        self.halflife = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'MeanReversionModel':
        """
        Fit OU process parameters.
        
        Args:
            X: Price series (1D array)
            y: Not used
        """
        prices = X.flatten() if len(X.shape) > 1 else X
        
        # Calculate mean and std
        self.mean = np.mean(prices)
        self.std = np.std(prices)
        
        # Estimate half-life using AR(1)
        self.halflife = self._estimate_halflife(prices)
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        logger.info(
            f"MeanReversion fitted: mean={self.mean:.4f}, "
            f"std={self.std:.4f}, halflife={self.halflife:.1f}"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Generate mean reversion signal.
        
        Args:
            X: Recent price series
        """
        prices = X.flatten() if len(X.shape) > 1 else X
        
        # Calculate rolling stats
        rolling_mean = np.mean(prices[-self.lookback:])
        rolling_std = np.std(prices[-self.lookback:])

        # Use rolling stats if not fitted, otherwise blend with fitted values
        if self.mean is None:
            mean = rolling_mean
            std = rolling_std
        else:
            # Blend fitted and rolling
            mean = 0.7 * rolling_mean + 0.3 * self.mean
            std = 0.7 * rolling_std + 0.3 * self.std
        
        current_price = prices[-1]
        
        # Calculate z-score
        zscore = (current_price - mean) / (std + 1e-10)
        
        # Generate signal
        if abs(zscore) >= self.entry_zscore:
            # Strong deviation - expect reversion
            direction = -np.sign(zscore)  # Opposite of deviation
            magnitude = abs(zscore) * std / current_price  # Expected move %
            confidence = min(1.0, abs(zscore) / 4)  # Higher zscore = higher confidence
        elif abs(zscore) <= self.exit_zscore:
            # Near mean - neutral
            direction = 0.0
            magnitude = 0.0
            confidence = 0.0
        else:
            # Moderate deviation
            direction = -np.sign(zscore) * 0.5
            magnitude = abs(zscore) * std / current_price * 0.5
            confidence = 0.3
        
        return ModelPrediction(
            model_name=self.name,
            direction=float(direction),
            magnitude=float(magnitude),
            confidence=float(confidence),
            metadata={
                'zscore': float(zscore),
                'mean': float(mean),
                'std': float(std),
                'halflife': float(self.halflife) if self.halflife else None
            }
        )
    
    def _estimate_halflife(self, prices: np.ndarray) -> float:
        """Estimate mean-reversion half-life."""
        log_prices = np.log(prices + 1e-10)
        
        y = log_prices[1:]
        x = log_prices[:-1]
        
        if len(y) < 20:
            return 50.0
        
        # OLS regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        
        if den == 0:
            return 50.0
        
        beta = num / den
        
        if beta >= 1 or beta <= 0:
            return 50.0
        
        halflife = -np.log(2) / np.log(beta)
        return max(1.0, min(200.0, halflife))
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'lookback': self.lookback,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'mean': self.mean,
            'std': self.std,
            'halflife': self.halflife,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.lookback = state.get('lookback', 50)
        self.entry_zscore = state.get('entry_zscore', 2.0)
        self.exit_zscore = state.get('exit_zscore', 0.5)
        self.mean = state.get('mean')
        self.std = state.get('std')
        self.halflife = state.get('halflife')


class RegimeDetector(BaseModel):
    """
    Market regime detection using multiple methods.
    
    Identifies:
    - TRENDING_UP / TRENDING_DOWN
    - MEAN_REVERTING
    - HIGH_VOLATILITY / LOW_VOLATILITY
    - CHOPPY
    """
    
    REGIMES = [
        'trending_up',
        'trending_down', 
        'mean_reverting',
        'high_volatility',
        'low_volatility',
        'choppy'
    ]
    
    def __init__(
        self,
        name: str = "regime_detector",
        lookback: int = 100
    ):
        super().__init__(name)
        self.lookback = lookback
        
        # Regime probabilities
        self.regime_probs: Dict[str, float] = {}
        self.current_regime: str = "unknown"
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'RegimeDetector':
        """
        Fit regime detector (calculates baseline statistics).
        """
        prices = X.flatten() if len(X.shape) > 1 else X
        
        # Store baseline volatility
        returns = np.diff(np.log(prices + 1e-10))
        self.baseline_volatility = np.std(returns)
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Detect current market regime.
        """
        prices = X.flatten() if len(X.shape) > 1 else X
        
        if len(prices) < self.lookback:
            return self._default_prediction()
        
        recent_prices = prices[-self.lookback:]
        
        # Calculate regime indicators
        hurst = self._calculate_hurst(recent_prices)
        volatility_ratio = self._calculate_volatility_ratio(recent_prices)
        trend_strength = self._calculate_trend_strength(recent_prices)
        
        # Determine regime
        regime, confidence = self._classify_regime(
            hurst, volatility_ratio, trend_strength, recent_prices
        )
        
        self.current_regime = regime
        
        # Convert to trading signal
        direction, magnitude = self._regime_to_signal(
            regime, trend_strength, recent_prices
        )
        
        return ModelPrediction(
            model_name=self.name,
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            metadata={
                'regime': regime,
                'hurst': float(hurst),
                'volatility_ratio': float(volatility_ratio),
                'trend_strength': float(trend_strength),
            }
        )
    
    def detect_regime(self, prices: np.ndarray) -> Tuple[str, float]:
        """
        Simplified regime detection.
        
        Returns:
            Tuple of (regime_name, probability)
        """
        pred = self.predict(prices)
        return pred.metadata['regime'], pred.confidence
    
    def _calculate_hurst(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent (simplified R/S method)."""
        n = len(prices)
        if n < 20:
            return 0.5
        
        returns = np.diff(np.log(prices + 1e-10))
        
        # R/S analysis
        rs_values = []
        
        for k in range(2, min(7, int(np.log2(n)))):
            size = int(2 ** k)
            num_subsets = len(returns) // size
            
            if num_subsets == 0:
                continue
            
            rs_list = []
            for i in range(num_subsets):
                subset = returns[i * size:(i + 1) * size]
                
                mean_adj = subset - np.mean(subset)
                cumsum = np.cumsum(mean_adj)
                
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(subset, ddof=1)
                
                if S > 0:
                    rs_list.append(R / S)
            
            if rs_list:
                rs_values.append((size, np.mean(rs_list)))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Fit log-log line
        log_n = np.log([r[0] for r in rs_values])
        log_rs = np.log([r[1] for r in rs_values])
        
        try:
            H, _ = np.polyfit(log_n, log_rs, 1)
        except:
            return 0.5
        
        return max(0.0, min(1.0, H))
    
    def _calculate_volatility_ratio(self, prices: np.ndarray) -> float:
        """Calculate recent vs long-term volatility ratio."""
        returns = np.diff(np.log(prices + 1e-10))
        
        recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        long_term_vol = np.std(returns)
        
        if long_term_vol == 0:
            return 1.0
        
        return recent_vol / long_term_vol
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength (-1 to 1)."""
        # Use linear regression slope
        n = len(prices)
        x = np.arange(n)
        
        # Normalize
        x_norm = (x - x.mean()) / (x.std() + 1e-10)
        y_norm = (prices - prices.mean()) / (prices.std() + 1e-10)
        
        # Correlation coefficient
        corr = np.corrcoef(x_norm, y_norm)[0, 1]
        
        if not np.isfinite(corr):
            return 0.0
        
        return corr
    
    def _classify_regime(
        self,
        hurst: float,
        volatility_ratio: float,
        trend_strength: float,
        prices: np.ndarray
    ) -> Tuple[str, float]:
        """Classify market regime."""
        
        # High volatility regime
        if volatility_ratio > 1.5:
            return 'high_volatility', min(1.0, volatility_ratio - 1.0)
        
        # Low volatility regime
        if volatility_ratio < 0.5:
            return 'low_volatility', min(1.0, 1.0 - volatility_ratio)
        
        # Trending regime
        if hurst > 0.55 and abs(trend_strength) > 0.6:
            if trend_strength > 0:
                return 'trending_up', abs(trend_strength)
            else:
                return 'trending_down', abs(trend_strength)
        
        # Mean reverting regime
        if hurst < 0.45:
            return 'mean_reverting', 1.0 - hurst
        
        # Choppy
        return 'choppy', 0.5
    
    def _regime_to_signal(
        self,
        regime: str,
        trend_strength: float,
        prices: np.ndarray
    ) -> Tuple[float, float]:
        """Convert regime to trading signal."""
        
        if regime == 'trending_up':
            return 0.7, abs(trend_strength) * 0.01
        
        elif regime == 'trending_down':
            return -0.7, abs(trend_strength) * 0.01
        
        elif regime == 'mean_reverting':
            # Signal based on current deviation from mean
            mean = np.mean(prices)
            std = np.std(prices)
            zscore = (prices[-1] - mean) / (std + 1e-10)
            
            direction = -np.sign(zscore) * 0.5
            magnitude = abs(zscore) * 0.005
            return direction, magnitude
        
        else:
            return 0.0, 0.0
    
    def _default_prediction(self) -> ModelPrediction:
        return ModelPrediction(
            model_name=self.name,
            direction=0.0,
            magnitude=0.0,
            confidence=0.0,
            metadata={'regime': 'unknown'}
        )


class LeadLagDetector:
    """
    Detects lead-lag relationships between assets.
    
    Some assets lead others (e.g., BTC leads ETH, EUR leads GBP).
    """
    
    def __init__(self, max_lag: int = 10):
        self.max_lag = max_lag
        self.relationships: List[Dict] = []
    
    def fit(self, price_df: pd.DataFrame) -> 'LeadLagDetector':
        """
        Fit lead-lag relationships from price DataFrame.
        
        Args:
            price_df: DataFrame with asset prices as columns
        """
        self.relationships = []
        
        columns = price_df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Calculate returns
                returns1 = price_df[col1].pct_change().dropna()
                returns2 = price_df[col2].pct_change().dropna()
                
                # Align
                common_idx = returns1.index.intersection(returns2.index)
                r1 = returns1.loc[common_idx].values
                r2 = returns2.loc[common_idx].values
                
                # Find best lag
                best_lag, best_corr = self._find_best_lag(r1, r2)
                
                if abs(best_corr) > 0.1:
                    if best_lag > 0:
                        leader, follower = col1, col2
                    else:
                        leader, follower = col2, col1
                        best_lag = -best_lag
                    
                    self.relationships.append({
                        'leader': leader,
                        'follower': follower,
                        'lag': abs(best_lag),
                        'correlation': best_corr,
                    })
        
        # Sort by correlation strength
        self.relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return self
    
    def _find_best_lag(
        self,
        series1: np.ndarray,
        series2: np.ndarray
    ) -> Tuple[int, float]:
        """Find lag with highest cross-correlation."""
        best_lag = 0
        best_corr = 0
        
        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag < 0:
                s1 = series1[:lag]
                s2 = series2[-lag:]
            elif lag > 0:
                s1 = series1[lag:]
                s2 = series2[:-lag]
            else:
                s1, s2 = series1, series2
            
            if len(s1) < 20:
                continue
            
            corr = np.corrcoef(s1, s2)[0, 1]
            
            if np.isfinite(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        return best_lag, best_corr
    
    def get_lead_lag_pairs(self) -> List[Tuple[str, str, int, float]]:
        """Get detected lead-lag relationships."""
        return [
            (r['leader'], r['follower'], r['lag'], r['correlation'])
            for r in self.relationships
        ]
    
    def predict_from_leader(
        self,
        leader: str,
        leader_return: float
    ) -> Dict[str, float]:
        """
        Predict follower moves based on leader move.
        
        Args:
            leader: Leader asset name
            leader_return: Leader's recent return
            
        Returns:
            Dictionary of follower -> expected return
        """
        predictions = {}
        
        for rel in self.relationships:
            if rel['leader'] == leader:
                expected = leader_return * rel['correlation']
                predictions[rel['follower']] = expected
        
        return predictions
