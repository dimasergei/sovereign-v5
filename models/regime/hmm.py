"""
Hidden Markov Model (HMM) - Regime detection using HMM.

HMMs are excellent for detecting market regimes as they model
the market as switching between hidden states (regimes).
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class MarketRegime:
    """Market regime constants."""
    BULL_QUIET = 0      # Low vol uptrend
    BULL_VOLATILE = 1   # High vol uptrend
    BEAR_QUIET = 2      # Low vol downtrend
    BEAR_VOLATILE = 3   # High vol downtrend
    SIDEWAYS = 4        # Ranging market
    
    NAMES = {
        0: 'bull_quiet',
        1: 'bull_volatile',
        2: 'bear_quiet',
        3: 'bear_volatile',
        4: 'sideways'
    }


class HMMRegimeModel(BaseModel):
    """
    Hidden Markov Model for market regime detection.
    
    The model learns to identify different market states from
    return and volatility features. States are interpreted post-hoc
    based on their characteristics.
    
    Features used:
    - Returns
    - Volatility (rolling std)
    - Return/Vol ratio
    
    Usage:
        model = HMMRegimeModel(n_regimes=4)
        model.fit(price_data)
        
        current_regime = model.predict(recent_data)
        regime_probs = model.get_regime_probabilities(recent_data)
    """
    
    def __init__(
        self,
        name: str = "hmm_regime",
        n_regimes: int = 4,
        n_iter: int = 100,
        covariance_type: str = 'full'
    ):
        """
        Initialize HMM regime model.
        
        Args:
            name: Model name
            n_regimes: Number of hidden states (regimes)
            n_iter: Max iterations for EM algorithm
            covariance_type: Type of covariance matrix ('full', 'diag', 'spherical')
        """
        super().__init__(name)
        
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn required for HMMRegimeModel")
        
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        
        self.model: Optional[hmm.GaussianHMM] = None
        self.regime_stats: Dict[int, Dict] = {}
        self.regime_mapping: Dict[int, str] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'HMMRegimeModel':
        """
        Fit HMM to price data.
        
        Args:
            X: Price series (1D) or OHLCV DataFrame
            y: Not used
        """
        # Prepare features
        features = self._prepare_features(X)
        
        # Initialize HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )
        
        # Fit model
        self.model.fit(features)
        
        # Get state sequence
        states = self.model.predict(features)
        
        # Characterize each regime
        self._characterize_regimes(features, states)
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        logger.info(f"HMM fitted with {self.n_regimes} regimes")
        
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Predict current regime and generate trading signal.
        
        Args:
            X: Recent price data
            
        Returns:
            ModelPrediction with regime-based signal
        """
        if not self.is_trained:
            raise RuntimeError("Model not fitted")
        
        features = self._prepare_features(X)
        
        # Get most likely state
        current_state = self.model.predict(features[-1:])[-1]
        
        # Get state probabilities
        probs = self.model.predict_proba(features[-1:])[-1]
        
        # Generate signal based on regime
        direction, confidence = self._regime_to_signal(current_state, probs)
        
        regime_name = self.regime_mapping.get(current_state, f"regime_{current_state}")
        
        return ModelPrediction(
            model_name=self.name,
            direction=direction,
            magnitude=0.01,  # Base expected move
            confidence=confidence,
            metadata={
                'regime': current_state,
                'regime_name': regime_name,
                'probabilities': {
                    self.regime_mapping.get(i, f"regime_{i}"): float(p)
                    for i, p in enumerate(probs)
                },
                'regime_stats': self.regime_stats.get(current_state, {})
            }
        )
    
    def get_regime_probabilities(self, X: np.ndarray) -> Dict[str, float]:
        """Get probability of being in each regime."""
        if not self.is_trained:
            return {}
        
        features = self._prepare_features(X)
        probs = self.model.predict_proba(features[-1:])[-1]
        
        return {
            self.regime_mapping.get(i, f"regime_{i}"): float(p)
            for i, p in enumerate(probs)
        }
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get regime transition probability matrix."""
        if not self.is_trained:
            return np.array([])
        return self.model.transmat_
    
    def get_expected_regime_duration(self, regime: int) -> float:
        """Get expected duration of staying in a regime (in bars)."""
        if not self.is_trained:
            return 0.0
        
        # Expected duration = 1 / (1 - P(stay in state))
        stay_prob = self.model.transmat_[regime, regime]
        
        if stay_prob >= 1:
            return float('inf')
        
        return 1 / (1 - stay_prob)
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare features for HMM."""
        if isinstance(X, pd.DataFrame):
            prices = X['close'].values
        else:
            prices = X.flatten() if len(X.shape) > 1 else X
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility (rolling 20-period std)
        vol = pd.Series(returns).rolling(20).std().fillna(method='bfill').values
        
        # Ensure same length
        returns = returns[19:]
        vol = vol[19:]
        
        # Return/Vol ratio
        ratio = returns / (vol + 1e-8)
        
        # Stack features
        features = np.column_stack([returns, vol, ratio])
        
        return features
    
    def _characterize_regimes(self, features: np.ndarray, states: np.ndarray):
        """Characterize each regime based on its statistics."""
        for state in range(self.n_regimes):
            mask = states == state
            state_features = features[mask]
            
            if len(state_features) == 0:
                continue
            
            returns = state_features[:, 0]
            vols = state_features[:, 1]
            
            avg_return = np.mean(returns)
            avg_vol = np.mean(vols)
            
            self.regime_stats[state] = {
                'avg_return': float(avg_return),
                'avg_volatility': float(avg_vol),
                'sharpe': float(avg_return / (avg_vol + 1e-8) * np.sqrt(252)),
                'frequency': float(np.sum(mask) / len(states))
            }
            
            # Name the regime
            if avg_return > 0 and avg_vol < np.median(features[:, 1]):
                self.regime_mapping[state] = 'bull_quiet'
            elif avg_return > 0 and avg_vol >= np.median(features[:, 1]):
                self.regime_mapping[state] = 'bull_volatile'
            elif avg_return < 0 and avg_vol < np.median(features[:, 1]):
                self.regime_mapping[state] = 'bear_quiet'
            elif avg_return < 0 and avg_vol >= np.median(features[:, 1]):
                self.regime_mapping[state] = 'bear_volatile'
            else:
                self.regime_mapping[state] = 'sideways'
    
    def _regime_to_signal(
        self,
        regime: int,
        probabilities: np.ndarray
    ) -> Tuple[float, float]:
        """Convert regime to trading signal."""
        regime_name = self.regime_mapping.get(regime, '')
        
        # Base direction from regime
        if 'bull' in regime_name:
            direction = 0.5
        elif 'bear' in regime_name:
            direction = -0.5
        else:
            direction = 0.0
        
        # Confidence from probability
        confidence = float(probabilities[regime])
        
        # Reduce signal in volatile regimes
        if 'volatile' in regime_name:
            direction *= 0.7
        
        return direction, confidence
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'n_regimes': self.n_regimes,
            'n_iter': self.n_iter,
            'covariance_type': self.covariance_type,
            'regime_stats': self.regime_stats,
            'regime_mapping': self.regime_mapping,
            'model_params': {
                'means': self.model.means_.tolist() if self.model else None,
                'covars': self.model.covars_.tolist() if self.model else None,
                'transmat': self.model.transmat_.tolist() if self.model else None,
                'startprob': self.model.startprob_.tolist() if self.model else None,
            }
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.n_regimes = state['n_regimes']
        self.n_iter = state['n_iter']
        self.covariance_type = state['covariance_type']
        self.regime_stats = state['regime_stats']
        self.regime_mapping = state['regime_mapping']
        
        if state.get('model_params') and state['model_params']['means']:
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type
            )
            self.model.means_ = np.array(state['model_params']['means'])
            self.model.covars_ = np.array(state['model_params']['covars'])
            self.model.transmat_ = np.array(state['model_params']['transmat'])
            self.model.startprob_ = np.array(state['model_params']['startprob'])


class RegimeAwareStrategy:
    """
    Adapts trading strategy based on detected regime.
    
    Different regimes require different approaches:
    - Bull quiet: Trend following, larger positions
    - Bull volatile: Trend following, smaller positions
    - Bear quiet: Short bias, larger positions
    - Bear volatile: Short bias, smaller positions
    - Sideways: Mean reversion
    """
    
    def __init__(self, hmm_model: HMMRegimeModel):
        """Initialize with trained HMM model."""
        self.hmm = hmm_model
        
        # Strategy parameters by regime
        self.regime_params = {
            'bull_quiet': {
                'bias': 1.0,
                'position_scalar': 1.0,
                'strategy': 'trend_following',
                'stop_loss_mult': 2.0,
                'take_profit_mult': 3.0
            },
            'bull_volatile': {
                'bias': 0.5,
                'position_scalar': 0.5,
                'strategy': 'trend_following',
                'stop_loss_mult': 3.0,
                'take_profit_mult': 2.0
            },
            'bear_quiet': {
                'bias': -0.5,
                'position_scalar': 0.7,
                'strategy': 'trend_following',
                'stop_loss_mult': 2.0,
                'take_profit_mult': 3.0
            },
            'bear_volatile': {
                'bias': -0.3,
                'position_scalar': 0.3,
                'strategy': 'mean_reversion',
                'stop_loss_mult': 4.0,
                'take_profit_mult': 1.5
            },
            'sideways': {
                'bias': 0.0,
                'position_scalar': 0.6,
                'strategy': 'mean_reversion',
                'stop_loss_mult': 1.5,
                'take_profit_mult': 1.5
            }
        }
    
    def get_strategy_params(self, X: np.ndarray) -> Dict[str, Any]:
        """Get strategy parameters for current regime."""
        prediction = self.hmm.predict(X)
        regime_name = prediction.metadata.get('regime_name', 'sideways')
        
        params = self.regime_params.get(regime_name, self.regime_params['sideways']).copy()
        params['regime'] = regime_name
        params['confidence'] = prediction.confidence
        
        return params
