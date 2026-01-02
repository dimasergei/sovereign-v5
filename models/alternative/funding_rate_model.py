"""
Funding Rate Model - Crypto funding rate arbitrage signals.

Funding rates represent the cost of holding perpetual futures vs spot.
High positive funding = pay to be long = bearish signal
High negative funding = pay to be short = bullish signal
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


@dataclass
class FundingSignal:
    """Funding rate signal."""
    symbol: str
    direction: float  # -1 to 1
    magnitude: float  # Expected move
    confidence: float
    funding_rate: float
    annualized_rate: float
    timestamp: datetime


class FundingRateModel(BaseModel):
    """
    Funding rate arbitrage model.
    
    Trading logic:
    - Very high funding (>0.1%/8h = 137% annual) → Short signal
    - Very low funding (<-0.1%/8h) → Long signal
    - Neutral funding → No signal
    
    The model is contrarian - extreme funding typically precedes reversals.
    
    Usage:
        model = FundingRateModel()
        
        # With live data
        signal = model.generate_signal(
            funding_rate=0.0015,  # 0.15% per 8h
            symbol="BTCUSD"
        )
        
        # With historical fitting
        model.fit(historical_funding_rates, historical_returns)
    """
    
    def __init__(
        self,
        name: str = "funding_rate",
        extreme_threshold: float = 0.001,  # 0.1% per 8h
        moderate_threshold: float = 0.0005,  # 0.05% per 8h
    ):
        """
        Initialize funding rate model.
        
        Args:
            name: Model name
            extreme_threshold: Funding rate for extreme signal
            moderate_threshold: Funding rate for moderate signal
        """
        super().__init__(name)
        
        self.extreme_threshold = extreme_threshold
        self.moderate_threshold = moderate_threshold
        
        # Calibrated parameters (from historical analysis)
        self.optimal_threshold: float = extreme_threshold
        self.signal_decay: float = 0.9  # Signal strength decay per period
        self.historical_accuracy: float = 0.5
    
    def fit(
        self,
        funding_rates: np.ndarray,
        returns: np.ndarray,
        lookahead: int = 24  # 24 periods (3 days at 8h funding)
    ) -> 'FundingRateModel':
        """
        Fit model to historical funding rates and returns.
        
        Finds optimal thresholds that maximize predictive accuracy.
        
        Args:
            funding_rates: Historical funding rates
            returns: Future returns (lookahead periods)
            lookahead: Prediction horizon
        """
        # Test different thresholds
        best_accuracy = 0
        best_threshold = self.extreme_threshold
        
        for threshold in np.arange(0.0003, 0.003, 0.0001):
            # Generate signals
            signals = np.where(
                funding_rates > threshold, -1,
                np.where(funding_rates < -threshold, 1, 0)
            )
            
            # Calculate accuracy (signal direction matches return direction)
            # Shift returns for lookahead
            future_returns = np.roll(returns, -lookahead)
            future_returns[-lookahead:] = 0
            
            correct = (signals * future_returns > 0) | (signals == 0)
            accuracy = correct[signals != 0].mean() if (signals != 0).any() else 0
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        self.historical_accuracy = best_accuracy
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        self.metadata = {
            'optimal_threshold': best_threshold,
            'historical_accuracy': best_accuracy,
            'n_samples': len(funding_rates)
        }
        
        logger.info(
            f"Funding model fitted: threshold={best_threshold:.4f}, "
            f"accuracy={best_accuracy:.2%}"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Generate prediction from funding rate.
        
        Args:
            X: Current funding rate (scalar or array)
        """
        funding_rate = float(X.flatten()[-1]) if hasattr(X, 'flatten') else float(X)
        
        return self.generate_signal(funding_rate)
    
    def generate_signal(
        self,
        funding_rate: float,
        symbol: str = "BTCUSD"
    ) -> ModelPrediction:
        """
        Generate trading signal from funding rate.
        
        Args:
            funding_rate: Current funding rate (decimal, e.g., 0.0001 = 0.01%)
            symbol: Trading symbol
        """
        threshold = self.optimal_threshold if self.is_trained else self.extreme_threshold
        
        # Annualized rate (funding every 8h = 3x/day = 1095x/year)
        annualized = funding_rate * 1095 * 100
        
        # Generate signal
        if funding_rate > threshold:
            # High positive funding - contrarian short
            direction = -1.0
            strength = min(1.0, funding_rate / threshold)
            confidence = min(0.9, 0.5 + strength * 0.4)
        elif funding_rate < -threshold:
            # High negative funding - contrarian long
            direction = 1.0
            strength = min(1.0, abs(funding_rate) / threshold)
            confidence = min(0.9, 0.5 + strength * 0.4)
        elif abs(funding_rate) > self.moderate_threshold:
            # Moderate signal
            direction = -np.sign(funding_rate) * 0.5
            confidence = 0.4
        else:
            # Neutral
            direction = 0.0
            confidence = 0.0
        
        # Adjust confidence by historical accuracy
        if self.is_trained:
            confidence *= self.historical_accuracy * 2  # Scale by accuracy
        
        return ModelPrediction(
            model_name=self.name,
            direction=direction,
            magnitude=abs(funding_rate) * 10,  # Expected move scaled
            confidence=min(1.0, confidence),
            metadata={
                'funding_rate': funding_rate,
                'annualized_rate': annualized,
                'threshold_used': threshold,
                'symbol': symbol
            }
        )
    
    def get_funding_regime(self, funding_rate: float) -> str:
        """Classify current funding regime."""
        if funding_rate > self.extreme_threshold * 2:
            return "extreme_positive"
        elif funding_rate > self.extreme_threshold:
            return "high_positive"
        elif funding_rate > self.moderate_threshold:
            return "moderate_positive"
        elif funding_rate < -self.extreme_threshold * 2:
            return "extreme_negative"
        elif funding_rate < -self.extreme_threshold:
            return "high_negative"
        elif funding_rate < -self.moderate_threshold:
            return "moderate_negative"
        else:
            return "neutral"
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'extreme_threshold': self.extreme_threshold,
            'moderate_threshold': self.moderate_threshold,
            'optimal_threshold': self.optimal_threshold,
            'signal_decay': self.signal_decay,
            'historical_accuracy': self.historical_accuracy
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.extreme_threshold = state['extreme_threshold']
        self.moderate_threshold = state['moderate_threshold']
        self.optimal_threshold = state['optimal_threshold']
        self.signal_decay = state['signal_decay']
        self.historical_accuracy = state['historical_accuracy']


class FundingArbitrageStrategy:
    """
    Complete funding arbitrage strategy.
    
    Combines funding rate signals with:
    - Position sizing based on funding magnitude
    - Entry/exit timing
    - Risk management
    """
    
    def __init__(
        self,
        model: FundingRateModel = None,
        max_position_pct: float = 0.5,
        entry_threshold: float = 0.001,
        exit_threshold: float = 0.0003
    ):
        """Initialize strategy."""
        self.model = model or FundingRateModel()
        self.max_position_pct = max_position_pct
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # State
        self.current_position: float = 0.0
        self.entry_funding: float = 0.0
    
    def update(self, funding_rate: float, price: float) -> Dict[str, Any]:
        """
        Update strategy with new funding rate.
        
        Returns:
            Dict with action and sizing
        """
        signal = self.model.generate_signal(funding_rate)
        
        action = "hold"
        size = 0.0
        
        # Entry logic
        if self.current_position == 0:
            if abs(funding_rate) > self.entry_threshold:
                action = "enter"
                size = self.max_position_pct * signal.confidence
                self.current_position = signal.direction * size
                self.entry_funding = funding_rate
        
        # Exit logic
        elif self.current_position != 0:
            # Exit if funding normalized
            if abs(funding_rate) < self.exit_threshold:
                action = "exit"
                size = abs(self.current_position)
                self.current_position = 0.0
            
            # Exit if funding reversed significantly
            elif np.sign(funding_rate) != np.sign(self.entry_funding):
                if abs(funding_rate) > self.entry_threshold:
                    action = "reverse"
                    size = abs(self.current_position) + self.max_position_pct * signal.confidence
                    self.current_position = signal.direction * self.max_position_pct * signal.confidence
        
        return {
            'action': action,
            'direction': signal.direction,
            'size': size,
            'current_position': self.current_position,
            'funding_rate': funding_rate,
            'signal': signal
        }
