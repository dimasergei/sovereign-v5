"""
Base Model - Abstract base class for all prediction models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Standardized prediction output from any model."""
    model_name: str
    direction: float  # -1 (short) to 1 (long)
    magnitude: float  # Expected % move
    confidence: float  # 0 to 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_action(self, threshold: float = 0.3) -> str:
        """Convert to trading action."""
        if abs(self.direction) < threshold:
            return "neutral"
        return "long" if self.direction > 0 else "short"


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    
    All models must implement:
    - fit(): Train the model
    - predict(): Generate predictions
    - save()/load(): Persistence
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.training_date: Optional[datetime] = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Feature matrix (samples, features)
            y: Target values
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Generate prediction.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            ModelPrediction with direction, magnitude, confidence
        """
        pass
    
    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Online update with new data (optional).
        
        Default implementation does nothing.
        Override for online learning models.
        """
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {}
    
    def save(self, path: str):
        """Save model to disk."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'name': self.name,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
        }
        
        # Add model-specific state
        state.update(self._get_state())
        
        joblib.dump(state, filepath)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'BaseModel':
        """Load model from disk."""
        state = joblib.load(path)
        
        self.name = state['name']
        self.is_trained = state['is_trained']
        self.training_date = state['training_date']
        self.feature_names = state['feature_names']
        self.metadata = state['metadata']
        
        self._set_state(state)
        
        logger.info(f"Model loaded from {path}")
        return self
    
    def _get_state(self) -> Dict[str, Any]:
        """Get model-specific state for serialization. Override in subclass."""
        return {}
    
    def _set_state(self, state: Dict[str, Any]):
        """Set model-specific state from serialization. Override in subclass."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"
