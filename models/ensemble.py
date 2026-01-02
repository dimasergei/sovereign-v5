"""
Ensemble Meta-Learner - Combines predictions from multiple models.

Uses dynamic weighting based on recent performance and
disagreement-aware position sizing.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from .base import BaseModel, ModelPrediction


logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble of models."""
    direction: float  # -1 to 1
    magnitude: float  # Expected move
    confidence: float  # 0 to 1
    action: str  # "long", "short", "neutral"
    position_size_scalar: float  # 0 to 1
    model_agreement: float  # 0 to 1
    contributing_models: List[str] = field(default_factory=list)
    disagreeing_models: List[str] = field(default_factory=list)
    individual_predictions: Dict[str, ModelPrediction] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformance:
    """Track individual model performance."""
    correct_predictions: int = 0
    total_predictions: int = 0
    cumulative_pnl: float = 0.0
    recent_accuracy: deque = field(default_factory=lambda: deque(maxlen=50))
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions
    
    @property
    def recent_accuracy_score(self) -> float:
        if len(self.recent_accuracy) == 0:
            return 0.5
        return sum(self.recent_accuracy) / len(self.recent_accuracy)


class EnsembleMetaLearner:
    """
    Combines predictions from multiple models using dynamic weighting.
    
    Features:
    - Weight adjustment based on recent performance
    - Disagreement-aware position sizing
    - Model diversity tracking
    - Regime-conditional model selection
    
    Usage:
        ensemble = EnsembleMetaLearner({
            'lstm': lstm_model,
            'mean_reversion': mr_model,
            'regime': regime_model
        })
        
        prediction = ensemble.predict(features)
        
        # After seeing actual outcome
        ensemble.update_weights(prediction, actual_return)
    """
    
    def __init__(
        self,
        models: Dict[str, BaseModel],
        min_confidence: float = 0.5,
        agreement_threshold: float = 0.6,
        weight_decay: float = 0.95
    ):
        """
        Initialize ensemble.
        
        Args:
            models: Dictionary of model_name -> BaseModel
            min_confidence: Minimum confidence to act
            agreement_threshold: Minimum agreement fraction to take position
            weight_decay: Weight decay factor for online learning
        """
        self.models = models
        self.min_confidence = min_confidence
        self.agreement_threshold = agreement_threshold
        self.weight_decay = weight_decay
        
        # Initialize equal weights
        n_models = len(models)
        self.weights = {name: 1.0 / n_models for name in models}
        
        # Performance tracking
        self.performance: Dict[str, ModelPerformance] = {
            name: ModelPerformance() for name in models
        }
        
        # Prediction history
        self.prediction_history: deque = deque(maxlen=1000)
        
        logger.info(f"Ensemble initialized with {n_models} models")
    
    def predict(self, features: Dict[str, np.ndarray]) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        
        Args:
            features: Dictionary of feature arrays for each model type
            
        Returns:
            EnsemblePrediction with combined signal
        """
        predictions = {}
        directions = []
        magnitudes = []
        confidences = []
        weights = []
        
        # Collect predictions from all models
        for name, model in self.models.items():
            try:
                # Get features for this model (use general if specific not provided)
                model_features = features.get(name, features.get('general', None))
                
                if model_features is None:
                    logger.warning(f"No features for model {name}")
                    continue
                
                pred = model.predict(model_features)
                predictions[name] = pred
                
                directions.append(pred.direction)
                magnitudes.append(pred.magnitude)
                confidences.append(pred.confidence)
                weights.append(self.weights[name])
                
            except Exception as e:
                logger.error(f"Model {name} prediction failed: {e}")
                continue
        
        if not predictions:
            return self._neutral_prediction()
        
        # Convert to arrays
        directions = np.array(directions)
        magnitudes = np.array(magnitudes)
        confidences = np.array(confidences)
        weights = np.array(weights)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate weighted averages
        weighted_direction = np.sum(directions * weights * confidences)
        weighted_magnitude = np.sum(magnitudes * weights * confidences)
        weighted_confidence = np.sum(confidences * weights)
        
        # Calculate model agreement
        agreement = self._calculate_agreement(directions)
        
        # Determine contributing and disagreeing models
        contributing = []
        disagreeing = []
        sign_direction = np.sign(weighted_direction)
        
        for name, pred in predictions.items():
            if np.sign(pred.direction) == sign_direction:
                contributing.append(name)
            else:
                disagreeing.append(name)
        
        # Calculate position size scalar
        position_scalar = self._calculate_position_scalar(
            weighted_confidence, agreement
        )
        
        # Determine action
        if agreement < self.agreement_threshold:
            action = "neutral"
            position_scalar = 0.0
        elif weighted_confidence < self.min_confidence:
            action = "neutral"
            position_scalar = 0.0
        elif abs(weighted_direction) < 0.2:
            action = "neutral"
            position_scalar = 0.0
        else:
            action = "long" if weighted_direction > 0 else "short"
        
        result = EnsemblePrediction(
            direction=float(weighted_direction),
            magnitude=float(weighted_magnitude),
            confidence=float(weighted_confidence),
            action=action,
            position_size_scalar=float(position_scalar),
            model_agreement=float(agreement),
            contributing_models=contributing,
            disagreeing_models=disagreeing,
            individual_predictions=predictions,
        )
        
        self.prediction_history.append(result)
        
        return result
    
    def update_weights(self, prediction: EnsemblePrediction, actual_return: float):
        """
        Update model weights based on prediction outcome.
        
        Args:
            prediction: The ensemble prediction that was made
            actual_return: The actual return that occurred
        """
        actual_direction = np.sign(actual_return)
        
        for name, pred in prediction.individual_predictions.items():
            perf = self.performance[name]
            perf.total_predictions += 1
            
            # Check if prediction was correct
            predicted_direction = np.sign(pred.direction)
            was_correct = predicted_direction == actual_direction
            
            if was_correct:
                perf.correct_predictions += 1
                perf.recent_accuracy.append(1)
            else:
                perf.recent_accuracy.append(0)
            
            # Update cumulative PnL
            perf.cumulative_pnl += pred.direction * actual_return
        
        # Adjust weights
        self._adjust_weights()
    
    def _adjust_weights(self):
        """Adjust model weights based on recent performance."""
        # Apply weight decay
        for name in self.weights:
            self.weights[name] *= self.weight_decay
        
        # Boost weights for better performers
        for name, perf in self.performance.items():
            accuracy = perf.recent_accuracy_score
            
            # Boost if accuracy > 50%
            if accuracy > 0.5:
                boost = 1 + (accuracy - 0.5) * 0.2  # Up to 10% boost
                self.weights[name] *= boost
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total
    
    def _calculate_agreement(self, directions: np.ndarray) -> float:
        """
        Calculate model agreement score.
        
        Args:
            directions: Array of direction predictions
            
        Returns:
            Agreement score 0 to 1
        """
        if len(directions) == 0:
            return 0.0
        
        if len(directions) == 1:
            return 1.0
        
        # Count models agreeing with majority
        mean_direction = np.mean(directions)
        majority_sign = np.sign(mean_direction)
        
        agreeing = np.sum(np.sign(directions) == majority_sign)
        
        return agreeing / len(directions)
    
    def _calculate_position_scalar(
        self,
        confidence: float,
        agreement: float
    ) -> float:
        """
        Calculate position size scaling factor.
        
        Args:
            confidence: Ensemble confidence
            agreement: Model agreement score
            
        Returns:
            Scalar 0 to 1 for position sizing
        """
        if confidence < self.min_confidence:
            return 0.0
        
        if agreement < self.agreement_threshold:
            return 0.0
        
        # Scale by both confidence and agreement
        scalar = confidence * agreement
        
        # Apply non-linear scaling (more conservative)
        scalar = scalar ** 1.5
        
        return min(1.0, scalar)
    
    def _neutral_prediction(self) -> EnsemblePrediction:
        """Return neutral prediction when no models available."""
        return EnsemblePrediction(
            direction=0.0,
            magnitude=0.0,
            confidence=0.0,
            action="neutral",
            position_size_scalar=0.0,
            model_agreement=0.0,
        )
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models."""
        return {
            name: {
                'accuracy': perf.accuracy,
                'recent_accuracy': perf.recent_accuracy_score,
                'total_predictions': perf.total_predictions,
                'cumulative_pnl': perf.cumulative_pnl,
            }
            for name, perf in self.performance.items()
        }
    
    def add_model(self, name: str, model: BaseModel, initial_weight: float = None):
        """Add a new model to the ensemble."""
        self.models[name] = model
        
        if initial_weight is None:
            initial_weight = 1.0 / (len(self.models) + 1)
        
        self.weights[name] = initial_weight
        self.performance[name] = ModelPerformance()
        
        # Re-normalize weights
        total = sum(self.weights.values())
        for n in self.weights:
            self.weights[n] /= total
        
        logger.info(f"Added model {name} to ensemble")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            del self.performance[name]
            
            # Re-normalize weights
            total = sum(self.weights.values())
            if total > 0:
                for n in self.weights:
                    self.weights[n] /= total
            
            logger.info(f"Removed model {name} from ensemble")
