"""
ML Signal Adapter - Bridges ML model predictions to trading signals.

Provides three operating modes:
- 'filter': ML filters rule-based signals (vetoes low confidence)
- 'confirm': Only trade when rule-based AND ML agree
- 'replace': ML generates signals directly (full ML mode)

Default is 'filter' mode for safe integration.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import joblib

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'storage' / 'models'


@dataclass
class MLPrediction:
    """Prediction from ML ensemble."""
    direction: float = 0.0  # -1 to 1
    confidence: float = 0.0  # 0 to 1
    magnitude: float = 0.0  # Expected move
    regime: str = "unknown"
    model_agreement: float = 0.0  # How many models agree
    individual_predictions: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


class MLSignalAdapter:
    """
    Adapts ML model predictions to trading signals.

    Can operate in three modes:
    - 'filter': ML filters rule-based signals (vetoes low confidence)
    - 'confirm': Only trade when rule-based AND ML agree
    - 'replace': ML generates signals directly

    Usage:
        adapter = MLSignalAdapter(mode='filter', confidence_threshold=0.5)

        # In signal generator:
        if adapter.is_ready():
            signal = adapter.filter_signal(rule_signal, features)
    """

    def __init__(
        self,
        mode: str = 'filter',
        confidence_threshold: float = 0.5,
        load_models: bool = True
    ):
        """
        Initialize ML signal adapter.

        Args:
            mode: Operating mode ('filter', 'confirm', 'replace')
            confidence_threshold: Minimum ML confidence to act
            load_models: Whether to load models on init
        """
        self.mode = mode
        self.confidence_threshold = confidence_threshold

        self.models_loaded = False
        self.ensemble = None
        self.hmm_model = None
        self.lstm_model = None
        self.transformer_model = None

        self._prediction_cache = {}
        self._cache_ttl_seconds = 60

        if load_models:
            self._load_models()

    def _load_models(self) -> bool:
        """Load trained models from disk."""

        logger.info("Loading ML models...")

        models_found = 0

        # Load HMM
        hmm_path = MODEL_PATH / 'hmm' / 'hmm_regime.joblib'
        if hmm_path.exists():
            try:
                from models.regime.hmm import HMMRegimeModel
                self.hmm_model = HMMRegimeModel()
                self.hmm_model.load(str(hmm_path))
                models_found += 1
                logger.info(f"  Loaded HMM model from {hmm_path}")
            except Exception as e:
                logger.warning(f"  Failed to load HMM: {e}")

        # Load LSTM
        lstm_path = MODEL_PATH / 'lstm' / 'lstm_attention.joblib'
        if lstm_path.exists():
            try:
                from models.temporal.lstm_attention import LSTMAttentionModel
                self.lstm_model = LSTMAttentionModel()
                self.lstm_model.load(str(lstm_path))
                models_found += 1
                logger.info(f"  Loaded LSTM model from {lstm_path}")
            except Exception as e:
                logger.warning(f"  Failed to load LSTM: {e}")

        # Load Transformer
        transformer_path = MODEL_PATH / 'transformer' / 'transformer.joblib'
        if transformer_path.exists():
            try:
                from models.temporal.transformer import TemporalTransformer
                self.transformer_model = TemporalTransformer()
                self.transformer_model.load(str(transformer_path))
                models_found += 1
                logger.info(f"  Loaded Transformer model from {transformer_path}")
            except Exception as e:
                logger.warning(f"  Failed to load Transformer: {e}")

        # Load ensemble config
        ensemble_path = MODEL_PATH / 'ensemble' / 'ensemble_config.joblib'
        if ensemble_path.exists():
            try:
                self.ensemble_config = joblib.load(ensemble_path)
                logger.info(f"  Loaded ensemble config")
            except Exception as e:
                logger.warning(f"  Failed to load ensemble config: {e}")

        if models_found > 0:
            self.models_loaded = True
            logger.info(f"ML models loaded: {models_found} available")
        else:
            logger.warning("No ML models found - run training first:")
            logger.warning("  python scripts/train_models.py --all")

        return self.models_loaded

    def is_ready(self) -> bool:
        """Check if ML models are loaded and ready."""
        return self.models_loaded

    def get_ml_prediction(
        self,
        features: np.ndarray,
        symbol: str = ""
    ) -> MLPrediction:
        """
        Get ML prediction for current market state.

        Args:
            features: Feature array (sequence for LSTM/Transformer, or raw for HMM)
            symbol: Trading symbol for caching

        Returns:
            MLPrediction with direction, confidence, and metadata
        """

        if not self.models_loaded:
            return MLPrediction(
                error="models_not_loaded"
            )

        # Check cache
        cache_key = f"{symbol}_{hash(features.tobytes()) if isinstance(features, np.ndarray) else 'no_features'}"
        if cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self._cache_ttl_seconds:
                return cached

        predictions = {}
        directions = []
        confidences = []

        # Get HMM regime prediction
        if self.hmm_model is not None:
            try:
                # HMM expects 2D observations (returns, volatility)
                if len(features.shape) == 3:
                    # Extract returns from sequence
                    returns = features[:, :, 0].flatten()[-50:]  # Last 50 returns
                    volatility = np.std(returns) * np.ones_like(returns)
                    obs = np.column_stack([returns, volatility])
                else:
                    obs = features

                hmm_pred = self.hmm_model.predict(obs)
                if hasattr(hmm_pred, 'direction'):
                    predictions['hmm'] = hmm_pred.direction
                    directions.append(hmm_pred.direction)
                    confidences.append(getattr(hmm_pred, 'confidence', 0.5))
            except Exception as e:
                logger.debug(f"HMM prediction failed: {e}")

        # Get LSTM prediction
        if self.lstm_model is not None:
            try:
                if len(features.shape) == 2:
                    # Add batch dimension
                    features_batch = features.reshape(1, *features.shape)
                else:
                    features_batch = features

                lstm_pred = self.lstm_model.predict(features_batch)
                predictions['lstm'] = lstm_pred.direction
                directions.append(lstm_pred.direction)
                confidences.append(lstm_pred.confidence)
            except Exception as e:
                logger.debug(f"LSTM prediction failed: {e}")

        # Get Transformer prediction
        if self.transformer_model is not None:
            try:
                if len(features.shape) == 2:
                    features_batch = features.reshape(1, *features.shape)
                else:
                    features_batch = features

                trans_pred = self.transformer_model.predict(features_batch)
                predictions['transformer'] = trans_pred.direction
                directions.append(trans_pred.direction)
                confidences.append(trans_pred.confidence)
            except Exception as e:
                logger.debug(f"Transformer prediction failed: {e}")

        if not predictions:
            return MLPrediction(error="all_predictions_failed")

        # Combine predictions
        directions = np.array(directions)
        confidences = np.array(confidences)

        # Weighted average (confidence-weighted)
        if len(directions) > 0:
            weights = confidences / (confidences.sum() + 1e-10)
            combined_direction = np.sum(directions * weights)
            combined_confidence = np.mean(confidences)

            # Calculate agreement
            signs = np.sign(directions)
            if len(signs) > 1:
                agreement = np.mean(signs == np.sign(combined_direction))
            else:
                agreement = 1.0
        else:
            combined_direction = 0.0
            combined_confidence = 0.0
            agreement = 0.0

        # Detect regime from HMM if available
        regime = "unknown"
        if 'hmm' in predictions and self.hmm_model is not None:
            try:
                if hasattr(self.hmm_model, 'regime_mapping'):
                    regime = self.hmm_model.regime_mapping.get(
                        int(predictions['hmm']), "unknown"
                    )
            except:
                pass

        result = MLPrediction(
            direction=float(combined_direction),
            confidence=float(combined_confidence),
            magnitude=abs(combined_direction) * 0.01,  # Simple magnitude estimate
            regime=regime,
            model_agreement=float(agreement),
            individual_predictions=predictions
        )

        # Cache result
        self._prediction_cache[cache_key] = result

        return result

    def filter_signal(
        self,
        rule_signal: Any,
        features: np.ndarray,
        symbol: str = ""
    ) -> Any:
        """
        Filter a rule-based signal using ML confidence.

        Args:
            rule_signal: TradingSignal from rule-based generator
            features: Feature array for ML prediction
            symbol: Trading symbol

        Returns:
            Modified TradingSignal (possibly vetoed)
        """

        ml_pred = self.get_ml_prediction(features, symbol)

        if ml_pred.error:
            # ML unavailable, pass through unchanged
            return rule_signal

        # Store ML metadata in signal
        if hasattr(rule_signal, '__dict__'):
            rule_signal.ml_confidence = ml_pred.confidence
            rule_signal.ml_direction = ml_pred.direction
            rule_signal.ml_regime = ml_pred.regime
            rule_signal.ml_agreement = ml_pred.model_agreement

        if self.mode == 'filter':
            # Veto low-confidence signals
            if ml_pred.confidence < self.confidence_threshold:
                logger.debug(
                    f"ML VETO: conf={ml_pred.confidence:.2f} < {self.confidence_threshold}"
                )
                rule_signal.direction = 0.0
                rule_signal.action = "neutral"
                rule_signal.entry_reason = f"ML_VETO (conf={ml_pred.confidence:.2f})"
                return rule_signal

            # Boost confidence if ML strongly agrees
            if abs(ml_pred.direction) > 0.5 and ml_pred.model_agreement > 0.7:
                if np.sign(ml_pred.direction) == np.sign(rule_signal.direction):
                    # Agreement - boost confidence
                    rule_signal.confidence = min(
                        1.0,
                        rule_signal.confidence * 1.2
                    )
                    if hasattr(rule_signal, 'contributing_models'):
                        rule_signal.contributing_models.extend(
                            list(ml_pred.individual_predictions.keys())
                        )

        elif self.mode == 'confirm':
            # Require ML and rule-based agreement
            rule_dir = np.sign(rule_signal.direction)
            ml_dir = np.sign(ml_pred.direction)

            if rule_dir != ml_dir and rule_dir != 0:
                # Disagreement - no trade
                logger.debug(
                    f"ML DISAGREE: rule={rule_dir}, ml={ml_dir}"
                )
                rule_signal.direction = 0.0
                rule_signal.action = "neutral"
                rule_signal.entry_reason = "NO_ML_CONSENSUS"
            else:
                # Agreement - average confidences
                rule_signal.confidence = (
                    rule_signal.confidence + ml_pred.confidence
                ) / 2

        elif self.mode == 'replace':
            # Full ML mode - ignore rule signal
            if abs(ml_pred.direction) > 0.3 and ml_pred.confidence > self.confidence_threshold:
                rule_signal.direction = ml_pred.direction
                rule_signal.action = "long" if ml_pred.direction > 0 else "short"
                rule_signal.confidence = ml_pred.confidence
                rule_signal.entry_reason = f"ML_SIGNAL (regime={ml_pred.regime})"
            else:
                rule_signal.direction = 0.0
                rule_signal.action = "neutral"
                rule_signal.entry_reason = "ML_NO_SIGNAL"

        return rule_signal

    def get_regime(self, features: np.ndarray) -> str:
        """Get current market regime from HMM."""

        if self.hmm_model is None:
            return "unknown"

        try:
            pred = self.get_ml_prediction(features)
            return pred.regime
        except:
            return "unknown"

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models."""

        return {
            'models_loaded': self.models_loaded,
            'mode': self.mode,
            'confidence_threshold': self.confidence_threshold,
            'hmm_loaded': self.hmm_model is not None,
            'lstm_loaded': self.lstm_model is not None,
            'transformer_loaded': self.transformer_model is not None,
            'cache_size': len(self._prediction_cache)
        }

    def clear_cache(self):
        """Clear prediction cache."""
        self._prediction_cache = {}
