"""
Signal Quality Scoring Module

Assesses the quality of trading signals based on multiple factors:
- Signal age and freshness (staleness detection)
- Model agreement and confidence
- Regime fitness (signal-regime alignment)
- Confluence with other signals
- Historical accuracy tracking

Quality scoring is used to:
1. Filter out low-quality signals
2. Adjust position sizing based on quality
3. Prioritize signals when multiple opportunities exist
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SignalQuality:
    """Quality assessment for a trading signal."""
    overall_score: float  # 0 to 1

    # Component scores
    freshness_score: float  # How recent the signal is
    confidence_score: float  # Model confidence
    agreement_score: float  # Model agreement
    regime_fit_score: float  # How well signal fits current regime
    confluence_score: float  # Multi-factor alignment

    # Flags
    is_stale: bool = False
    passes_threshold: bool = True

    # Details
    age_seconds: float = 0.0
    decay_factor: float = 1.0

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_position_scalar(self) -> float:
        """Calculate position size multiplier based on quality."""
        if self.is_stale:
            return 0.0
        if not self.passes_threshold:
            return 0.0
        return self.overall_score


@dataclass
class SignalMetrics:
    """Metrics for tracking signal performance over time."""
    signal_id: str
    direction: float
    confidence: float
    timestamp: datetime

    # Outcome tracking (filled after result known)
    actual_return: Optional[float] = None
    was_correct: Optional[bool] = None
    hold_time_minutes: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalQualityScorer:
    """
    Scores trading signals based on multiple quality factors.

    Quality assessment helps filter low-quality signals and adjust
    position sizing based on signal reliability indicators.

    Usage:
        scorer = SignalQualityScorer(
            stale_threshold_seconds=300,
            min_quality_threshold=0.4
        )

        quality = scorer.score_signal(
            direction=0.7,
            confidence=0.65,
            model_agreement=0.8,
            regime="trending_up",
            signal_regime_match=True,
            signal_timestamp=signal_time,
            confluence_score=0.75
        )

        if quality.passes_threshold:
            position_size *= quality.get_position_scalar()
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 300,  # 5 minutes
        min_quality_threshold: float = 0.4,
        confidence_weight: float = 0.25,
        agreement_weight: float = 0.20,
        freshness_weight: float = 0.20,
        regime_fit_weight: float = 0.20,
        confluence_weight: float = 0.15,
        decay_half_life_seconds: float = 120  # Signal decays to half in 2 min
    ):
        """
        Initialize signal quality scorer.

        Args:
            stale_threshold_seconds: Time after which signal is considered stale
            min_quality_threshold: Minimum quality score to pass
            confidence_weight: Weight for model confidence
            agreement_weight: Weight for model agreement
            freshness_weight: Weight for signal freshness
            regime_fit_weight: Weight for regime fitness
            confluence_weight: Weight for confluence
            decay_half_life_seconds: Half-life for signal decay
        """
        self.stale_threshold_seconds = stale_threshold_seconds
        self.min_quality_threshold = min_quality_threshold
        self.decay_half_life_seconds = decay_half_life_seconds

        # Weights must sum to 1
        total = (confidence_weight + agreement_weight + freshness_weight +
                regime_fit_weight + confluence_weight)
        self.weights = {
            'confidence': confidence_weight / total,
            'agreement': agreement_weight / total,
            'freshness': freshness_weight / total,
            'regime_fit': regime_fit_weight / total,
            'confluence': confluence_weight / total
        }

        # Historical tracking
        self.signal_history: deque = deque(maxlen=1000)
        self.accuracy_by_model: Dict[str, deque] = {}
        self.accuracy_by_regime: Dict[str, deque] = {}

    def score_signal(
        self,
        direction: float,
        confidence: float,
        model_agreement: float = 1.0,
        regime: str = "unknown",
        signal_regime_match: bool = True,
        signal_timestamp: datetime = None,
        confluence_score: float = 0.5,
        contributing_models: List[str] = None
    ) -> SignalQuality:
        """
        Score a trading signal's quality.

        Args:
            direction: Signal direction (-1 to 1)
            confidence: Model confidence (0 to 1)
            model_agreement: Agreement between models (0 to 1)
            regime: Current market regime
            signal_regime_match: Whether signal type fits regime
            signal_timestamp: When signal was generated
            confluence_score: Multi-timeframe/factor confluence
            contributing_models: List of contributing model names

        Returns:
            SignalQuality with overall score and component scores
        """
        now = datetime.now()
        signal_timestamp = signal_timestamp or now

        # 1. Freshness score
        age_seconds = (now - signal_timestamp).total_seconds()
        freshness_score, decay_factor = self._calculate_freshness(age_seconds)

        # 2. Is stale?
        is_stale = age_seconds > self.stale_threshold_seconds

        # 3. Confidence score (apply decay)
        confidence_score = confidence * decay_factor

        # 4. Agreement score
        agreement_score = model_agreement

        # 5. Regime fit score
        regime_fit_score = self._calculate_regime_fit(
            direction, regime, signal_regime_match
        )

        # 6. Confluence score (external)
        # Already provided, just clamp
        confluence_score = min(1.0, max(0.0, confluence_score))

        # 7. Calculate overall score
        overall = (
            self.weights['confidence'] * confidence_score +
            self.weights['agreement'] * agreement_score +
            self.weights['freshness'] * freshness_score +
            self.weights['regime_fit'] * regime_fit_score +
            self.weights['confluence'] * confluence_score
        )

        # Apply direction strength penalty (weak direction = lower quality)
        direction_factor = 0.5 + 0.5 * abs(direction)
        overall *= direction_factor

        passes = overall >= self.min_quality_threshold and not is_stale

        return SignalQuality(
            overall_score=overall,
            freshness_score=freshness_score,
            confidence_score=confidence_score,
            agreement_score=agreement_score,
            regime_fit_score=regime_fit_score,
            confluence_score=confluence_score,
            is_stale=is_stale,
            passes_threshold=passes,
            age_seconds=age_seconds,
            decay_factor=decay_factor,
            metadata={
                'direction': direction,
                'regime': regime,
                'signal_regime_match': signal_regime_match,
                'contributing_models': contributing_models or [],
                'weights': self.weights.copy()
            }
        )

    def _calculate_freshness(self, age_seconds: float) -> Tuple[float, float]:
        """
        Calculate freshness score and decay factor.

        Uses exponential decay based on half-life.

        Returns:
            (freshness_score, decay_factor)
        """
        if age_seconds <= 0:
            return 1.0, 1.0

        # Exponential decay
        decay_constant = np.log(2) / self.decay_half_life_seconds
        decay_factor = np.exp(-decay_constant * age_seconds)

        # Freshness score (similar but more forgiving for young signals)
        if age_seconds < 10:
            freshness_score = 1.0
        elif age_seconds < self.stale_threshold_seconds:
            # Linear decay for freshness (more readable)
            freshness_score = 1.0 - (age_seconds / self.stale_threshold_seconds)
        else:
            freshness_score = 0.0

        return freshness_score, decay_factor

    def _calculate_regime_fit(
        self,
        direction: float,
        regime: str,
        signal_regime_match: bool
    ) -> float:
        """
        Calculate how well the signal fits the current regime.

        Args:
            direction: Signal direction
            regime: Current market regime
            signal_regime_match: External assessment of match

        Returns:
            Regime fit score (0 to 1)
        """
        if signal_regime_match:
            base_score = 0.9
        else:
            base_score = 0.5

        # Adjust based on specific regime-direction combinations
        regime_lower = regime.lower() if regime else "unknown"

        if "trend" in regime_lower:
            if "up" in regime_lower and direction > 0:
                base_score = 1.0  # Long in uptrend
            elif "down" in regime_lower and direction < 0:
                base_score = 1.0  # Short in downtrend
            elif abs(direction) < 0.3:
                base_score = 0.6  # Neutral in trending market

        elif "mean_revert" in regime_lower:
            # Mean reversion regime - contrarian better
            base_score = 0.8

        elif "volatile" in regime_lower or "high_vol" in regime_lower:
            # High volatility - reduce all signals
            base_score *= 0.7

        elif "choppy" in regime_lower or "ranging" in regime_lower:
            # Choppy market - reduce trend signals
            if abs(direction) > 0.5:
                base_score *= 0.6

        return min(1.0, base_score)

    def record_signal(
        self,
        signal_id: str,
        direction: float,
        confidence: float,
        contributing_models: List[str] = None,
        regime: str = None
    ):
        """
        Record a signal for historical tracking.

        Call this when generating a signal. Later call record_outcome
        to track accuracy.
        """
        metrics = SignalMetrics(
            signal_id=signal_id,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={
                'contributing_models': contributing_models or [],
                'regime': regime
            }
        )

        self.signal_history.append(metrics)

    def record_outcome(
        self,
        signal_id: str,
        actual_return: float,
        hold_time_minutes: float = None
    ):
        """
        Record the outcome of a previously recorded signal.

        Call this after a trade is closed to track model accuracy.
        """
        # Find signal in history
        for metrics in self.signal_history:
            if metrics.signal_id == signal_id:
                metrics.actual_return = actual_return
                metrics.was_correct = (np.sign(metrics.direction) == np.sign(actual_return))
                metrics.hold_time_minutes = hold_time_minutes

                # Update model accuracy
                for model in metrics.metadata.get('contributing_models', []):
                    if model not in self.accuracy_by_model:
                        self.accuracy_by_model[model] = deque(maxlen=100)
                    self.accuracy_by_model[model].append(
                        1.0 if metrics.was_correct else 0.0
                    )

                # Update regime accuracy
                regime = metrics.metadata.get('regime')
                if regime:
                    if regime not in self.accuracy_by_regime:
                        self.accuracy_by_regime[regime] = deque(maxlen=100)
                    self.accuracy_by_regime[regime].append(
                        1.0 if metrics.was_correct else 0.0
                    )

                break

    def get_model_accuracy(self, model_name: str) -> float:
        """Get historical accuracy for a specific model."""
        if model_name not in self.accuracy_by_model:
            return 0.5  # Default 50%

        history = list(self.accuracy_by_model[model_name])
        if not history:
            return 0.5

        return np.mean(history)

    def get_regime_accuracy(self, regime: str) -> float:
        """Get historical accuracy for signals in a specific regime."""
        if regime not in self.accuracy_by_regime:
            return 0.5

        history = list(self.accuracy_by_regime[regime])
        if not history:
            return 0.5

        return np.mean(history)

    def get_overall_accuracy(self) -> float:
        """Get overall historical signal accuracy."""
        completed = [m for m in self.signal_history if m.was_correct is not None]
        if not completed:
            return 0.5

        correct = sum(1 for m in completed if m.was_correct)
        return correct / len(completed)

    def get_statistics(self) -> Dict[str, Any]:
        """Get quality scoring statistics."""
        completed = [m for m in self.signal_history if m.was_correct is not None]

        return {
            'total_signals': len(self.signal_history),
            'completed_signals': len(completed),
            'overall_accuracy': self.get_overall_accuracy(),
            'model_accuracies': {
                model: self.get_model_accuracy(model)
                for model in self.accuracy_by_model.keys()
            },
            'regime_accuracies': {
                regime: self.get_regime_accuracy(regime)
                for regime in self.accuracy_by_regime.keys()
            },
            'weights': self.weights.copy(),
            'stale_threshold_seconds': self.stale_threshold_seconds,
            'min_quality_threshold': self.min_quality_threshold
        }


class SignalFilter:
    """
    Filters signals based on quality and other criteria.

    Works with SignalQualityScorer to reject low-quality signals
    before they reach execution.
    """

    def __init__(
        self,
        scorer: SignalQualityScorer = None,
        min_direction: float = 0.2,
        min_confidence: float = 0.4,
        min_quality: float = 0.4,
        max_staleness_seconds: float = 300
    ):
        """
        Initialize signal filter.

        Args:
            scorer: SignalQualityScorer instance
            min_direction: Minimum absolute direction to trade
            min_confidence: Minimum model confidence
            min_quality: Minimum quality score
            max_staleness_seconds: Maximum signal age
        """
        self.scorer = scorer or SignalQualityScorer()
        self.min_direction = min_direction
        self.min_confidence = min_confidence
        self.min_quality = min_quality
        self.max_staleness_seconds = max_staleness_seconds

        # Track filter statistics
        self.total_checked = 0
        self.total_passed = 0
        self.rejection_reasons: Dict[str, int] = {
            'direction': 0,
            'confidence': 0,
            'quality': 0,
            'stale': 0
        }

    def should_trade(
        self,
        direction: float,
        confidence: float,
        quality: SignalQuality = None,
        signal_timestamp: datetime = None
    ) -> Tuple[bool, str]:
        """
        Check if signal should be traded.

        Args:
            direction: Signal direction
            confidence: Model confidence
            quality: Pre-computed quality score
            signal_timestamp: When signal was generated

        Returns:
            (should_trade, rejection_reason if any)
        """
        self.total_checked += 1

        # Check direction strength
        if abs(direction) < self.min_direction:
            self.rejection_reasons['direction'] += 1
            return False, f"Direction too weak: {abs(direction):.2f} < {self.min_direction}"

        # Check confidence
        if confidence < self.min_confidence:
            self.rejection_reasons['confidence'] += 1
            return False, f"Confidence too low: {confidence:.2f} < {self.min_confidence}"

        # Check staleness
        if signal_timestamp:
            age = (datetime.now() - signal_timestamp).total_seconds()
            if age > self.max_staleness_seconds:
                self.rejection_reasons['stale'] += 1
                return False, f"Signal stale: {age:.0f}s > {self.max_staleness_seconds}s"

        # Check quality score
        if quality:
            if not quality.passes_threshold:
                self.rejection_reasons['quality'] += 1
                return False, f"Quality score failed: {quality.overall_score:.2f}"

            if quality.overall_score < self.min_quality:
                self.rejection_reasons['quality'] += 1
                return False, f"Quality too low: {quality.overall_score:.2f} < {self.min_quality}"

        self.total_passed += 1
        return True, ""

    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics."""
        pass_rate = self.total_passed / self.total_checked if self.total_checked > 0 else 0

        return {
            'total_checked': self.total_checked,
            'total_passed': self.total_passed,
            'pass_rate': pass_rate,
            'rejection_reasons': self.rejection_reasons.copy()
        }

    def reset_statistics(self):
        """Reset filter statistics."""
        self.total_checked = 0
        self.total_passed = 0
        self.rejection_reasons = {k: 0 for k in self.rejection_reasons}
