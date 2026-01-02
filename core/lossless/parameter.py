"""
Lossless Parameter System - Base Classes.

The foundation of the Lossless Principle: ALL parameters must be
derived from market data, never hardcoded.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any, Optional, List, Tuple
from collections import deque
import logging

from ..exceptions import InsufficientDataError, NotCalibratedError


logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a parameter calibration."""
    name: str
    value: float
    confidence: float
    method: str
    calibrated_at: datetime
    market_context: dict = field(default_factory=dict)
    
    def get(self) -> float:
        """Get the calibrated value."""
        return self.value


class LosslessParameter:
    """
    Self-calibrating parameter that derives its value from market observation.
    
    INVARIANT: No hardcoded default values allowed. Initial value must come
    from historical data analysis during initialization.
    
    Usage:
        # Define derivation function
        def derive_rsi_period(market_data):
            # Use spectral analysis, entropy, etc.
            return optimal_period
        
        # Create parameter
        rsi_period = LosslessParameter(
            name="rsi_period",
            derivation_function=derive_rsi_period,
            min_samples=100
        )
        
        # Calibrate from data
        rsi_period.calibrate(historical_df)
        
        # Use the value
        period = rsi_period.get()
    """
    
    def __init__(
        self,
        name: str,
        derivation_function: Callable,
        min_samples: int = 100,
        recalibration_threshold: float = 0.15,
        history_size: int = 100
    ):
        """
        Initialize a lossless parameter.
        
        Args:
            name: Parameter name for logging
            derivation_function: Function that derives value from market data
            min_samples: Minimum data points required for calibration
            recalibration_threshold: Threshold for triggering recalibration
            history_size: Number of historical calibrations to track
        """
        self.name = name
        self.derive = derivation_function
        self.min_samples = min_samples
        self.recalibration_threshold = recalibration_threshold
        
        self.value: Optional[float] = None
        self.confidence: float = 0.0
        self.last_calibration: Optional[datetime] = None
        self.calibration_count: int = 0
        
        self.history: deque = deque(maxlen=history_size)
    
    def calibrate(self, market_data: Any) -> CalibrationResult:
        """
        Derive parameter value from market data.
        
        Args:
            market_data: Data to derive parameter from (usually DataFrame)
            
        Returns:
            CalibrationResult with derived value
            
        Raises:
            InsufficientDataError: If not enough data for calibration
        """
        # Check data sufficiency
        data_len = len(market_data) if hasattr(market_data, '__len__') else 0
        if data_len < self.min_samples:
            raise InsufficientDataError(
                f"Need {self.min_samples} samples for {self.name}, got {data_len}"
            )
        
        # Derive the value
        try:
            new_value = self.derive(market_data)
        except Exception as e:
            logger.error(f"Derivation failed for {self.name}: {e}")
            raise
        
        # Calculate confidence based on data quality and stability
        confidence = self._compute_confidence(market_data, new_value)
        
        # Store result
        self.value = new_value
        self.confidence = confidence
        self.last_calibration = datetime.now()
        self.calibration_count += 1
        
        result = CalibrationResult(
            name=self.name,
            value=new_value,
            confidence=confidence,
            method="derivation",
            calibrated_at=self.last_calibration,
            market_context={"data_points": data_len}
        )
        
        self.history.append(result)
        
        logger.info(
            f"Calibrated {self.name}: {new_value:.4f} "
            f"(confidence: {confidence:.2f}, samples: {data_len})"
        )
        
        return result
    
    def get(self) -> float:
        """
        Get current calibrated value.
        
        Returns:
            The calibrated parameter value
            
        Raises:
            NotCalibratedError: If parameter hasn't been calibrated
        """
        if self.value is None:
            raise NotCalibratedError(
                f"Parameter '{self.name}' not yet calibrated from market data"
            )
        return self.value
    
    def get_with_default(self, default: float) -> float:
        """
        Get value with fallback default.
        
        Note: Using defaults defeats the lossless principle.
        This should only be used during system initialization.
        
        Args:
            default: Fallback value if not calibrated
            
        Returns:
            Calibrated value or default
        """
        if self.value is None:
            logger.warning(
                f"Using default value for {self.name}: {default} "
                f"(lossless principle violation)"
            )
            return default
        return self.value
    
    def needs_recalibration(self, current_data: Any = None) -> bool:
        """
        Check if parameter needs recalibration.
        
        Args:
            current_data: Optional current data to compare against
            
        Returns:
            True if recalibration recommended
        """
        if self.value is None:
            return True
        
        if self.last_calibration is None:
            return True
        
        # Check age (recalibrate every 4 hours by default)
        age_hours = (datetime.now() - self.last_calibration).total_seconds() / 3600
        if age_hours > 4:
            return True
        
        # If we have current data, check for regime shift
        if current_data is not None:
            try:
                new_value = self.derive(current_data)
                change_pct = abs(new_value - self.value) / abs(self.value) if self.value != 0 else 0
                if change_pct > self.recalibration_threshold:
                    logger.info(
                        f"{self.name} changed by {change_pct:.1%}, recalibration needed"
                    )
                    return True
            except:
                pass
        
        return False
    
    def get_stability(self) -> float:
        """
        Calculate parameter stability over recent calibrations.
        
        Returns:
            Stability score 0-1 (1 = very stable)
        """
        if len(self.history) < 3:
            return 0.5  # Not enough history
        
        values = [r.value for r in self.history]
        mean = sum(values) / len(values)
        
        if mean == 0:
            return 0.5
        
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        cv = (variance ** 0.5) / abs(mean)  # Coefficient of variation
        
        # Convert CV to stability score (lower CV = higher stability)
        stability = max(0, min(1, 1 - cv))
        
        return stability
    
    def _compute_confidence(self, market_data: Any, value: float) -> float:
        """
        Compute confidence in the derived value.
        
        Args:
            market_data: Source data
            value: Derived value
            
        Returns:
            Confidence score 0-1
        """
        confidence = 0.5  # Base confidence
        
        # More data = higher confidence
        data_len = len(market_data) if hasattr(market_data, '__len__') else 0
        data_factor = min(1.0, data_len / (self.min_samples * 5))
        confidence += 0.25 * data_factor
        
        # Stability from history
        if len(self.history) >= 3:
            stability = self.get_stability()
            confidence += 0.25 * stability
        
        return min(1.0, confidence)
    
    def __repr__(self) -> str:
        return (
            f"LosslessParameter(name='{self.name}', value={self.value}, "
            f"confidence={self.confidence:.2f})"
        )


class ParameterSet:
    """
    Collection of lossless parameters that are calibrated together.
    
    Usage:
        params = ParameterSet()
        params.add("fast_period", derive_fast_period)
        params.add("slow_period", derive_slow_period)
        
        params.calibrate_all(market_data)
        
        fast = params.get("fast_period")
        slow = params.get("slow_period")
    """
    
    def __init__(self):
        self.parameters: dict[str, LosslessParameter] = {}
        self.last_full_calibration: Optional[datetime] = None
    
    def add(
        self,
        name: str,
        derivation_function: Callable,
        min_samples: int = 100
    ) -> 'ParameterSet':
        """
        Add a parameter to the set.
        
        Args:
            name: Parameter name
            derivation_function: Function to derive value
            min_samples: Minimum samples required
            
        Returns:
            Self for chaining
        """
        self.parameters[name] = LosslessParameter(
            name=name,
            derivation_function=derivation_function,
            min_samples=min_samples
        )
        return self
    
    def calibrate_all(self, market_data: Any) -> dict[str, CalibrationResult]:
        """
        Calibrate all parameters from market data.
        
        Args:
            market_data: Source data for calibration
            
        Returns:
            Dictionary of calibration results
        """
        results = {}
        
        for name, param in self.parameters.items():
            try:
                results[name] = param.calibrate(market_data)
            except Exception as e:
                logger.error(f"Failed to calibrate {name}: {e}")
                results[name] = None
        
        self.last_full_calibration = datetime.now()
        
        successful = sum(1 for r in results.values() if r is not None)
        logger.info(
            f"Calibrated {successful}/{len(self.parameters)} parameters"
        )
        
        return results
    
    def get(self, name: str) -> float:
        """Get a parameter value."""
        if name not in self.parameters:
            raise KeyError(f"Unknown parameter: {name}")
        return self.parameters[name].get()
    
    def get_all(self) -> dict[str, float]:
        """Get all parameter values as a dictionary."""
        return {
            name: param.get() if param.value is not None else None
            for name, param in self.parameters.items()
        }
    
    def needs_recalibration(self) -> bool:
        """Check if any parameter needs recalibration."""
        return any(p.needs_recalibration() for p in self.parameters.values())
    
    def get_confidence_report(self) -> dict[str, float]:
        """Get confidence scores for all parameters."""
        return {
            name: param.confidence
            for name, param in self.parameters.items()
        }
