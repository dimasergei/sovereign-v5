"""
Lossless Parameter System - Market-Derived Parameters.

The core of the Lossless Principle: ALL trading parameters
are derived from market data observation, never hardcoded.
"""

from .parameter import (
    LosslessParameter,
    CalibrationResult,
    ParameterSet,
)

from .calibrator import (
    MarketCalibrator,
    FullCalibrationResult,
)

from .entropy import (
    market_entropy,
    conditional_entropy,
    optimal_lookback_from_entropy,
    sample_entropy,
    approximate_entropy,
    permutation_entropy,
    multiscale_entropy,
    entropy_based_regime,
)

from .spectral import (
    spectral_density,
    dominant_cycle_period,
    find_all_cycles,
    derive_fast_period,
    derive_slow_period,
    spectral_coherence,
    hilbert_instantaneous_frequency,
    adaptive_period_from_spectrum,
)

from .fractal import (
    fractal_dimension,
    higuchi_fractal_dimension,
    katz_fractal_dimension,
    petrosian_fractal_dimension,
    regime_from_fractal_dimension,
    fractal_efficiency_ratio,
    derive_period_from_fractal,
    detrended_fluctuation_analysis,
)

from .hurst import (
    hurst_exponent,
    hurst_exponent_dfa,
    hurst_exponent_variance,
    rolling_hurst,
    regime_from_hurst,
    expected_range,
    optimal_holding_period,
    mean_reversion_halflife,
    derive_period_from_hurst,
    hurst_confidence,
    adaptive_hurst,
)


__all__ = [
    # Parameter classes
    'LosslessParameter',
    'CalibrationResult',
    'ParameterSet',
    'MarketCalibrator',
    'FullCalibrationResult',
    
    # Entropy
    'market_entropy',
    'conditional_entropy',
    'optimal_lookback_from_entropy',
    'sample_entropy',
    'approximate_entropy',
    'permutation_entropy',
    'multiscale_entropy',
    'entropy_based_regime',
    
    # Spectral
    'spectral_density',
    'dominant_cycle_period',
    'find_all_cycles',
    'derive_fast_period',
    'derive_slow_period',
    'spectral_coherence',
    'hilbert_instantaneous_frequency',
    'adaptive_period_from_spectrum',
    
    # Fractal
    'fractal_dimension',
    'higuchi_fractal_dimension',
    'katz_fractal_dimension',
    'petrosian_fractal_dimension',
    'regime_from_fractal_dimension',
    'fractal_efficiency_ratio',
    'derive_period_from_fractal',
    'detrended_fluctuation_analysis',
    
    # Hurst
    'hurst_exponent',
    'hurst_exponent_dfa',
    'hurst_exponent_variance',
    'rolling_hurst',
    'regime_from_hurst',
    'expected_range',
    'optimal_holding_period',
    'mean_reversion_halflife',
    'derive_period_from_hurst',
    'hurst_confidence',
    'adaptive_hurst',
]
