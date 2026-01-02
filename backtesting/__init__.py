"""
Backtesting Module - Strategy backtesting and analysis.
"""

from .engine.vectorized import (
    VectorizedBacktester,
    BacktestConfig,
    BacktestResults,
    Trade,
    MonteCarloSimulator
)

from .validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardAnalysis,
    ParameterStability,
    OverfitDetector
)


__all__ = [
    'VectorizedBacktester',
    'BacktestConfig',
    'BacktestResults',
    'Trade',
    'MonteCarloSimulator',
    'WalkForwardValidator',
    'WalkForwardAnalysis',
    'ParameterStability',
    'OverfitDetector',
]
