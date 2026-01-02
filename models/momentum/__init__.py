"""
Momentum Models Package - Trend following strategies.
"""

from .trend_following import (
    TrendFollowingModel,
    TrendSignal,
    AdaptiveTrendDetector,
)


__all__ = [
    'TrendFollowingModel',
    'TrendSignal',
    'AdaptiveTrendDetector',
]
