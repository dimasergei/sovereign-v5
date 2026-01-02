"""
Multi-Timeframe Confluence Module - Signal alignment across timeframes.

Provides tools for analyzing signal agreement across multiple timeframes
and generating confluence-based trading signals.
"""

from .multi_timeframe import (
    MultiTimeframeAnalyzer,
    ConfluenceSignal,
    TimeframeData,
    TimeframeTrend,
    Timeframe,
)


__all__ = [
    'MultiTimeframeAnalyzer',
    'ConfluenceSignal',
    'TimeframeData',
    'TimeframeTrend',
    'Timeframe',
]
