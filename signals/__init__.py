"""
Signals Module - Signal generation and combination.

Provides comprehensive signal generation including:
- Core signal generation (SignalGenerator)
- Microstructure analysis (VPIN, order flow, trade imbalance)
- Multi-timeframe confluence analysis
- Signal quality scoring
"""

import logging

logger = logging.getLogger(__name__)

# Core signal generation
from .generator import SignalGenerator, TradingSignal

# Legacy microstructure (from generators/)
from .generators import (
    MicrostructureAnalyzer,
    MicrostructureSignals,
    TickData,
)

# Signal quality scoring
from .quality import (
    SignalQualityScorer,
    SignalQuality,
    SignalFilter,
    SignalMetrics,
)

# Multi-timeframe confluence
try:
    from .confluence import (
        MultiTimeframeAnalyzer,
        ConfluenceSignal,
        TimeframeData,
        TimeframeTrend,
        Timeframe,
    )
except ImportError as e:
    logger.debug(f"Confluence module not available: {e}")
    MultiTimeframeAnalyzer = None
    ConfluenceSignal = None
    TimeframeData = None
    TimeframeTrend = None
    Timeframe = None

# Modular microstructure signals
try:
    from .microstructure import (
        VPINCalculator,
        VPINSignal,
        VPINBucket,
        OrderFlowAnalyzer,
        OrderFlowSignal,
        LiquidityState,
        TradeImbalanceDetector,
        TradeImbalanceSignal,
        ImbalanceType,
    )
except ImportError as e:
    logger.debug(f"Microstructure module not available: {e}")
    VPINCalculator = None
    VPINSignal = None
    VPINBucket = None
    OrderFlowAnalyzer = None
    OrderFlowSignal = None
    LiquidityState = None
    TradeImbalanceDetector = None
    TradeImbalanceSignal = None
    ImbalanceType = None


__all__ = [
    # Core
    'SignalGenerator',
    'TradingSignal',
    # Legacy microstructure
    'MicrostructureAnalyzer',
    'MicrostructureSignals',
    'TickData',
    # Quality scoring
    'SignalQualityScorer',
    'SignalQuality',
    'SignalFilter',
    'SignalMetrics',
    # Confluence
    'MultiTimeframeAnalyzer',
    'ConfluenceSignal',
    'TimeframeData',
    'TimeframeTrend',
    'Timeframe',
    # Modular microstructure
    'VPINCalculator',
    'VPINSignal',
    'VPINBucket',
    'OrderFlowAnalyzer',
    'OrderFlowSignal',
    'LiquidityState',
    'TradeImbalanceDetector',
    'TradeImbalanceSignal',
    'ImbalanceType',
]
