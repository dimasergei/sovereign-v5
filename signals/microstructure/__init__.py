"""
Microstructure Signals Module - Order flow and market microstructure analysis.

Provides modular components for microstructure analysis:
- VPIN (Volume-synchronized Probability of Informed Trading)
- Order flow imbalance and analysis
- Trade imbalance detection

These signals are used to detect informed trading, toxic flow,
and institutional order activity.
"""

from .vpin import VPINCalculator, VPINSignal, VPINBucket
from .order_flow import OrderFlowAnalyzer, OrderFlowSignal, LiquidityState
from .trade_imbalance import (
    TradeImbalanceDetector,
    TradeImbalanceSignal,
    ImbalanceType,
)


__all__ = [
    # VPIN
    'VPINCalculator',
    'VPINSignal',
    'VPINBucket',
    # Order Flow
    'OrderFlowAnalyzer',
    'OrderFlowSignal',
    'LiquidityState',
    # Trade Imbalance
    'TradeImbalanceDetector',
    'TradeImbalanceSignal',
    'ImbalanceType',
]
