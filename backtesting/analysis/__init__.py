"""
Backtesting Analysis Module - Performance metrics and analysis.
"""

from .metrics import (
    PerformanceMetrics,
    RiskMetrics,
    TradeMetrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_omega_ratio,
    calculate_tail_ratio,
)

from .drawdown_analysis import (
    DrawdownAnalyzer,
    DrawdownPeriod,
    UnderwaterAnalysis,
)

from .regime_analysis import (
    RegimeAnalyzer,
    RegimePerformance,
)


__all__ = [
    'PerformanceMetrics',
    'RiskMetrics',
    'TradeMetrics',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_information_ratio',
    'calculate_omega_ratio',
    'calculate_tail_ratio',
    'DrawdownAnalyzer',
    'DrawdownPeriod',
    'UnderwaterAnalysis',
    'RegimeAnalyzer',
    'RegimePerformance',
]
