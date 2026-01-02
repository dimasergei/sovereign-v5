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

from .analysis.metrics import (
    PerformanceMetrics,
    RiskMetrics,
    TradeMetrics,
    MetricsCalculator,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_omega_ratio,
    calculate_tail_ratio,
)

from .analysis.drawdown_analysis import (
    DrawdownAnalyzer,
    DrawdownPeriod,
    UnderwaterAnalysis,
)

from .analysis.regime_analysis import (
    RegimeAnalyzer,
    RegimePerformance,
)

from .reporting.tearsheet import (
    TearsheetGenerator,
    TearsheetReport,
    generate_html_report,
    generate_json_report,
)


__all__ = [
    # Engine
    'VectorizedBacktester',
    'BacktestConfig',
    'BacktestResults',
    'Trade',
    'MonteCarloSimulator',
    # Validation
    'WalkForwardValidator',
    'WalkForwardAnalysis',
    'ParameterStability',
    'OverfitDetector',
    # Metrics
    'PerformanceMetrics',
    'RiskMetrics',
    'TradeMetrics',
    'MetricsCalculator',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_information_ratio',
    'calculate_omega_ratio',
    'calculate_tail_ratio',
    # Drawdown
    'DrawdownAnalyzer',
    'DrawdownPeriod',
    'UnderwaterAnalysis',
    # Regime
    'RegimeAnalyzer',
    'RegimePerformance',
    # Reporting
    'TearsheetGenerator',
    'TearsheetReport',
    'generate_html_report',
    'generate_json_report',
]
