"""
Tearsheet Generator - Professional performance report generation.

Creates comprehensive performance reports for strategy analysis.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from ..analysis.metrics import MetricsCalculator, PerformanceMetrics, RiskMetrics, TradeMetrics
from ..analysis.drawdown_analysis import DrawdownAnalyzer
from ..analysis.regime_analysis import RegimeAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TearsheetReport:
    """Complete tearsheet report container."""
    # Metadata
    strategy_name: str
    generated_at: str
    period_start: str
    period_end: str
    total_days: int

    # Performance
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    trade_metrics: Optional[Dict[str, Any]]

    # Drawdown analysis
    drawdown_summary: Dict[str, Any]
    prop_firm_safety: Dict[str, Any]

    # Regime analysis
    regime_analysis: Dict[str, Any]
    regime_recommendations: List[str]

    # Monthly/yearly returns
    monthly_returns: Dict[str, float]
    yearly_returns: Dict[str, float]

    # Rolling metrics
    rolling_sharpe: Dict[str, float]
    rolling_volatility: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TearsheetGenerator:
    """
    Generate comprehensive performance tearsheets.

    Features:
    - Full performance metrics
    - Drawdown analysis
    - Regime breakdown
    - Monthly/yearly returns
    - Rolling statistics
    - Prop firm safety check
    """

    def __init__(
        self,
        guardian_threshold_pct: float = 7.0,
        max_drawdown_pct: float = 8.0,
        periods_per_year: int = 252
    ):
        """
        Initialize generator.

        Args:
            guardian_threshold_pct: Guardian drawdown threshold
            max_drawdown_pct: Maximum allowed drawdown
            periods_per_year: Trading periods per year
        """
        self.guardian_threshold = guardian_threshold_pct
        self.max_drawdown = max_drawdown_pct
        self.periods_per_year = periods_per_year

        self.metrics_calc = MetricsCalculator(periods_per_year)
        self.drawdown_analyzer = DrawdownAnalyzer(guardian_threshold_pct)
        self.regime_analyzer = RegimeAnalyzer()

    def generate(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        market_data: pd.DataFrame = None,
        trades: List[Any] = None,
        strategy_name: str = "Strategy",
        benchmark_returns: pd.Series = None
    ) -> TearsheetReport:
        """
        Generate complete tearsheet report.

        Args:
            returns: Strategy returns series
            equity_curve: Equity curve series
            market_data: Market OHLCV data (for regime analysis)
            trades: List of trade objects
            strategy_name: Name of strategy
            benchmark_returns: Optional benchmark for comparison

        Returns:
            TearsheetReport object
        """
        logger.info(f"Generating tearsheet for {strategy_name}")

        # Calculate all metrics
        all_metrics = self.metrics_calc.calculate_all(
            returns, equity_curve, trades, benchmark_returns
        )

        # Drawdown analysis
        dd_analysis = self.drawdown_analyzer.analyze(equity_curve, trades)
        prop_safety = self.drawdown_analyzer.check_prop_firm_safety(
            equity_curve, self.max_drawdown, self.guardian_threshold
        )

        # Regime analysis
        if market_data is not None:
            regime_analysis = self.regime_analyzer.analyze(returns, market_data, trades)
            regime_recommendations = self.regime_analyzer.get_regime_recommendations(regime_analysis)
            # Convert regime performance objects to dicts
            for key in ['volatility_regimes', 'trend_regimes', 'structure_regimes']:
                if key in regime_analysis:
                    regime_analysis[key] = {
                        k: v.__dict__ if hasattr(v, '__dict__') else v
                        for k, v in regime_analysis[key].items()
                    }
            # Remove series from output
            regime_analysis.pop('regime_series', None)
        else:
            regime_analysis = {}
            regime_recommendations = []

        # Monthly/yearly returns
        monthly_returns = self._calculate_monthly_returns(returns)
        yearly_returns = self._calculate_yearly_returns(returns)

        # Rolling metrics
        rolling_sharpe = self._calculate_rolling_sharpe(returns)
        rolling_vol = self._calculate_rolling_volatility(returns)

        # Period info
        period_start = str(returns.index[0]) if hasattr(returns.index, '__getitem__') else "unknown"
        period_end = str(returns.index[-1]) if hasattr(returns.index, '__getitem__') else "unknown"
        total_days = len(returns)

        return TearsheetReport(
            strategy_name=strategy_name,
            generated_at=datetime.now().isoformat(),
            period_start=period_start,
            period_end=period_end,
            total_days=total_days,
            performance_metrics=all_metrics['performance'],
            risk_metrics=all_metrics['risk'],
            trade_metrics=all_metrics['trades'],
            drawdown_summary=dd_analysis['summary'],
            prop_firm_safety=prop_safety,
            regime_analysis=regime_analysis,
            regime_recommendations=regime_recommendations,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            rolling_sharpe=rolling_sharpe,
            rolling_volatility=rolling_vol,
        )

    def _calculate_monthly_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate monthly returns."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            return {}

        try:
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            return {str(k): float(v) for k, v in monthly.items()}
        except:
            return {}

    def _calculate_yearly_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate yearly returns."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            return {}

        try:
            yearly = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
            return {str(k.year): float(v) for k, v in yearly.items()}
        except:
            return {}

    def _calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 63  # ~3 months
    ) -> Dict[str, float]:
        """Calculate rolling Sharpe ratio."""
        try:
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()

            rolling_sharpe = np.sqrt(252) * rolling_mean / rolling_std

            # Sample at regular intervals
            sample_idx = np.linspace(window, len(returns) - 1, min(50, len(returns) - window)).astype(int)
            result = {}

            for idx in sample_idx:
                key = str(returns.index[idx]) if hasattr(returns.index, '__getitem__') else str(idx)
                result[key] = float(rolling_sharpe.iloc[idx]) if not np.isnan(rolling_sharpe.iloc[idx]) else 0.0

            return result
        except:
            return {}

    def _calculate_rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 21  # ~1 month
    ) -> Dict[str, float]:
        """Calculate rolling volatility."""
        try:
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

            # Sample at regular intervals
            sample_idx = np.linspace(window, len(returns) - 1, min(50, len(returns) - window)).astype(int)
            result = {}

            for idx in sample_idx:
                key = str(returns.index[idx]) if hasattr(returns.index, '__getitem__') else str(idx)
                result[key] = float(rolling_vol.iloc[idx]) if not np.isnan(rolling_vol.iloc[idx]) else 0.0

            return result
        except:
            return {}


def generate_html_report(report: TearsheetReport) -> str:
    """
    Generate HTML tearsheet report.

    Args:
        report: TearsheetReport object

    Returns:
        HTML string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report.strategy_name} - Performance Tearsheet</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
            .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
            .metric {{ padding: 15px; background: #f9f9f9; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #666; font-size: 12px; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .warning {{ color: #f39c12; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f5f5f5; }}
            .safety-badge {{ padding: 5px 10px; border-radius: 3px; font-weight: bold; }}
            .safe {{ background: #27ae60; color: white; }}
            .unsafe {{ background: #e74c3c; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{report.strategy_name}</h1>
                <p>Performance Tearsheet | {report.period_start} to {report.period_end}</p>
                <p>Generated: {report.generated_at}</p>
            </div>

            <div class="section">
                <h2>Key Metrics</h2>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-value {'positive' if report.performance_metrics.get('total_return_pct', 0) > 0 else 'negative'}">
                            {report.performance_metrics.get('total_return_pct', 0):.2f}%
                        </div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {'positive' if report.performance_metrics.get('sharpe_ratio', 0) > 1 else 'warning'}">
                            {report.performance_metrics.get('sharpe_ratio', 0):.2f}
                        </div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value negative">
                            {report.risk_metrics.get('max_drawdown', 0) * 100:.2f}%
                        </div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">
                            {report.trade_metrics.get('total_trades', 0) if report.trade_metrics else 0}
                        </div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Prop Firm Safety</h2>
                <span class="safety-badge {'safe' if report.prop_firm_safety.get('is_safe') else 'unsafe'}">
                    {'SAFE' if report.prop_firm_safety.get('is_safe') else 'NOT SAFE'}
                </span>
                <p>{report.prop_firm_safety.get('recommendation', '')}</p>
                <table>
                    <tr>
                        <td>Max Historical Drawdown</td>
                        <td>{report.prop_firm_safety.get('max_historical_drawdown', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Buffer to Guardian</td>
                        <td>{report.prop_firm_safety.get('buffer_to_guardian', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Guardian Breaches</td>
                        <td>{report.prop_firm_safety.get('guardian_breaches', 0)}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>CAGR</td><td>{report.performance_metrics.get('cagr', 0):.2f}%</td></tr>
                    <tr><td>Sortino Ratio</td><td>{report.performance_metrics.get('sortino_ratio', 0):.2f}</td></tr>
                    <tr><td>Calmar Ratio</td><td>{report.performance_metrics.get('calmar_ratio', 0):.2f}</td></tr>
                    <tr><td>Omega Ratio</td><td>{report.performance_metrics.get('omega_ratio', 0):.2f}</td></tr>
                    <tr><td>VaR (95%)</td><td>{report.performance_metrics.get('var_95', 0) * 100:.2f}%</td></tr>
                    <tr><td>CVaR (95%)</td><td>{report.performance_metrics.get('cvar_95', 0) * 100:.2f}%</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Trade Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Win Rate</td><td>{report.trade_metrics.get('win_rate', 0) * 100 if report.trade_metrics else 0:.1f}%</td></tr>
                    <tr><td>Profit Factor</td><td>{report.trade_metrics.get('profit_factor', 0) if report.trade_metrics else 0:.2f}</td></tr>
                    <tr><td>Expectancy</td><td>${report.trade_metrics.get('expectancy', 0) if report.trade_metrics else 0:.2f}</td></tr>
                    <tr><td>Avg Win</td><td>${report.trade_metrics.get('avg_winner', 0) if report.trade_metrics else 0:.2f}</td></tr>
                    <tr><td>Avg Loss</td><td>${report.trade_metrics.get('avg_loser', 0) if report.trade_metrics else 0:.2f}</td></tr>
                    <tr><td>Max Consecutive Wins</td><td>{report.trade_metrics.get('max_consecutive_wins', 0) if report.trade_metrics else 0}</td></tr>
                    <tr><td>Max Consecutive Losses</td><td>{report.trade_metrics.get('max_consecutive_losses', 0) if report.trade_metrics else 0}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Regime Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in report.regime_recommendations) if report.regime_recommendations else '<li>No specific recommendations</li>'}
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    return html


def generate_json_report(report: TearsheetReport) -> str:
    """
    Generate JSON tearsheet report.

    Args:
        report: TearsheetReport object

    Returns:
        JSON string
    """
    return json.dumps(report.to_dict(), indent=2, default=str)
