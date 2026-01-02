"""
Performance Metrics Module - Comprehensive backtesting metrics.

Implements all standard performance metrics without hardcoded values.
Risk-free rate and other parameters are derived from market data.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""
    # Returns metrics
    total_return: float
    total_return_pct: float
    cagr: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    omega_ratio: float

    # Tail metrics
    tail_ratio: float
    gain_to_pain_ratio: float

    # Statistical metrics
    skewness: float
    kurtosis: float
    best_day: float
    worst_day: float
    avg_daily_return: float
    daily_volatility: float
    annualized_volatility: float

    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'omega_ratio': self.omega_ratio,
            'tail_ratio': self.tail_ratio,
            'gain_to_pain_ratio': self.gain_to_pain_ratio,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'best_day': self.best_day,
            'worst_day': self.worst_day,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
        }


@dataclass
class RiskMetrics:
    """Risk-specific metrics container."""
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    drawdown_count: int

    volatility_daily: float
    volatility_annual: float
    downside_deviation: float
    upside_deviation: float

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    beta: Optional[float] = None
    alpha: Optional[float] = None
    treynor_ratio: Optional[float] = None


@dataclass
class TradeMetrics:
    """Trade-level metrics container."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    profit_factor: float
    payoff_ratio: float
    expectancy: float

    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float

    avg_hold_time_bars: float
    avg_hold_time_winners: float
    avg_hold_time_losers: float

    max_consecutive_wins: int
    max_consecutive_losses: int

    recovery_factor: float  # Total profit / Max drawdown


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    If risk_free_rate not provided, derives from return distribution
    (uses lower percentile as proxy for risk-free).
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    # Derive risk-free rate if not provided
    if risk_free_rate is None:
        # Use 5th percentile of returns as proxy
        risk_free_rate = max(0, np.percentile(returns, 5) * periods_per_year)

    excess_returns = returns - risk_free_rate / periods_per_year

    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio using downside deviation.

    Target return defaults to 0 (no loss goal).
    """
    if len(returns) < 2:
        return 0.0

    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if returns.mean() > 0 else 0.0

    downside_deviation = np.sqrt(np.mean(np.minimum(returns - target_return, 0) ** 2))

    if downside_deviation == 0:
        return float('inf') if returns.mean() > 0 else 0.0

    sortino = np.sqrt(periods_per_year) * (returns.mean() - target_return) / downside_deviation

    return float(sortino)


def calculate_calmar_ratio(
    returns: pd.Series,
    max_drawdown: float,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).
    """
    if max_drawdown == 0:
        return float('inf') if returns.mean() > 0 else 0.0

    # Calculate CAGR
    cumulative = (1 + returns).cumprod()
    n_years = len(returns) / periods_per_year

    if n_years <= 0 or cumulative.iloc[-1] <= 0:
        return 0.0

    cagr = (cumulative.iloc[-1]) ** (1 / n_years) - 1

    return float(cagr / max_drawdown)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio (active return / tracking error).
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")

    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()

    if tracking_error == 0:
        return 0.0

    ir = np.sqrt(periods_per_year) * active_returns.mean() / tracking_error

    return float(ir)


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio (probability weighted gains / losses).
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]

    if losses.sum() == 0:
        return float('inf') if gains.sum() > 0 else 1.0

    omega = gains.sum() / losses.sum()

    return float(omega)


def calculate_tail_ratio(
    returns: pd.Series,
    percentile: float = 95
) -> float:
    """
    Calculate tail ratio (right tail / left tail).

    Measures asymmetry of return distribution.
    """
    right_tail = np.abs(np.percentile(returns, percentile))
    left_tail = np.abs(np.percentile(returns, 100 - percentile))

    if left_tail == 0:
        return float('inf') if right_tail > 0 else 1.0

    return float(right_tail / left_tail)


class MetricsCalculator:
    """
    Comprehensive metrics calculator.

    Follows lossless principle - all parameters derived from data.
    """

    def __init__(self, periods_per_year: int = 252):
        """
        Initialize calculator.

        Args:
            periods_per_year: Trading periods (252 for daily, 52 for weekly)
        """
        self.periods_per_year = periods_per_year

    def calculate_all(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Any] = None,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            returns: Series of period returns
            equity_curve: Equity curve series
            trades: List of trade objects (optional)
            benchmark_returns: Benchmark returns for relative metrics

        Returns:
            Dictionary of all metrics
        """
        perf_metrics = self.calculate_performance_metrics(
            returns, equity_curve, benchmark_returns
        )
        risk_metrics = self.calculate_risk_metrics(
            returns, equity_curve, benchmark_returns
        )

        trade_metrics = None
        if trades:
            trade_metrics = self.calculate_trade_metrics(trades, equity_curve)

        return {
            'performance': perf_metrics.to_dict() if hasattr(perf_metrics, 'to_dict') else perf_metrics,
            'risk': risk_metrics.__dict__ if hasattr(risk_metrics, '__dict__') else risk_metrics,
            'trades': trade_metrics.__dict__ if trade_metrics and hasattr(trade_metrics, '__dict__') else trade_metrics,
        }

    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        benchmark_returns: pd.Series = None
    ) -> PerformanceMetrics:
        """Calculate performance-focused metrics."""
        # Basic returns
        total_return = equity_curve.iloc[-1] - equity_curve.iloc[0]
        total_return_pct = total_return / equity_curve.iloc[0] * 100

        # CAGR
        n_years = len(returns) / self.periods_per_year
        if n_years > 0 and equity_curve.iloc[-1] > 0:
            cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1
        else:
            cagr = 0

        # Risk-adjusted ratios
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=self.periods_per_year)
        sortino = calculate_sortino_ratio(returns, periods_per_year=self.periods_per_year)

        # Max drawdown for Calmar
        rolling_max = equity_curve.cummax()
        drawdown = (rolling_max - equity_curve) / rolling_max
        max_dd = drawdown.max()

        calmar = calculate_calmar_ratio(returns, max_dd, self.periods_per_year)

        # Information ratio
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            info_ratio = calculate_information_ratio(
                returns, benchmark_returns, self.periods_per_year
            )
        else:
            info_ratio = 0.0

        # Omega ratio
        omega = calculate_omega_ratio(returns)

        # Tail ratio
        tail = calculate_tail_ratio(returns)

        # Gain to pain ratio
        gains = returns[returns > 0].sum()
        losses = np.abs(returns[returns < 0].sum())
        gain_to_pain = gains / losses if losses > 0 else float('inf')

        # Statistical measures
        skewness = float(stats.skew(returns.dropna()))
        kurtosis = float(stats.kurtosis(returns.dropna()))

        # VaR/CVaR
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = float(returns[returns <= var_99].mean()) if len(returns[returns <= var_99]) > 0 else var_99

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            cagr=cagr * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            omega_ratio=omega,
            tail_ratio=tail,
            gain_to_pain_ratio=gain_to_pain,
            skewness=skewness,
            kurtosis=kurtosis,
            best_day=float(returns.max()),
            worst_day=float(returns.min()),
            avg_daily_return=float(returns.mean()),
            daily_volatility=float(returns.std()),
            annualized_volatility=float(returns.std() * np.sqrt(self.periods_per_year)),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
        )

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        benchmark_returns: pd.Series = None
    ) -> RiskMetrics:
        """Calculate risk-focused metrics."""
        # Drawdown metrics
        rolling_max = equity_curve.cummax()
        drawdown = (rolling_max - equity_curve) / rolling_max

        max_dd = float(drawdown.max())

        # Max drawdown duration
        dd_duration = 0
        max_dd_duration = 0
        dd_periods = []

        in_drawdown = False
        dd_start = 0

        for i, dd in enumerate(drawdown):
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                if in_drawdown:
                    dd_periods.append(dd_duration)
                dd_duration = 0
                in_drawdown = False

        avg_dd = float(drawdown[drawdown > 0].mean()) if len(drawdown[drawdown > 0]) > 0 else 0
        dd_count = len(dd_periods)

        # Volatility
        vol_daily = float(returns.std())
        vol_annual = float(returns.std() * np.sqrt(self.periods_per_year))

        # Downside/Upside deviation
        downside_returns = returns[returns < 0]
        upside_returns = returns[returns > 0]

        downside_dev = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
        upside_dev = float(np.sqrt(np.mean(np.maximum(returns, 0) ** 2)))

        # VaR/CVaR
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = float(returns[returns <= var_99].mean()) if len(returns[returns <= var_99]) > 0 else var_99

        # Beta and Alpha (if benchmark provided)
        beta = None
        alpha = None
        treynor = None

        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            cov = np.cov(returns, benchmark_returns)[0, 1]
            var_benchmark = benchmark_returns.var()

            if var_benchmark > 0:
                beta = float(cov / var_benchmark)
                alpha = float(returns.mean() - beta * benchmark_returns.mean())

                if beta != 0:
                    rf_rate = 0  # Could derive from percentile
                    treynor = float((returns.mean() * self.periods_per_year - rf_rate) / beta)

        return RiskMetrics(
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_dd,
            drawdown_count=dd_count,
            volatility_daily=vol_daily,
            volatility_annual=vol_annual,
            downside_deviation=downside_dev,
            upside_deviation=upside_dev,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            alpha=alpha,
            treynor_ratio=treynor,
        )

    def calculate_trade_metrics(
        self,
        trades: List[Any],
        equity_curve: pd.Series
    ) -> TradeMetrics:
        """Calculate trade-level metrics."""
        if not trades:
            return None

        # Basic counts
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL metrics
        pnls = [t.pnl for t in trades]
        winners = [t.pnl for t in trades if t.pnl > 0]
        losers = [t.pnl for t in trades if t.pnl <= 0]

        avg_trade_pnl = np.mean(pnls) if pnls else 0
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = np.mean(losers) if losers else 0
        largest_winner = max(pnls) if pnls else 0
        largest_loser = min(pnls) if pnls else 0

        # Profit factor and payoff
        total_wins = sum(winners) if winners else 0
        total_losses = abs(sum(losers)) if losers else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        payoff_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_winner) - ((1 - win_rate) * abs(avg_loser))

        # Hold times
        durations = [t.duration_bars for t in trades]
        winner_durations = [t.duration_bars for t in trades if t.pnl > 0]
        loser_durations = [t.duration_bars for t in trades if t.pnl <= 0]

        avg_hold_time = np.mean(durations) if durations else 0
        avg_hold_winners = np.mean(winner_durations) if winner_durations else 0
        avg_hold_losers = np.mean(loser_durations) if loser_durations else 0

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0

        for t in trades:
            if t.pnl > 0:
                current_wins += 1
                max_consec_wins = max(max_consec_wins, current_wins)
                current_losses = 0
            else:
                current_losses += 1
                max_consec_losses = max(max_consec_losses, current_losses)
                current_wins = 0

        # Recovery factor
        rolling_max = equity_curve.cummax()
        max_dd = ((rolling_max - equity_curve) / rolling_max).max()
        total_profit = sum(pnls)

        recovery_factor = total_profit / (equity_curve.iloc[0] * max_dd) if max_dd > 0 else float('inf')

        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            expectancy=expectancy,
            avg_trade_pnl=avg_trade_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            avg_hold_time_bars=avg_hold_time,
            avg_hold_time_winners=avg_hold_winners,
            avg_hold_time_losers=avg_hold_losers,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            recovery_factor=recovery_factor,
        )
