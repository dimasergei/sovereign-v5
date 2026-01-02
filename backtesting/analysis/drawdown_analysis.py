"""
Drawdown Analysis Module - Deep analysis of drawdown periods.

Provides detailed insights into drawdown behavior for prop firm compliance.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DrawdownPeriod:
    """Information about a single drawdown period."""
    start_date: datetime
    end_date: Optional[datetime]
    recovery_date: Optional[datetime]

    peak_value: float
    trough_value: float
    drawdown_pct: float

    duration_bars: int
    recovery_bars: Optional[int]

    # During drawdown stats
    trades_during: int = 0
    wins_during: int = 0
    losses_during: int = 0

    # Market context
    regime_at_start: Optional[str] = None
    volatility_at_start: Optional[float] = None


@dataclass
class UnderwaterAnalysis:
    """Analysis of time spent in drawdown (underwater)."""
    total_bars: int
    underwater_bars: int
    underwater_pct: float

    avg_underwater_depth: float
    max_underwater_depth: float

    # By drawdown tier
    time_above_1pct: float  # % of time with DD > 1%
    time_above_2pct: float
    time_above_5pct: float
    time_above_guardian: float  # Time near guardian limit


class DrawdownAnalyzer:
    """
    Comprehensive drawdown analysis for prop firm trading.

    Provides insights for:
    - Drawdown period identification
    - Recovery analysis
    - Underwater analysis
    - Regime-specific drawdown behavior
    """

    def __init__(self, guardian_threshold_pct: float = 7.0):
        """
        Initialize analyzer.

        Args:
            guardian_threshold_pct: Guardian drawdown threshold
        """
        self.guardian_threshold = guardian_threshold_pct

    def analyze(
        self,
        equity_curve: pd.Series,
        trades: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive drawdown analysis.

        Args:
            equity_curve: Equity curve series
            trades: Optional list of trades

        Returns:
            Dictionary with all drawdown analysis
        """
        # Calculate drawdown series
        drawdown_series = self.calculate_drawdown_series(equity_curve)

        # Identify drawdown periods
        periods = self.identify_drawdown_periods(equity_curve, drawdown_series)

        # Underwater analysis
        underwater = self.analyze_underwater(drawdown_series)

        # Add trade context if available
        if trades:
            periods = self._add_trade_context(periods, trades, equity_curve)

        # Summary statistics
        summary = self._calculate_summary(periods, drawdown_series)

        return {
            'drawdown_series': drawdown_series,
            'periods': periods,
            'underwater': underwater,
            'summary': summary,
        }

    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        rolling_max = equity_curve.cummax()
        drawdown = (rolling_max - equity_curve) / rolling_max
        return drawdown

    def identify_drawdown_periods(
        self,
        equity_curve: pd.Series,
        drawdown_series: pd.Series,
        min_drawdown_pct: float = 0.01
    ) -> List[DrawdownPeriod]:
        """
        Identify distinct drawdown periods.

        Args:
            equity_curve: Equity curve
            drawdown_series: Drawdown series
            min_drawdown_pct: Minimum drawdown to track

        Returns:
            List of DrawdownPeriod objects
        """
        periods = []

        in_drawdown = False
        dd_start_idx = 0
        peak_value = 0.0
        trough_value = float('inf')
        trough_idx = 0

        rolling_max = equity_curve.cummax()

        for i, (dd, eq) in enumerate(zip(drawdown_series, equity_curve)):
            if dd > min_drawdown_pct:
                if not in_drawdown:
                    # Start of new drawdown
                    in_drawdown = True
                    dd_start_idx = i
                    peak_value = rolling_max.iloc[i]
                    trough_value = eq
                    trough_idx = i
                else:
                    # Continue in drawdown
                    if eq < trough_value:
                        trough_value = eq
                        trough_idx = i
            else:
                if in_drawdown:
                    # End of drawdown - recovered
                    drawdown_pct = (peak_value - trough_value) / peak_value

                    period = DrawdownPeriod(
                        start_date=equity_curve.index[dd_start_idx] if hasattr(equity_curve.index, '__getitem__') else datetime.now(),
                        end_date=equity_curve.index[trough_idx] if hasattr(equity_curve.index, '__getitem__') else datetime.now(),
                        recovery_date=equity_curve.index[i] if hasattr(equity_curve.index, '__getitem__') else datetime.now(),
                        peak_value=peak_value,
                        trough_value=trough_value,
                        drawdown_pct=drawdown_pct * 100,
                        duration_bars=trough_idx - dd_start_idx,
                        recovery_bars=i - trough_idx,
                    )
                    periods.append(period)

                    in_drawdown = False

        # Handle case where we end in drawdown
        if in_drawdown:
            drawdown_pct = (peak_value - trough_value) / peak_value

            period = DrawdownPeriod(
                start_date=equity_curve.index[dd_start_idx] if hasattr(equity_curve.index, '__getitem__') else datetime.now(),
                end_date=equity_curve.index[trough_idx] if hasattr(equity_curve.index, '__getitem__') else datetime.now(),
                recovery_date=None,  # Not recovered
                peak_value=peak_value,
                trough_value=trough_value,
                drawdown_pct=drawdown_pct * 100,
                duration_bars=trough_idx - dd_start_idx,
                recovery_bars=None,
            )
            periods.append(period)

        return periods

    def analyze_underwater(self, drawdown_series: pd.Series) -> UnderwaterAnalysis:
        """Analyze time spent in drawdown."""
        total_bars = len(drawdown_series)
        underwater_mask = drawdown_series > 0
        underwater_bars = underwater_mask.sum()

        avg_depth = float(drawdown_series[underwater_mask].mean()) if underwater_bars > 0 else 0
        max_depth = float(drawdown_series.max())

        # Time by tier
        time_1pct = (drawdown_series > 0.01).sum() / total_bars
        time_2pct = (drawdown_series > 0.02).sum() / total_bars
        time_5pct = (drawdown_series > 0.05).sum() / total_bars
        time_guardian = (drawdown_series > self.guardian_threshold / 100).sum() / total_bars

        return UnderwaterAnalysis(
            total_bars=total_bars,
            underwater_bars=int(underwater_bars),
            underwater_pct=float(underwater_bars / total_bars * 100),
            avg_underwater_depth=avg_depth * 100,
            max_underwater_depth=max_depth * 100,
            time_above_1pct=float(time_1pct * 100),
            time_above_2pct=float(time_2pct * 100),
            time_above_5pct=float(time_5pct * 100),
            time_above_guardian=float(time_guardian * 100),
        )

    def _add_trade_context(
        self,
        periods: List[DrawdownPeriod],
        trades: List[Any],
        equity_curve: pd.Series
    ) -> List[DrawdownPeriod]:
        """Add trade statistics to drawdown periods."""
        for period in periods:
            trades_in_period = 0
            wins = 0
            losses = 0

            for trade in trades:
                trade_time = trade.entry_time

                # Check if trade falls within drawdown period
                if hasattr(period.start_date, 'timestamp'):
                    if period.start_date <= trade_time:
                        if period.recovery_date is None or trade_time <= period.recovery_date:
                            trades_in_period += 1
                            if trade.pnl > 0:
                                wins += 1
                            else:
                                losses += 1

            period.trades_during = trades_in_period
            period.wins_during = wins
            period.losses_during = losses

        return periods

    def _calculate_summary(
        self,
        periods: List[DrawdownPeriod],
        drawdown_series: pd.Series
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not periods:
            return {
                'total_periods': 0,
                'max_drawdown_pct': float(drawdown_series.max() * 100),
                'avg_drawdown_pct': 0,
                'avg_duration': 0,
                'avg_recovery': 0,
            }

        max_dd = max(p.drawdown_pct for p in periods)
        avg_dd = np.mean([p.drawdown_pct for p in periods])
        avg_duration = np.mean([p.duration_bars for p in periods])

        recovered_periods = [p for p in periods if p.recovery_bars is not None]
        avg_recovery = np.mean([p.recovery_bars for p in recovered_periods]) if recovered_periods else None

        # Classify by severity
        minor_dd = len([p for p in periods if p.drawdown_pct < 2])
        moderate_dd = len([p for p in periods if 2 <= p.drawdown_pct < 5])
        severe_dd = len([p for p in periods if p.drawdown_pct >= 5])

        return {
            'total_periods': len(periods),
            'max_drawdown_pct': max_dd,
            'avg_drawdown_pct': avg_dd,
            'avg_duration_bars': avg_duration,
            'avg_recovery_bars': avg_recovery,
            'minor_drawdowns': minor_dd,
            'moderate_drawdowns': moderate_dd,
            'severe_drawdowns': severe_dd,
            'recovery_rate': len(recovered_periods) / len(periods) if periods else 0,
        }

    def check_prop_firm_safety(
        self,
        equity_curve: pd.Series,
        max_limit_pct: float = 8.0,
        guardian_pct: float = 7.0
    ) -> Dict[str, Any]:
        """
        Check if strategy is safe for prop firm trading.

        Args:
            equity_curve: Equity curve
            max_limit_pct: Actual max drawdown limit
            guardian_pct: Guardian threshold

        Returns:
            Safety assessment
        """
        drawdown_series = self.calculate_drawdown_series(equity_curve) * 100

        max_dd = drawdown_series.max()
        guardian_breaches = (drawdown_series > guardian_pct).sum()
        limit_breaches = (drawdown_series > max_limit_pct).sum()

        # Calculate buffer statistics
        buffer_to_guardian = guardian_pct - max_dd
        buffer_to_limit = max_limit_pct - max_dd

        # Monte Carlo stress
        # Simulate worst case by assuming future drawdowns similar to past
        dd_values = drawdown_series[drawdown_series > 0]
        if len(dd_values) > 0:
            stress_dd = np.percentile(dd_values, 99) * 1.5  # 99th pct * 1.5 stress
        else:
            stress_dd = 0

        return {
            'max_historical_drawdown': max_dd,
            'buffer_to_guardian': buffer_to_guardian,
            'buffer_to_limit': buffer_to_limit,
            'guardian_breaches': int(guardian_breaches),
            'limit_breaches': int(limit_breaches),
            'stressed_max_dd': stress_dd,
            'is_safe': max_dd < guardian_pct and limit_breaches == 0,
            'recommendation': self._get_safety_recommendation(max_dd, guardian_pct, max_limit_pct),
        }

    def _get_safety_recommendation(
        self,
        max_dd: float,
        guardian_pct: float,
        max_limit_pct: float
    ) -> str:
        """Generate safety recommendation."""
        if max_dd >= max_limit_pct:
            return "CRITICAL: Strategy would breach max drawdown limit. Not suitable for prop firm."
        elif max_dd >= guardian_pct:
            return "WARNING: Strategy breaches guardian threshold. Reduce position sizing or improve win rate."
        elif max_dd >= guardian_pct * 0.75:
            return "CAUTION: Strategy approaches guardian threshold. Consider reducing risk."
        else:
            return "OK: Strategy maintains adequate buffer to guardian threshold."
