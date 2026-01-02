"""
Regime Analysis Module - Performance breakdown by market regime.

Analyzes strategy performance across different market conditions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime."""
    regime_name: str
    total_bars: int
    pct_of_total: float

    # Returns
    total_return: float
    avg_return: float
    volatility: float
    sharpe_ratio: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float

    # Trade metrics (if trades provided)
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class RegimeAnalyzer:
    """
    Analyze strategy performance across market regimes.

    Regimes are derived from market data using:
    - Volatility levels (low/normal/high)
    - Trend direction (bullish/bearish/sideways)
    - Market structure (trending/ranging)
    """

    def __init__(self):
        """Initialize regime analyzer."""
        pass

    def analyze(
        self,
        returns: pd.Series,
        market_data: pd.DataFrame,
        trades: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance by regime.

        Args:
            returns: Strategy returns
            market_data: Market OHLCV data
            trades: Optional trade list

        Returns:
            Dictionary with regime analysis
        """
        # Detect regimes
        volatility_regime = self._detect_volatility_regime(market_data)
        trend_regime = self._detect_trend_regime(market_data)
        structure_regime = self._detect_market_structure(market_data)

        # Analyze by each regime type
        vol_analysis = self._analyze_by_regime(returns, volatility_regime, trades)
        trend_analysis = self._analyze_by_regime(returns, trend_regime, trades)
        structure_analysis = self._analyze_by_regime(returns, structure_regime, trades)

        # Cross-regime analysis
        cross_regime = self._cross_regime_analysis(
            returns, volatility_regime, trend_regime
        )

        return {
            'volatility_regimes': vol_analysis,
            'trend_regimes': trend_analysis,
            'structure_regimes': structure_analysis,
            'cross_regime': cross_regime,
            'regime_series': {
                'volatility': volatility_regime,
                'trend': trend_regime,
                'structure': structure_regime,
            }
        }

    def _detect_volatility_regime(
        self,
        market_data: pd.DataFrame,
        lookback: int = None
    ) -> pd.Series:
        """
        Detect volatility regime from price data.

        Uses adaptive lookback based on data characteristics.
        """
        # Calculate returns
        returns = market_data['close'].pct_change()

        # Adaptive lookback - derived from data
        if lookback is None:
            # Use autocorrelation to determine appropriate lookback
            lookback = self._derive_volatility_lookback(returns)

        # Rolling volatility
        rolling_vol = returns.rolling(window=lookback).std()

        # Derive thresholds from distribution
        vol_percentiles = rolling_vol.quantile([0.25, 0.75])
        low_threshold = vol_percentiles.iloc[0]
        high_threshold = vol_percentiles.iloc[1]

        # Classify
        regime = pd.Series(index=market_data.index, dtype=str)
        regime[rolling_vol <= low_threshold] = 'low_volatility'
        regime[(rolling_vol > low_threshold) & (rolling_vol < high_threshold)] = 'normal_volatility'
        regime[rolling_vol >= high_threshold] = 'high_volatility'

        return regime.fillna('normal_volatility')

    def _detect_trend_regime(
        self,
        market_data: pd.DataFrame,
        lookback: int = None
    ) -> pd.Series:
        """
        Detect trend regime from price data.
        """
        close = market_data['close']

        # Adaptive lookback
        if lookback is None:
            lookback = self._derive_trend_lookback(close)

        # Calculate trend indicators
        sma = close.rolling(window=lookback).mean()
        returns = close.pct_change(periods=lookback)

        # Derive threshold from return distribution
        threshold = np.abs(returns).quantile(0.33)

        # Classify
        regime = pd.Series(index=market_data.index, dtype=str)
        regime[(close > sma) & (returns > threshold)] = 'bullish'
        regime[(close < sma) & (returns < -threshold)] = 'bearish'
        regime[~regime.isin(['bullish', 'bearish'])] = 'sideways'

        return regime.fillna('sideways')

    def _detect_market_structure(
        self,
        market_data: pd.DataFrame,
        lookback: int = None
    ) -> pd.Series:
        """
        Detect market structure (trending vs ranging).

        Uses ADX-like calculation without hardcoded thresholds.
        """
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']

        # Adaptive lookback
        if lookback is None:
            lookback = self._derive_structure_lookback(close)

        # Calculate directional movement
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # True range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=market_data.index)

        # Smooth
        atr = tr.rolling(window=lookback).mean()
        plus_di = pd.Series(plus_dm, index=market_data.index).rolling(window=lookback).mean() / atr * 100
        minus_di = pd.Series(minus_dm, index=market_data.index).rolling(window=lookback).mean() / atr * 100

        # ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(window=lookback).mean()

        # Derive threshold from ADX distribution
        adx_threshold = adx.quantile(0.5)  # Median as threshold

        # Classify
        regime = pd.Series(index=market_data.index, dtype=str)
        regime[adx >= adx_threshold] = 'trending'
        regime[adx < adx_threshold] = 'ranging'

        return regime.fillna('ranging')

    def _derive_volatility_lookback(self, returns: pd.Series) -> int:
        """Derive appropriate volatility lookback from data."""
        # Use half-life of volatility autocorrelation
        vol = returns.rolling(window=20).std()

        try:
            autocorr = [vol.autocorr(lag=i) for i in range(1, 51)]
            autocorr = [a if not np.isnan(a) else 0 for a in autocorr]

            # Find half-life
            half_life = next((i for i, a in enumerate(autocorr) if a < 0.5), 20)
            return max(10, min(50, half_life * 2))
        except:
            return 20

    def _derive_trend_lookback(self, close: pd.Series) -> int:
        """Derive appropriate trend lookback from data."""
        # Use dominant cycle from simple analysis
        returns = close.pct_change()

        try:
            # Simple autocorrelation analysis
            autocorr = [returns.autocorr(lag=i) for i in range(10, 100)]
            autocorr = [a if not np.isnan(a) else 0 for a in autocorr]

            # Find first negative autocorrelation (mean reversion point)
            trend_period = next((i + 10 for i, a in enumerate(autocorr) if a < 0), 20)
            return max(10, min(50, trend_period))
        except:
            return 20

    def _derive_structure_lookback(self, close: pd.Series) -> int:
        """Derive appropriate structure lookback."""
        # Similar to trend lookback
        return self._derive_trend_lookback(close)

    def _analyze_by_regime(
        self,
        returns: pd.Series,
        regime_series: pd.Series,
        trades: List[Any] = None
    ) -> Dict[str, RegimePerformance]:
        """Analyze performance for each regime."""
        results = {}
        total_bars = len(returns)

        for regime_name in regime_series.unique():
            if pd.isna(regime_name):
                continue

            mask = regime_series == regime_name
            regime_returns = returns[mask]

            if len(regime_returns) == 0:
                continue

            # Calculate metrics
            total_return = (1 + regime_returns).prod() - 1
            avg_return = regime_returns.mean()
            vol = regime_returns.std()
            sharpe = np.sqrt(252) * avg_return / vol if vol > 0 else 0

            # Drawdown
            equity = (1 + regime_returns).cumprod()
            rolling_max = equity.cummax()
            drawdown = (rolling_max - equity) / rolling_max
            max_dd = drawdown.max()
            avg_dd = drawdown[drawdown > 0].mean() if len(drawdown[drawdown > 0]) > 0 else 0

            # Trade metrics
            n_trades = 0
            win_rate = 0
            pf = 0

            if trades:
                regime_trades = []
                for t in trades:
                    if hasattr(t, 'entry_time') and t.entry_time in mask.index:
                        if mask.loc[t.entry_time]:
                            regime_trades.append(t)

                if regime_trades:
                    n_trades = len(regime_trades)
                    winners = [t for t in regime_trades if t.pnl > 0]
                    win_rate = len(winners) / n_trades

                    wins = sum(t.pnl for t in regime_trades if t.pnl > 0)
                    losses = abs(sum(t.pnl for t in regime_trades if t.pnl <= 0))
                    pf = wins / losses if losses > 0 else float('inf')

            results[regime_name] = RegimePerformance(
                regime_name=regime_name,
                total_bars=int(mask.sum()),
                pct_of_total=float(mask.sum() / total_bars * 100),
                total_return=float(total_return * 100),
                avg_return=float(avg_return * 100),
                volatility=float(vol * 100),
                sharpe_ratio=float(sharpe),
                max_drawdown=float(max_dd * 100),
                avg_drawdown=float(avg_dd * 100),
                total_trades=n_trades,
                win_rate=float(win_rate * 100),
                profit_factor=float(pf),
            )

        return results

    def _cross_regime_analysis(
        self,
        returns: pd.Series,
        regime1: pd.Series,
        regime2: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance across regime combinations."""
        results = {}

        for r1 in regime1.unique():
            if pd.isna(r1):
                continue
            for r2 in regime2.unique():
                if pd.isna(r2):
                    continue

                mask = (regime1 == r1) & (regime2 == r2)
                combined_returns = returns[mask]

                if len(combined_returns) < 10:
                    continue

                key = f"{r1}_{r2}"
                results[key] = {
                    'bars': int(mask.sum()),
                    'pct_of_total': float(mask.sum() / len(returns) * 100),
                    'avg_return': float(combined_returns.mean() * 100),
                    'volatility': float(combined_returns.std() * 100),
                    'total_return': float((1 + combined_returns).prod() - 1) * 100,
                }

        return results

    def get_regime_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate trading recommendations based on regime analysis."""
        recommendations = []

        # Analyze volatility regimes
        vol_regimes = analysis.get('volatility_regimes', {})

        if 'high_volatility' in vol_regimes:
            hv = vol_regimes['high_volatility']
            if hv.sharpe_ratio < 0:
                recommendations.append(
                    "Reduce position sizing during high volatility periods - negative Sharpe observed"
                )
            elif hv.sharpe_ratio < 0.5:
                recommendations.append(
                    "Consider reducing activity during high volatility - low risk-adjusted returns"
                )

        # Analyze trend regimes
        trend_regimes = analysis.get('trend_regimes', {})

        best_regime = None
        best_sharpe = -float('inf')

        for name, perf in trend_regimes.items():
            if perf.sharpe_ratio > best_sharpe:
                best_sharpe = perf.sharpe_ratio
                best_regime = name

        if best_regime:
            recommendations.append(
                f"Strategy performs best in {best_regime} regime (Sharpe: {best_sharpe:.2f})"
            )

        # Analyze structure
        struct_regimes = analysis.get('structure_regimes', {})

        if 'trending' in struct_regimes and 'ranging' in struct_regimes:
            trend_perf = struct_regimes['trending']
            range_perf = struct_regimes['ranging']

            if trend_perf.sharpe_ratio > range_perf.sharpe_ratio * 2:
                recommendations.append(
                    "Strategy is trend-following - reduce activity in ranging markets"
                )
            elif range_perf.sharpe_ratio > trend_perf.sharpe_ratio * 2:
                recommendations.append(
                    "Strategy is mean-reverting - reduce activity in trending markets"
                )

        return recommendations
