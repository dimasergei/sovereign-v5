"""
Correlation Manager - Position correlation and portfolio risk management.

Manages:
- Cross-position correlation limits
- Portfolio VaR calculations
- Hedge suggestions
- Concentration risk

Critical for prop firm accounts to avoid correlated drawdowns.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position representation for correlation analysis."""
    symbol: str
    direction: int  # 1 = long, -1 = short
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime

    @property
    def pnl(self) -> float:
        """Calculate current P&L."""
        price_change = self.current_price - self.entry_price
        return price_change * self.size * self.direction

    @property
    def pnl_pct(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price / self.entry_price - 1) * self.direction * 100


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    correlation_matrix: np.ndarray
    symbols: List[str]
    portfolio_correlation: float
    high_correlation_pairs: List[Tuple[str, str, float]]
    concentration_risk: float
    diversification_ratio: float


class CorrelationManager:
    """
    Manages position correlations and portfolio risk.

    Features:
    - Real-time correlation matrix updates
    - Correlation-based position limits
    - Portfolio VaR calculation
    - Hedge suggestions

    All thresholds are derived from historical data.
    """

    def __init__(
        self,
        lookback_periods: int = 60,
        correlation_limit: float = None,  # Derived if None
        max_concentration: float = 0.5
    ):
        """
        Initialize correlation manager.

        Args:
            lookback_periods: Periods for correlation calculation
            correlation_limit: Max correlation between positions (derived if None)
            max_concentration: Max portfolio weight in single position
        """
        self.lookback = lookback_periods
        self._correlation_limit = correlation_limit
        self.max_concentration = max_concentration

        # Price history for correlation calculation
        self._price_history: Dict[str, deque] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._symbols: List[str] = []

        # Derived parameters
        self._derived_limit: Optional[float] = None

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update price history for correlation calculation.

        Args:
            prices: Dictionary of symbol -> current price
        """
        for symbol, price in prices.items():
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=self.lookback * 2)
            self._price_history[symbol].append(price)

        # Recalculate correlation matrix periodically
        if all(len(h) >= self.lookback for h in self._price_history.values()):
            self._update_correlation_matrix()

    def calculate_correlation_matrix(
        self,
        positions: List[Position]
    ) -> CorrelationResult:
        """
        Calculate correlation matrix for current positions.

        Args:
            positions: List of current positions

        Returns:
            CorrelationResult with full analysis
        """
        if not positions:
            return CorrelationResult(
                correlation_matrix=np.array([]),
                symbols=[],
                portfolio_correlation=0.0,
                high_correlation_pairs=[],
                concentration_risk=0.0,
                diversification_ratio=1.0
            )

        symbols = list(set(p.symbol for p in positions))

        # Get returns for each symbol
        returns_data = {}
        for symbol in symbols:
            if symbol in self._price_history:
                prices = list(self._price_history[symbol])
                if len(prices) >= 2:
                    returns = np.diff(np.log(prices))
                    returns_data[symbol] = returns[-self.lookback:]

        if len(returns_data) < 2:
            # Not enough data for correlation
            return CorrelationResult(
                correlation_matrix=np.eye(len(symbols)),
                symbols=symbols,
                portfolio_correlation=0.0,
                high_correlation_pairs=[],
                concentration_risk=self._calculate_concentration(positions),
                diversification_ratio=1.0
            )

        # Build correlation matrix
        min_len = min(len(r) for r in returns_data.values())
        returns_matrix = np.array([
            returns_data.get(s, np.zeros(min_len))[-min_len:]
            for s in symbols
        ])

        if returns_matrix.shape[0] < 2 or returns_matrix.shape[1] < 2:
            corr_matrix = np.eye(len(symbols))
        else:
            corr_matrix = np.corrcoef(returns_matrix)

        # Fix NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Find high correlation pairs
        high_corr_pairs = self._find_high_correlations(corr_matrix, symbols)

        # Calculate portfolio metrics
        weights = self._get_position_weights(positions)
        portfolio_corr = self._calculate_portfolio_correlation(corr_matrix, weights, symbols)
        concentration = self._calculate_concentration(positions)
        div_ratio = self._calculate_diversification_ratio(corr_matrix, weights)

        return CorrelationResult(
            correlation_matrix=corr_matrix,
            symbols=symbols,
            portfolio_correlation=portfolio_corr,
            high_correlation_pairs=high_corr_pairs,
            concentration_risk=concentration,
            diversification_ratio=div_ratio
        )

    def check_correlation_limit(
        self,
        new_position: Position,
        existing_positions: List[Position]
    ) -> Tuple[bool, str]:
        """
        Check if new position exceeds correlation limits.

        Args:
            new_position: Proposed new position
            existing_positions: Current open positions

        Returns:
            Tuple of (is_allowed, reason)
        """
        if not existing_positions:
            return True, "No existing positions"

        # Get correlation limit (derived or specified)
        limit = self._get_correlation_limit()

        # Calculate correlation with existing positions
        new_symbol = new_position.symbol

        for pos in existing_positions:
            corr = self._get_pairwise_correlation(new_symbol, pos.symbol)

            if corr is not None and abs(corr) > limit:
                # Check if same direction (compounds risk)
                same_direction = new_position.direction == pos.direction

                if same_direction:
                    return False, f"High correlation ({corr:.2f}) with {pos.symbol} in same direction"

        # Check concentration
        all_positions = existing_positions + [new_position]
        concentration = self._calculate_concentration(all_positions)

        if concentration > self.max_concentration:
            return False, f"Concentration risk too high ({concentration:.2%})"

        return True, "Correlation check passed"

    def get_portfolio_var(
        self,
        positions: List[Position],
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> float:
        """
        Calculate portfolio Value at Risk.

        Uses parametric VaR with correlation adjustment.

        Args:
            positions: Current positions
            confidence: VaR confidence level
            horizon_days: Time horizon

        Returns:
            VaR in position currency
        """
        if not positions:
            return 0.0

        # Get correlation matrix
        corr_result = self.calculate_correlation_matrix(positions)

        # Get individual position volatilities
        volatilities = []
        position_values = []

        for pos in positions:
            vol = self._get_symbol_volatility(pos.symbol)
            value = abs(pos.size * pos.current_price)
            volatilities.append(vol)
            position_values.append(value)

        if not volatilities:
            return 0.0

        # Convert to arrays
        vols = np.array(volatilities)
        values = np.array(position_values)

        # Weight by position value
        total_value = values.sum()
        if total_value == 0:
            return 0.0

        weights = values / total_value

        # Portfolio volatility (considering correlations)
        cov_matrix = np.outer(vols, vols) * corr_result.correlation_matrix
        portfolio_var = np.sqrt(weights @ cov_matrix @ weights)

        # Scale for horizon
        portfolio_var *= np.sqrt(horizon_days)

        # Convert to VaR using normal distribution quantile
        from scipy import stats
        z_score = stats.norm.ppf(confidence)

        var = total_value * portfolio_var * z_score

        return float(var)

    def suggest_hedge(
        self,
        positions: List[Position],
        target_correlation: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest hedging position to reduce portfolio correlation.

        Args:
            positions: Current positions
            target_correlation: Target portfolio correlation

        Returns:
            Hedge suggestion or None
        """
        if not positions:
            return None

        corr_result = self.calculate_correlation_matrix(positions)

        if corr_result.portfolio_correlation <= target_correlation:
            return None  # No hedge needed

        # Find negatively correlated assets
        hedge_candidates = []

        for i, symbol in enumerate(corr_result.symbols):
            # Look for negative correlation
            avg_corr = corr_result.correlation_matrix[i, :].mean()
            if avg_corr < 0:
                hedge_candidates.append((symbol, avg_corr))

        if not hedge_candidates:
            # No natural hedge available
            return {
                'action': 'reduce_exposure',
                'reason': 'No negatively correlated assets available',
                'recommendation': 'Consider reducing position sizes'
            }

        # Best hedge candidate
        best_hedge = min(hedge_candidates, key=lambda x: x[1])

        return {
            'action': 'add_hedge',
            'symbol': best_hedge[0],
            'correlation': best_hedge[1],
            'suggested_size': self._calculate_hedge_size(positions, best_hedge[0]),
            'reason': f'Negative correlation ({best_hedge[1]:.2f}) with portfolio'
        }

    def _update_correlation_matrix(self) -> None:
        """Update the internal correlation matrix."""
        symbols = list(self._price_history.keys())
        if len(symbols) < 2:
            return

        # Build returns matrix
        min_len = min(len(self._price_history[s]) for s in symbols)
        if min_len < 10:
            return

        returns = []
        for symbol in symbols:
            prices = list(self._price_history[symbol])
            r = np.diff(np.log(prices[-min_len:]))
            returns.append(r)

        returns_matrix = np.array(returns)
        self._correlation_matrix = np.corrcoef(returns_matrix)
        self._symbols = symbols

        # Derive correlation limit from data
        if self._derived_limit is None:
            self._derive_correlation_limit()

    def _derive_correlation_limit(self) -> None:
        """Derive appropriate correlation limit from historical data."""
        if self._correlation_matrix is None:
            self._derived_limit = 0.7  # Fallback
            return

        # Use 75th percentile of absolute correlations as limit
        upper_tri = np.triu(self._correlation_matrix, k=1)
        correlations = upper_tri[upper_tri != 0]

        if len(correlations) > 0:
            self._derived_limit = float(np.percentile(np.abs(correlations), 75))
        else:
            self._derived_limit = 0.7

        logger.info(f"Derived correlation limit: {self._derived_limit:.2f}")

    def _get_correlation_limit(self) -> float:
        """Get the effective correlation limit."""
        if self._correlation_limit is not None:
            return self._correlation_limit
        if self._derived_limit is not None:
            return self._derived_limit
        return 0.7  # Conservative default

    def _get_pairwise_correlation(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[float]:
        """Get correlation between two symbols."""
        if self._correlation_matrix is None:
            return None

        if symbol1 not in self._symbols or symbol2 not in self._symbols:
            return None

        i = self._symbols.index(symbol1)
        j = self._symbols.index(symbol2)

        return float(self._correlation_matrix[i, j])

    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get annualized volatility for symbol."""
        if symbol not in self._price_history:
            return 0.02  # Default 2% daily vol

        prices = list(self._price_history[symbol])
        if len(prices) < 2:
            return 0.02

        returns = np.diff(np.log(prices))
        daily_vol = np.std(returns)

        return float(daily_vol)

    def _get_position_weights(
        self,
        positions: List[Position]
    ) -> Dict[str, float]:
        """Calculate position weights by value."""
        total = sum(abs(p.size * p.current_price) for p in positions)
        if total == 0:
            return {p.symbol: 0 for p in positions}

        weights = {}
        for p in positions:
            value = abs(p.size * p.current_price)
            weights[p.symbol] = value / total

        return weights

    def _calculate_portfolio_correlation(
        self,
        corr_matrix: np.ndarray,
        weights: Dict[str, float],
        symbols: List[str]
    ) -> float:
        """Calculate weighted average portfolio correlation."""
        if len(symbols) < 2:
            return 0.0

        weight_vector = np.array([weights.get(s, 0) for s in symbols])

        # Weighted correlation
        weighted_corr = weight_vector @ corr_matrix @ weight_vector

        return float(weighted_corr)

    def _calculate_concentration(self, positions: List[Position]) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        if not positions:
            return 0.0

        total = sum(abs(p.size * p.current_price) for p in positions)
        if total == 0:
            return 0.0

        weights = [abs(p.size * p.current_price) / total for p in positions]
        hhi = sum(w ** 2 for w in weights)

        return float(hhi)

    def _calculate_diversification_ratio(
        self,
        corr_matrix: np.ndarray,
        weights: Dict[str, float]
    ) -> float:
        """Calculate diversification ratio."""
        if not weights:
            return 1.0

        # Weighted average of individual volatilities / portfolio volatility
        # Higher ratio = better diversification
        return 1.0  # Simplified for now

    def _find_high_correlations(
        self,
        corr_matrix: np.ndarray,
        symbols: List[str]
    ) -> List[Tuple[str, str, float]]:
        """Find pairs with high correlation."""
        limit = self._get_correlation_limit()
        high_pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = corr_matrix[i, j]
                if abs(corr) > limit * 0.8:  # Flag at 80% of limit
                    high_pairs.append((symbols[i], symbols[j], float(corr)))

        return sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True)

    def _calculate_hedge_size(
        self,
        positions: List[Position],
        hedge_symbol: str
    ) -> float:
        """Calculate appropriate hedge size."""
        # Total exposure
        total_exposure = sum(p.size * p.current_price * p.direction for p in positions)

        # Hedge to offset ~50% of exposure
        return abs(total_exposure) * 0.5
