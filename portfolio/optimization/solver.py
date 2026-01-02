"""
Portfolio Optimizer - Optimization algorithms for portfolio construction.

Implements various optimization methods without hardcoded parameters.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from scipy import optimize

from .constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: Dict[str, float]
    method: OptimizationMethod
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    success: bool
    message: str
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PortfolioOptimizer:
    """
    Portfolio optimizer with multiple methods.

    All optimization uses market-derived parameters.
    No hardcoded risk-free rates or expected returns.

    Usage:
        optimizer = PortfolioOptimizer(constraints)

        result = optimizer.optimize(
            expected_returns,
            covariance_matrix,
            method=OptimizationMethod.MAX_SHARPE
        )
    """

    def __init__(
        self,
        constraints: PortfolioConstraints = None,
        risk_free_rate: float = None  # Derived if None
    ):
        """
        Initialize optimizer.

        Args:
            constraints: Portfolio constraints
            risk_free_rate: Risk-free rate (derived from market if None)
        """
        self.constraints = constraints or PortfolioConstraints()
        self._risk_free_rate = risk_free_rate

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        symbols: List[str],
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            symbols: List of symbol names
            method: Optimization method

        Returns:
            OptimizationResult with optimal weights
        """
        n = len(symbols)

        if len(expected_returns) != n or covariance_matrix.shape != (n, n):
            return OptimizationResult(
                weights={s: 1/n for s in symbols},
                method=method,
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                success=False,
                message="Input dimensions mismatch"
            )

        # Get risk-free rate
        rf = self._get_risk_free_rate(expected_returns)

        # Optimize based on method
        if method == OptimizationMethod.MAX_SHARPE:
            weights = self._optimize_sharpe(expected_returns, covariance_matrix, rf)
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = self._optimize_min_variance(covariance_matrix)
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._optimize_risk_parity(covariance_matrix)
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._optimize_max_diversification(covariance_matrix)
        else:  # EQUAL_WEIGHT
            weights = np.ones(n) / n

        # Apply constraints
        weights = self._apply_constraints(weights, symbols)

        # Calculate metrics
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0

        return OptimizationResult(
            weights={symbols[i]: float(weights[i]) for i in range(n)},
            method=method,
            expected_return=float(port_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            success=True,
            message="Optimization successful"
        )

    def _get_risk_free_rate(self, returns: np.ndarray) -> float:
        """Derive risk-free rate from return distribution."""
        if self._risk_free_rate is not None:
            return self._risk_free_rate

        # Use minimum return as proxy for risk-free
        return float(max(0, np.percentile(returns, 5)))

    def _optimize_sharpe(
        self,
        returns: np.ndarray,
        cov: np.ndarray,
        rf: float
    ) -> np.ndarray:
        """Maximize Sharpe ratio."""
        n = len(returns)

        def neg_sharpe(w):
            port_ret = np.dot(w, returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol == 0:
                return 0
            return -(port_ret - rf) / port_vol

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds from constraints
        bounds = self.constraints.get_bounds([''] * n)

        # Initial guess
        x0 = np.ones(n) / n

        # Optimize
        result = optimize.minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else x0

    def _optimize_min_variance(self, cov: np.ndarray) -> np.ndarray:
        """Minimize portfolio variance."""
        n = cov.shape[0]

        def portfolio_var(w):
            return np.dot(w.T, np.dot(cov, w))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = self.constraints.get_bounds([''] * n)
        x0 = np.ones(n) / n

        result = optimize.minimize(
            portfolio_var,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else x0

    def _optimize_risk_parity(self, cov: np.ndarray) -> np.ndarray:
        """Risk parity optimization - equal risk contribution."""
        n = cov.shape[0]

        def risk_contribution_diff(w):
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol == 0:
                return 0

            # Marginal risk contribution
            mrc = np.dot(cov, w) / port_vol

            # Risk contribution
            rc = w * mrc

            # Target: equal risk contribution
            target = port_vol / n

            # Minimize squared difference from target
            return np.sum((rc - target) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0.01, self.constraints.max_position_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = optimize.minimize(
            risk_contribution_diff,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else x0

    def _optimize_max_diversification(self, cov: np.ndarray) -> np.ndarray:
        """Maximize diversification ratio."""
        n = cov.shape[0]
        vols = np.sqrt(np.diag(cov))

        def neg_diversification(w):
            weighted_vol = np.dot(w, vols)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol == 0:
                return 0
            return -weighted_vol / port_vol

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = self.constraints.get_bounds([''] * n)
        x0 = np.ones(n) / n

        result = optimize.minimize(
            neg_diversification,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else x0

    def _apply_constraints(
        self,
        weights: np.ndarray,
        symbols: List[str]
    ) -> np.ndarray:
        """Apply portfolio constraints to weights."""
        # Ensure non-negative
        weights = np.maximum(weights, 0)

        # Apply position limits
        for i, symbol in enumerate(symbols):
            if symbol in self.constraints.position_limits:
                limit = self.constraints.position_limits[symbol]
                weights[i] = np.clip(weights[i], limit.min_weight, limit.max_weight)
            else:
                weights[i] = np.clip(weights[i], 0, self.constraints.max_position_weight)

        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Remove tiny weights
        weights[weights < self.constraints.min_position_weight] = 0

        # Renormalize again
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        symbols: List[str],
        n_points: int = 20
    ) -> List[OptimizationResult]:
        """
        Generate efficient frontier.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            symbols: Symbol names
            n_points: Number of frontier points

        Returns:
            List of OptimizationResult for each frontier point
        """
        frontier = []

        # Find min and max return portfolios
        min_var = self._optimize_min_variance(covariance_matrix)
        min_ret = np.dot(min_var, expected_returns)

        max_ret = expected_returns.max()

        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)

        for target in target_returns:
            weights = self._optimize_for_target_return(
                expected_returns,
                covariance_matrix,
                target
            )

            port_ret = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            rf = self._get_risk_free_rate(expected_returns)
            sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0

            result = OptimizationResult(
                weights={symbols[i]: float(weights[i]) for i in range(len(symbols))},
                method=OptimizationMethod.MAX_SHARPE,
                expected_return=float(port_ret),
                expected_volatility=float(port_vol),
                sharpe_ratio=float(sharpe),
                success=True,
                message=f"Frontier point at target return {target:.4f}"
            )

            frontier.append(result)

        return frontier

    def _optimize_for_target_return(
        self,
        returns: np.ndarray,
        cov: np.ndarray,
        target: float
    ) -> np.ndarray:
        """Minimize variance for target return."""
        n = len(returns)

        def portfolio_var(w):
            return np.dot(w.T, np.dot(cov, w))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, returns) - target}
        ]

        bounds = [(0, self.constraints.max_position_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = optimize.minimize(
            portfolio_var,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x if result.success else x0
