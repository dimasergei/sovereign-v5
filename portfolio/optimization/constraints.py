"""
Portfolio Constraints Module - Position limits and constraints.

Defines constraints for portfolio optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionLimit:
    """Position size limit."""
    symbol: str
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_size: float = 0.0
    max_size: float = float('inf')


@dataclass
class CorrelationConstraint:
    """Correlation constraint between positions."""
    symbol1: str
    symbol2: str
    max_combined_weight: float = 0.5


@dataclass
class PortfolioConstraints:
    """
    Collection of portfolio constraints.

    Constraints are used in portfolio optimization
    to ensure prop firm compliance.
    """
    # Overall constraints
    max_positions: int = 5
    max_position_weight: float = 0.5
    min_position_weight: float = 0.01

    # Risk constraints
    max_portfolio_volatility: float = 0.05  # 5% daily
    max_correlation: float = 0.7
    max_drawdown_budget: float = 0.05  # 5% of remaining budget

    # Position-specific limits
    position_limits: Dict[str, PositionLimit] = field(default_factory=dict)

    # Correlation constraints
    correlation_constraints: List[CorrelationConstraint] = field(default_factory=list)

    def add_position_limit(
        self,
        symbol: str,
        min_weight: float = 0.0,
        max_weight: float = None
    ) -> None:
        """Add position-specific limit."""
        self.position_limits[symbol] = PositionLimit(
            symbol=symbol,
            min_weight=min_weight,
            max_weight=max_weight or self.max_position_weight
        )

    def add_correlation_constraint(
        self,
        symbol1: str,
        symbol2: str,
        max_combined_weight: float = 0.5
    ) -> None:
        """Add correlation constraint between two symbols."""
        self.correlation_constraints.append(
            CorrelationConstraint(
                symbol1=symbol1,
                symbol2=symbol2,
                max_combined_weight=max_combined_weight
            )
        )

    def validate_weights(
        self,
        weights: Dict[str, float]
    ) -> tuple:
        """
        Validate if weights satisfy constraints.

        Args:
            weights: Symbol to weight mapping

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []

        # Check number of positions
        active = sum(1 for w in weights.values() if w > 0)
        if active > self.max_positions:
            violations.append(f"Too many positions: {active} > {self.max_positions}")

        # Check individual weights
        for symbol, weight in weights.items():
            if weight > self.max_position_weight:
                violations.append(f"{symbol} weight {weight:.2%} > max {self.max_position_weight:.2%}")

            if weight > 0 and weight < self.min_position_weight:
                violations.append(f"{symbol} weight {weight:.2%} < min {self.min_position_weight:.2%}")

            # Check specific limits
            if symbol in self.position_limits:
                limit = self.position_limits[symbol]
                if weight > limit.max_weight:
                    violations.append(f"{symbol} exceeds specific limit: {weight:.2%} > {limit.max_weight:.2%}")

        # Check weight sum
        total = sum(weights.values())
        if total > 1.0:
            violations.append(f"Total weight {total:.2%} > 100%")

        return len(violations) == 0, violations

    def get_bounds(
        self,
        symbols: List[str]
    ) -> List[tuple]:
        """
        Get weight bounds for optimization.

        Args:
            symbols: List of symbols

        Returns:
            List of (min, max) bounds for each symbol
        """
        bounds = []

        for symbol in symbols:
            if symbol in self.position_limits:
                limit = self.position_limits[symbol]
                bounds.append((limit.min_weight, limit.max_weight))
            else:
                bounds.append((0.0, self.max_position_weight))

        return bounds
