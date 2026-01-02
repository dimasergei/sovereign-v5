"""
Trade Manager - Institutional Exit Management.

Handles:
1. Initial stop placement
2. Breakeven stop movement
3. Trailing stop logic
4. Partial profit taking
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


logger = logging.getLogger(__name__)


class StopType(Enum):
    """Types of stop loss."""
    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"


@dataclass
class Position:
    """Active position state."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    initial_stop: float
    current_stop: float
    take_profit: float
    size: float
    initial_size: float
    stop_type: StopType = StopType.INITIAL
    partials_taken: int = 0
    entry_reason: str = ""
    r_multiple: float = 0.0
    bars_held: int = 0


@dataclass
class TradeResult:
    """Result of a closed trade."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    partials_taken: int


class TradeManager:
    """
    Institutional exit management.

    Strategy:
    1. Initial stop at defined level (based on ATR and swing)
    2. Move stop to breakeven after 1R profit
    3. Trail stop using ATR after 1.5R
    4. Take partial profits at 1.5R and 2.5R
    """

    def __init__(
        self,
        breakeven_trigger_r: float = 1.0,
        trail_activation_r: float = 1.5,
        trail_distance_atr: float = 2.0,
        partial_1_at_r: float = 1.5,
        partial_1_size: float = 0.33,
        partial_2_at_r: float = 2.5,
        partial_2_size: float = 0.33
    ):
        """
        Initialize trade manager.

        Args:
            breakeven_trigger_r: Move to BE after this R profit
            trail_activation_r: Start trailing after this R profit
            trail_distance_atr: Trail stop this many ATRs behind price
            partial_1_at_r: Take first partial at this R
            partial_1_size: Size of first partial (0.33 = 33%)
            partial_2_at_r: Take second partial at this R
            partial_2_size: Size of second partial
        """
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trail_activation_r = trail_activation_r
        self.trail_distance_atr = trail_distance_atr
        self.partial_1_at_r = partial_1_at_r
        self.partial_1_size = partial_1_size
        self.partial_2_at_r = partial_2_at_r
        self.partial_2_size = partial_2_size

    def update_position(
        self,
        position: Position,
        current_price: float,
        atr: float
    ) -> Position:
        """
        Update position with current price, adjust stops.

        Args:
            position: Current position state
            current_price: Current market price
            atr: Current ATR value

        Returns:
            Updated position with new stop levels
        """
        position.current_price = current_price
        position.bars_held += 1

        # Calculate current R-multiple
        initial_risk = abs(position.entry_price - position.initial_stop)
        if initial_risk == 0:
            return position

        if position.direction == 'long':
            current_profit = current_price - position.entry_price
        else:
            current_profit = position.entry_price - current_price

        position.r_multiple = current_profit / initial_risk

        # Update stop based on R-multiple
        position = self._update_stop_logic(position, current_price, atr, initial_risk)

        return position

    def _update_stop_logic(
        self,
        position: Position,
        current_price: float,
        atr: float,
        initial_risk: float
    ) -> Position:
        """Core stop update logic."""

        if position.direction == 'long':
            # Move to breakeven after 1R
            if position.r_multiple >= self.breakeven_trigger_r:
                if position.stop_type == StopType.INITIAL:
                    # Move stop to just above entry
                    new_stop = position.entry_price + (initial_risk * 0.1)
                    if new_stop > position.current_stop:
                        position.current_stop = new_stop
                        position.stop_type = StopType.BREAKEVEN
                        logger.debug(
                            f"BREAKEVEN: {position.symbol} stop moved to "
                            f"{position.current_stop:.5f}"
                        )

            # Start trailing after 1.5R
            if position.r_multiple >= self.trail_activation_r:
                trail_stop = current_price - (atr * self.trail_distance_atr)
                if trail_stop > position.current_stop:
                    position.current_stop = trail_stop
                    position.stop_type = StopType.TRAILING
                    logger.debug(
                        f"TRAILING: {position.symbol} stop updated to "
                        f"{position.current_stop:.5f}"
                    )

        else:  # short
            # Move to breakeven after 1R
            if position.r_multiple >= self.breakeven_trigger_r:
                if position.stop_type == StopType.INITIAL:
                    new_stop = position.entry_price - (initial_risk * 0.1)
                    if new_stop < position.current_stop:
                        position.current_stop = new_stop
                        position.stop_type = StopType.BREAKEVEN
                        logger.debug(
                            f"BREAKEVEN: {position.symbol} stop moved to "
                            f"{position.current_stop:.5f}"
                        )

            # Start trailing after 1.5R
            if position.r_multiple >= self.trail_activation_r:
                trail_stop = current_price + (atr * self.trail_distance_atr)
                if trail_stop < position.current_stop:
                    position.current_stop = trail_stop
                    position.stop_type = StopType.TRAILING
                    logger.debug(
                        f"TRAILING: {position.symbol} stop updated to "
                        f"{position.current_stop:.5f}"
                    )

        return position

    def check_partial_profit(self, position: Position) -> Dict[str, Any]:
        """
        Check if we should take partial profits.

        Returns dict with:
        - take_profit: bool
        - size_reduction: float (0.33 = close 33%)
        - reason: str
        """
        # First partial at 1.5R
        if position.r_multiple >= self.partial_1_at_r and position.partials_taken == 0:
            return {
                'take_profit': True,
                'size_reduction': self.partial_1_size,
                'reason': f'partial_1_at_{self.partial_1_at_r}R'
            }

        # Second partial at 2.5R
        if position.r_multiple >= self.partial_2_at_r and position.partials_taken == 1:
            return {
                'take_profit': True,
                'size_reduction': self.partial_2_size,
                'reason': f'partial_2_at_{self.partial_2_at_r}R'
            }

        return {'take_profit': False}

    def check_exit(self, position: Position) -> Dict[str, Any]:
        """
        Check if position should be exited.

        Returns dict with:
        - exit: bool
        - reason: str
        - exit_price: float
        """
        if position.direction == 'long':
            # Stop loss hit
            if position.current_price <= position.current_stop:
                return {
                    'exit': True,
                    'reason': f'stop_{position.stop_type.value}',
                    'exit_price': position.current_stop
                }
            # Take profit hit
            if position.current_price >= position.take_profit:
                return {
                    'exit': True,
                    'reason': 'take_profit',
                    'exit_price': position.take_profit
                }

        else:  # short
            if position.current_price >= position.current_stop:
                return {
                    'exit': True,
                    'reason': f'stop_{position.stop_type.value}',
                    'exit_price': position.current_stop
                }
            if position.current_price <= position.take_profit:
                return {
                    'exit': True,
                    'reason': 'take_profit',
                    'exit_price': position.take_profit
                }

        return {'exit': False}


def simulate_trade_with_trailing(
    entry_price: float,
    direction: str,
    initial_stop: float,
    take_profit: float,
    price_series: List[float],
    atr: float,
    breakeven_r: float = 1.0,
    trail_r: float = 1.5,
    trail_atr: float = 2.0
) -> Dict[str, Any]:
    """
    Simulate a trade with trailing stop logic.

    Used in backtesting to simulate realistic exits.

    Args:
        entry_price: Entry price
        direction: 'long' or 'short'
        initial_stop: Initial stop loss
        take_profit: Take profit target
        price_series: List of subsequent prices (close prices)
        atr: ATR for trailing calculation
        breakeven_r: Move to BE at this R
        trail_r: Start trailing at this R
        trail_atr: Trail distance in ATR

    Returns:
        Dict with exit_price, exit_reason, bars_held, final_r
    """
    stop = initial_stop
    initial_risk = abs(entry_price - initial_stop)

    if initial_risk == 0:
        return {
            'exit_price': entry_price,
            'exit_reason': 'no_risk',
            'bars_held': 0,
            'final_r': 0.0
        }

    stop_type = 'initial'

    for i, price in enumerate(price_series):
        # Calculate R-multiple
        if direction == 'long':
            profit = price - entry_price
            r_multiple = profit / initial_risk

            # Check stop hit
            if price <= stop:
                return {
                    'exit_price': stop,
                    'exit_reason': f'stop_{stop_type}',
                    'bars_held': i + 1,
                    'final_r': (stop - entry_price) / initial_risk
                }

            # Check target hit
            if price >= take_profit:
                return {
                    'exit_price': take_profit,
                    'exit_reason': 'take_profit',
                    'bars_held': i + 1,
                    'final_r': (take_profit - entry_price) / initial_risk
                }

            # Update trailing stop
            if r_multiple >= breakeven_r and stop < entry_price:
                stop = entry_price + initial_risk * 0.1
                stop_type = 'breakeven'

            if r_multiple >= trail_r:
                trail_stop = price - atr * trail_atr
                if trail_stop > stop:
                    stop = trail_stop
                    stop_type = 'trailing'

        else:  # short
            profit = entry_price - price
            r_multiple = profit / initial_risk

            if price >= stop:
                return {
                    'exit_price': stop,
                    'exit_reason': f'stop_{stop_type}',
                    'bars_held': i + 1,
                    'final_r': (entry_price - stop) / initial_risk
                }

            if price <= take_profit:
                return {
                    'exit_price': take_profit,
                    'exit_reason': 'take_profit',
                    'bars_held': i + 1,
                    'final_r': (entry_price - take_profit) / initial_risk
                }

            if r_multiple >= breakeven_r and stop > entry_price:
                stop = entry_price - initial_risk * 0.1
                stop_type = 'breakeven'

            if r_multiple >= trail_r:
                trail_stop = price + atr * trail_atr
                if trail_stop < stop:
                    stop = trail_stop
                    stop_type = 'trailing'

    # Still in trade at end of data
    final_price = price_series[-1] if price_series else entry_price
    if direction == 'long':
        final_r = (final_price - entry_price) / initial_risk
    else:
        final_r = (entry_price - final_price) / initial_risk

    return {
        'exit_price': final_price,
        'exit_reason': 'end_of_data',
        'bars_held': len(price_series),
        'final_r': final_r
    }
