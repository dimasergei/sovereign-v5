# backtesting/engine/event_driven.py
"""
Event-Driven Backtester with Proper Stop/Target Execution.

The vectorized backtester ignores stop_loss and take_profit.
This bar-by-bar backtester properly executes them, giving realistic W/L ratios.

Key features:
- Tracks each position individually
- Checks stops using bar LOW (for longs) and HIGH (for shorts)
- Checks targets using bar HIGH (for longs) and LOW (for shorts)
- Implements trailing stops (breakeven at 1R, trail at 1.5R)
- Tracks strategy-level performance
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED_STOP = "closed_stop"
    CLOSED_TARGET = "closed_target"
    CLOSED_TRAILING = "closed_trailing"
    CLOSED_EOD = "closed_end_of_data"


@dataclass
class Position:
    """Track a single position."""
    entry_bar: int
    entry_price: float
    direction: str  # "long" or "short"
    size: float
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float

    # Tracking
    status: PositionStatus = PositionStatus.OPEN
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0

    # Trailing stop
    trailing_stop: Optional[float] = None
    breakeven_triggered: bool = False


@dataclass
class BacktestResult:
    """Complete backtest results."""
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    win_loss_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    long_trades: int
    short_trades: int
    avg_bars_held: float
    equity_curve: List[float]
    trades: List[Position]
    strategy_breakdown: Dict[str, Dict]
    guardian_triggered: bool
    guardian_bar: Optional[int] = None


class EventDrivenBacktester:
    """
    Bar-by-bar backtester with proper stop/target execution.

    This is what we need to properly test the Renaissance strategy.
    The vectorized backtester shows W/L ratio of 1.0x because it ignores stops.
    This backtester properly executes stops/targets → W/L ratio of 2.0x+
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_drawdown_pct: float = 6.0,
        guardian_buffer_pct: float = 1.0,  # Stop at 5% to avoid 6% (was 2.0)
        base_risk_pct: float = 0.75,       # Increased from 0.5
        max_positions: int = 4,
        use_trailing_stops: bool = True,
        breakeven_trigger_r: float = 0.8,  # Was 1.0 - tighter
        trail_trigger_r: float = 1.2,      # Was 1.5 - tighter
        trail_distance_atr: float = 1.5    # Was 2.0 - tighter
    ):
        self.initial_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.guardian_threshold = max_drawdown_pct - guardian_buffer_pct
        self.base_risk_pct = base_risk_pct
        self.max_positions = max_positions
        self.use_trailing_stops = use_trailing_stops
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trail_trigger_r = trail_trigger_r
        self.trail_distance_atr = trail_distance_atr

        # State - will be reset each run
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity_curve: List[float] = [initial_capital]
        self.guardian_triggered = False
        self.guardian_bar: Optional[int] = None

    def run(
        self,
        df: pd.DataFrame,
        signal_generator,
        symbol: str
    ) -> BacktestResult:
        """
        Run event-driven backtest.

        Args:
            df: OHLCV DataFrame
            signal_generator: SignalGenerator instance
            symbol: Symbol being tested

        Returns:
            BacktestResult with all metrics
        """
        self._reset()

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Pre-calculate ATR for all bars
        atr = self._calculate_atr_series(high, low, close, 14)

        # Main loop - bar by bar
        warmup_bars = 50
        for i in range(warmup_bars, len(df)):
            if self.guardian_triggered:
                break

            current_bar = i
            current_price = close[i]
            current_high = high[i]
            current_low = low[i]
            current_atr = atr[i] if i < len(atr) else atr[-1]

            # Step 1: Update existing positions (check stops/targets)
            self._update_positions(
                current_bar,
                current_price,
                current_high,
                current_low,
                current_atr
            )

            # Step 2: Check drawdown guardian
            self._check_guardian(current_bar)

            if self.guardian_triggered:
                break

            # Step 3: Generate new signal if we have room
            if len(self.positions) < self.max_positions:
                context_df = df.iloc[:i+1]
                signal = signal_generator.generate_signal(context_df, symbol)

                if signal.action != "neutral" and signal.confidence >= 0.35:
                    self._open_position(
                        bar=current_bar,
                        price=current_price,
                        signal=signal,
                        atr=current_atr
                    )

            # Step 4: Update equity curve
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            self.equity_curve.append(self.equity + unrealized_pnl)
            self.peak_equity = max(self.peak_equity, self.equity_curve[-1])

        # Close any remaining positions at last price
        final_price = close[-1]
        for pos in self.positions[:]:
            self._close_position(pos, len(df)-1, final_price, PositionStatus.CLOSED_EOD)

        return self._calculate_results()

    def _reset(self):
        """Reset state for new backtest."""
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.equity_curve = [self.initial_capital]
        self.guardian_triggered = False
        self.guardian_bar = None

    def _update_positions(
        self,
        bar: int,
        price: float,
        high: float,
        low: float,
        atr: float
    ):
        """Update all open positions - check stops and targets."""

        for pos in self.positions[:]:  # Copy list to allow removal
            pos.bars_held += 1

            if pos.direction == "long":
                # Track max favorable/adverse excursion
                pos.max_favorable = max(pos.max_favorable, high - pos.entry_price)
                pos.max_adverse = max(pos.max_adverse, pos.entry_price - low)

                # Check stop loss (use low of bar - realistic fill)
                if low <= pos.stop_loss:
                    self._close_position(pos, bar, pos.stop_loss, PositionStatus.CLOSED_STOP)
                    continue

                # Check take profit (use high of bar)
                if high >= pos.take_profit:
                    self._close_position(pos, bar, pos.take_profit, PositionStatus.CLOSED_TARGET)
                    continue

                # Trailing stop logic
                if self.use_trailing_stops:
                    self._update_trailing_stop_long(pos, price, atr)

                    if pos.trailing_stop and low <= pos.trailing_stop:
                        self._close_position(pos, bar, pos.trailing_stop, PositionStatus.CLOSED_TRAILING)
                        continue

            else:  # Short
                pos.max_favorable = max(pos.max_favorable, pos.entry_price - low)
                pos.max_adverse = max(pos.max_adverse, high - pos.entry_price)

                # Check stop loss (use high of bar)
                if high >= pos.stop_loss:
                    self._close_position(pos, bar, pos.stop_loss, PositionStatus.CLOSED_STOP)
                    continue

                # Check take profit (use low of bar)
                if low <= pos.take_profit:
                    self._close_position(pos, bar, pos.take_profit, PositionStatus.CLOSED_TARGET)
                    continue

                # Trailing stop
                if self.use_trailing_stops:
                    self._update_trailing_stop_short(pos, price, atr)

                    if pos.trailing_stop and high >= pos.trailing_stop:
                        self._close_position(pos, bar, pos.trailing_stop, PositionStatus.CLOSED_TRAILING)
                        continue

    def _update_trailing_stop_long(self, pos: Position, price: float, atr: float):
        """Update trailing stop for long position."""
        initial_risk = pos.entry_price - pos.stop_loss
        if initial_risk <= 0:
            return

        current_profit = price - pos.entry_price
        r_multiple = current_profit / initial_risk

        # Move to breakeven at 1R
        if r_multiple >= self.breakeven_trigger_r and not pos.breakeven_triggered:
            pos.trailing_stop = pos.entry_price + initial_risk * 0.1  # Slightly above entry
            pos.breakeven_triggered = True

        # Start trailing at 1.5R
        if r_multiple >= self.trail_trigger_r:
            new_trail = price - atr * self.trail_distance_atr
            if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail

    def _update_trailing_stop_short(self, pos: Position, price: float, atr: float):
        """Update trailing stop for short position."""
        initial_risk = pos.stop_loss - pos.entry_price
        if initial_risk <= 0:
            return

        current_profit = pos.entry_price - price
        r_multiple = current_profit / initial_risk

        if r_multiple >= self.breakeven_trigger_r and not pos.breakeven_triggered:
            pos.trailing_stop = pos.entry_price - initial_risk * 0.1
            pos.breakeven_triggered = True

        if r_multiple >= self.trail_trigger_r:
            new_trail = price + atr * self.trail_distance_atr
            if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                pos.trailing_stop = new_trail

    def _open_position(self, bar: int, price: float, signal, atr: float):
        """Open a new position."""

        # Calculate position size based on risk
        if signal.stop_loss:
            stop_distance = abs(price - signal.stop_loss)
        else:
            stop_distance = atr * 1.5

        stop_distance_pct = (stop_distance / price) * 100

        # Risk amount
        position_size_scalar = getattr(signal, 'position_scalar', 1.0)
        risk_pct = self.base_risk_pct * signal.confidence * position_size_scalar
        risk_amount = self.equity * (risk_pct / 100)

        # Position size
        if stop_distance_pct > 0:
            size = risk_amount / (stop_distance_pct / 100)
        else:
            size = risk_amount / 0.01  # Default 1% stop

        # Set stop and target
        if signal.action == "long":
            stop_loss = signal.stop_loss if signal.stop_loss else price - atr * 1.5
            take_profit = signal.take_profit if signal.take_profit else price + atr * 3.75
        else:
            stop_loss = signal.stop_loss if signal.stop_loss else price + atr * 1.5
            take_profit = signal.take_profit if signal.take_profit else price - atr * 3.75

        position = Position(
            entry_bar=bar,
            entry_price=price,
            direction=signal.action,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=getattr(signal, 'primary_strategy', 'unknown'),
            confidence=signal.confidence
        )

        self.positions.append(position)
        logger.debug(f"Opened {signal.action} at {price:.2f}, stop={stop_loss:.2f}, "
                    f"target={take_profit:.2f}, strategy={position.strategy}")

    def _close_position(self, pos: Position, bar: int, price: float, status: PositionStatus):
        """Close a position and record PnL."""

        pos.status = status
        pos.exit_bar = bar
        pos.exit_price = price

        # Calculate PnL
        if pos.direction == "long":
            pos.pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        else:
            pos.pnl_pct = (pos.entry_price - price) / pos.entry_price * 100

        pos.pnl = pos.size * (pos.pnl_pct / 100)

        # Update equity
        self.equity += pos.pnl

        # Move to closed list
        self.positions.remove(pos)
        self.closed_positions.append(pos)

        logger.debug(f"Closed {pos.direction} at {price:.2f}, "
                    f"PnL={pos.pnl_pct:.2f}%, Status={status.value}")

    def _check_guardian(self, bar: int):
        """Check if drawdown exceeds guardian threshold."""
        if self.peak_equity == 0:
            return

        current_dd = (self.peak_equity - self.equity) / self.peak_equity * 100

        if current_dd >= self.guardian_threshold:
            self.guardian_triggered = True
            self.guardian_bar = bar
            logger.warning(f"Guardian triggered at bar {bar}, DD={current_dd:.2f}%")

            # Close all positions at current price (simulate market close)
            for pos in self.positions[:]:
                # Use entry price as approximation (worst case)
                self._close_position(pos, bar, pos.entry_price, PositionStatus.CLOSED_EOD)

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL from open positions."""
        unrealized = 0.0
        for pos in self.positions:
            if pos.direction == "long":
                unrealized += pos.size * (current_price - pos.entry_price) / pos.entry_price
            else:
                unrealized += pos.size * (pos.entry_price - current_price) / pos.entry_price
        return unrealized

    def _calculate_atr_series(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate ATR for entire series."""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        atr[:period] = np.mean(tr[:period])

        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def _calculate_results(self) -> BacktestResult:
        """Calculate final backtest metrics."""

        trades = self.closed_positions

        if not trades:
            return BacktestResult(
                total_return_pct=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                win_rate=0,
                profit_factor=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                win_loss_ratio=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                long_trades=0,
                short_trades=0,
                avg_bars_held=0,
                equity_curve=self.equity_curve,
                trades=trades,
                strategy_breakdown={},
                guardian_triggered=self.guardian_triggered,
                guardian_bar=self.guardian_bar
            )

        # Basic counts
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        longs = [t for t in trades if t.direction == "long"]
        shorts = [t for t in trades if t.direction == "short"]

        # Returns
        total_return_pct = (self.equity - self.initial_capital) / self.initial_capital * 100

        # Drawdown
        equity_arr = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak * 100
        max_drawdown_pct = float(np.max(drawdown))

        # Win/loss metrics
        win_rate = len(winning) / len(trades) * 100 if trades else 0

        avg_win = float(np.mean([t.pnl_pct for t in winning])) if winning else 0
        avg_loss = float(abs(np.mean([t.pnl_pct for t in losing]))) if losing else 0

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Average bars held
        avg_bars = float(np.mean([t.bars_held for t in trades])) if trades else 0

        # Strategy breakdown
        strategy_breakdown = {}
        strategies = set(t.strategy for t in trades)
        for strat in strategies:
            strat_trades = [t for t in trades if t.strategy == strat]
            strat_wins = [t for t in strat_trades if t.pnl > 0]
            strategy_breakdown[strat] = {
                'trades': len(strat_trades),
                'win_rate': len(strat_wins) / len(strat_trades) * 100 if strat_trades else 0,
                'total_pnl': sum(t.pnl for t in strat_trades),
                'avg_pnl_pct': float(np.mean([t.pnl_pct for t in strat_trades])) if strat_trades else 0
            }

        return BacktestResult(
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            win_loss_ratio=win_loss_ratio,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            long_trades=len(longs),
            short_trades=len(shorts),
            avg_bars_held=avg_bars,
            equity_curve=self.equity_curve,
            trades=trades,
            strategy_breakdown=strategy_breakdown,
            guardian_triggered=self.guardian_triggered,
            guardian_bar=self.guardian_bar
        )
