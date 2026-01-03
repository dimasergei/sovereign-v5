# core/paper_executor.py
"""
Paper Trading Executor - Full simulation with position tracking and compliance.

Simulates real trading without executing on MT5:
- Tracks open positions with entry price, stop loss, take profit
- Monitors for stop/target hits using market data
- Calculates running P&L and drawdown
- Enforces prop firm compliance rules
- Logs all trades for analysis

Usage:
    executor = PaperExecutor(
        initial_balance=10000,
        account_type="GFT",
        config=config
    )

    # Open a position
    result = executor.open_position(
        symbol="XAUUSD.x",
        direction="buy",
        size=0.1,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2680.0,
        signal_info={"strategy": "trend", "confidence": 0.75}
    )

    # Update with current prices (call every tick/bar)
    executor.update_positions(current_prices)

    # Get account state
    state = executor.get_account_state()
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import uuid

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status."""
    OPEN = "open"
    CLOSED_STOP = "closed_stop"
    CLOSED_TARGET = "closed_target"
    CLOSED_MANUAL = "closed_manual"
    CLOSED_GUARDIAN = "closed_guardian"


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class PaperPosition:
    """Paper trading position."""
    id: str
    symbol: str
    direction: TradeDirection
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float

    # Optional trailing stop
    trailing_stop: Optional[float] = None
    trailing_trigger: Optional[float] = None  # Price to start trailing

    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = 0.0   # For trailing stop

    # Closure info
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0

    # Metadata
    signal_info: Dict = field(default_factory=dict)
    comment: str = ""

    def update_price(self, price: float) -> None:
        """Update position with current price."""
        self.current_price = price

        # Track extremes for trailing stop
        if price > self.highest_price:
            self.highest_price = price
        if price < self.lowest_price or self.lowest_price == 0:
            self.lowest_price = price

        # Calculate unrealized P&L
        if self.direction == TradeDirection.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.size

        self.unrealized_pnl_pct = (self.unrealized_pnl / self.entry_price / self.size) * 100

    def check_stop_target(self, high: float, low: float) -> Optional[PositionStatus]:
        """
        Check if stop or target hit using bar high/low.

        Returns PositionStatus if closed, None if still open.
        """
        if self.status != PositionStatus.OPEN:
            return None

        if self.direction == TradeDirection.LONG:
            # Check stop loss (use bar low)
            if low <= self.stop_loss:
                return PositionStatus.CLOSED_STOP
            # Check take profit (use bar high)
            if high >= self.take_profit:
                return PositionStatus.CLOSED_TARGET
        else:
            # Short position
            # Check stop loss (use bar high)
            if high >= self.stop_loss:
                return PositionStatus.CLOSED_STOP
            # Check take profit (use bar low)
            if low <= self.take_profit:
                return PositionStatus.CLOSED_TARGET

        return None

    def close(self, exit_price: float, status: PositionStatus) -> float:
        """Close position and calculate realized P&L."""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.status = status

        if self.direction == TradeDirection.LONG:
            self.realized_pnl = (exit_price - self.entry_price) * self.size
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.size

        self.unrealized_pnl = 0.0
        return self.realized_pnl


@dataclass
class TradeRecord:
    """Record of a completed trade for logging."""
    id: str
    symbol: str
    direction: str
    size: float
    entry_price: float
    entry_time: str
    exit_price: float
    exit_time: str
    stop_loss: float
    take_profit: float
    realized_pnl: float
    realized_pnl_pct: float
    status: str
    duration_minutes: float
    signal_info: Dict
    comment: str


@dataclass
class AccountState:
    """Paper trading account state."""
    initial_balance: float
    balance: float
    equity: float
    open_positions: int
    unrealized_pnl: float
    realized_pnl: float

    # Drawdown tracking
    equity_hwm: float
    current_dd_pct: float
    max_dd_pct: float

    # Daily tracking
    daily_starting_equity: float
    daily_pnl: float
    daily_dd_pct: float

    # Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class PaperExecutor:
    """
    Paper trading executor with full position management and compliance.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        account_type: str = "GFT",
        account_name: str = "Paper",
        state_file: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize paper executor.

        Args:
            initial_balance: Starting account balance
            account_type: "GFT" or "THE5ERS" for compliance rules
            account_name: Account identifier
            state_file: Path to save/load state (optional)
            config: Account configuration dict
        """
        self.initial_balance = initial_balance
        self.account_type = account_type.upper()
        self.account_name = account_name
        self.state_file = state_file or f"storage/state/{account_name}_paper.json"
        self.config = config or {}

        # Account state
        self.balance = initial_balance
        self.equity = initial_balance
        self.equity_hwm = initial_balance
        self.max_dd_pct = 0.0

        # Daily tracking (resets at 5PM EST for GFT)
        self.daily_starting_equity = initial_balance
        self.daily_pnl = 0.0
        self.daily_date = datetime.now().date()

        # Positions
        self.open_positions: Dict[str, PaperPosition] = {}
        self.closed_positions: List[PaperPosition] = []
        self.trade_history: List[TradeRecord] = []

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_wins = 0.0
        self.total_losses = 0.0

        # Compliance
        self._init_compliance()

        # Load saved state if exists
        self._load_state()

        logger.info(
            f"[{account_name}] Paper executor initialized: "
            f"${initial_balance:.2f}, type={account_type}"
        )

    def _init_compliance(self):
        """Initialize compliance limits based on account type."""
        if self.account_type == "GFT":
            self.max_floating_loss_pct = 2.0
            self.guardian_floating_pct = 1.8
            self.max_daily_dd_pct = 3.0
            self.guardian_daily_dd_pct = 2.5
            self.max_total_dd_pct = 6.0
            self.guardian_total_dd_pct = 5.0
            self.daily_profit_cap = 3000.0
        else:  # THE5ERS
            self.max_floating_loss_pct = 10.0  # No per-trade limit
            self.guardian_floating_pct = 10.0
            self.max_daily_dd_pct = 5.0
            self.guardian_daily_dd_pct = 4.0
            self.max_total_dd_pct = 10.0
            self.guardian_total_dd_pct = 8.0
            self.daily_profit_cap = float('inf')

    def open_position(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_info: Optional[Dict] = None,
        comment: str = ""
    ) -> Tuple[bool, str, Optional[PaperPosition]]:
        """
        Open a new paper position.

        Args:
            symbol: Trading symbol
            direction: "buy"/"long" or "sell"/"short"
            size: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_info: Signal metadata
            comment: Trade comment

        Returns:
            Tuple of (success, message, position)
        """
        # Normalize direction
        dir_enum = TradeDirection.LONG if direction.lower() in ("buy", "long") else TradeDirection.SHORT

        # Pre-trade compliance checks
        compliance_ok, compliance_msg = self._check_pre_trade_compliance(
            symbol, size, entry_price, stop_loss
        )
        if not compliance_ok:
            logger.warning(f"[{self.account_name}] Trade blocked: {compliance_msg}")
            return False, compliance_msg, None

        # Create position
        position = PaperPosition(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=dir_enum,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            signal_info=signal_info or {},
            comment=comment
        )

        self.open_positions[position.id] = position
        self.total_trades += 1

        logger.info(
            f"[{self.account_name}] PAPER OPEN: {dir_enum.value.upper()} {symbol} "
            f"size={size} @ {entry_price:.2f} SL={stop_loss:.2f} TP={take_profit:.2f}"
        )

        self._save_state()
        return True, "Position opened", position

    def update_positions(
        self,
        prices: Dict[str, Dict[str, float]]
    ) -> List[PaperPosition]:
        """
        Update all positions with current prices and check stops/targets.

        Args:
            prices: Dict of symbol -> {bid, ask, high, low} or just {price}

        Returns:
            List of positions that were closed
        """
        closed = []

        # Check for daily reset
        self._check_daily_reset()

        for pos_id, pos in list(self.open_positions.items()):
            if pos.symbol not in prices:
                continue

            price_data = prices[pos.symbol]

            # Get current price (use bid for longs, ask for shorts)
            if isinstance(price_data, dict):
                if pos.direction == TradeDirection.LONG:
                    current = price_data.get('bid', price_data.get('price', price_data.get('close', 0)))
                else:
                    current = price_data.get('ask', price_data.get('price', price_data.get('close', 0)))
                high = price_data.get('high', current)
                low = price_data.get('low', current)
            else:
                current = high = low = price_data

            # Update position price
            pos.update_price(current)

            # Check floating loss compliance (GFT critical!)
            if self._check_floating_loss_breach(pos):
                exit_price = pos.stop_loss  # Use stop as exit
                pnl = pos.close(exit_price, PositionStatus.CLOSED_GUARDIAN)
                self._record_trade(pos)
                del self.open_positions[pos_id]
                closed.append(pos)
                self._update_balance(pnl)
                logger.critical(
                    f"[{self.account_name}] GUARDIAN CLOSE: {pos.symbol} "
                    f"floating loss approaching -2% limit"
                )
                continue

            # Check stop/target
            close_status = pos.check_stop_target(high, low)

            if close_status:
                if close_status == PositionStatus.CLOSED_STOP:
                    exit_price = pos.stop_loss
                else:
                    exit_price = pos.take_profit

                pnl = pos.close(exit_price, close_status)
                self._record_trade(pos)
                del self.open_positions[pos_id]
                closed.append(pos)
                self._update_balance(pnl)

                logger.info(
                    f"[{self.account_name}] PAPER CLOSE: {pos.symbol} "
                    f"{close_status.value} @ {exit_price:.2f} P&L=${pnl:.2f}"
                )

        # Update equity
        self._update_equity()

        # Check account-level compliance
        self._check_account_compliance()

        if closed:
            self._save_state()

        return closed

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Tuple[bool, float]:
        """
        Manually close a position.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            reason: Close reason

        Returns:
            Tuple of (success, realized_pnl)
        """
        if position_id not in self.open_positions:
            return False, 0.0

        pos = self.open_positions[position_id]
        status = PositionStatus.CLOSED_MANUAL

        pnl = pos.close(exit_price, status)
        self._record_trade(pos)
        del self.open_positions[position_id]
        self._update_balance(pnl)

        logger.info(
            f"[{self.account_name}] PAPER CLOSE (manual): {pos.symbol} "
            f"@ {exit_price:.2f} P&L=${pnl:.2f} reason={reason}"
        )

        self._save_state()
        return True, pnl

    def close_all_positions(self, prices: Dict[str, float], reason: str = "close_all") -> float:
        """Close all open positions at current prices."""
        total_pnl = 0.0

        for pos_id in list(self.open_positions.keys()):
            pos = self.open_positions[pos_id]
            exit_price = prices.get(pos.symbol, pos.current_price)
            _, pnl = self.close_position(pos_id, exit_price, reason)
            total_pnl += pnl

        return total_pnl

    def get_account_state(self) -> AccountState:
        """Get current account state."""
        unrealized = sum(p.unrealized_pnl for p in self.open_positions.values())
        realized = self.balance - self.initial_balance

        # Calculate drawdown
        current_dd = ((self.equity_hwm - self.equity) / self.equity_hwm * 100) if self.equity_hwm > 0 else 0
        daily_dd = ((self.daily_starting_equity - self.equity) / self.daily_starting_equity * 100) if self.daily_starting_equity > 0 else 0

        # Win/loss stats
        total = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total * 100) if total > 0 else 0
        avg_win = (self.total_wins / self.winning_trades) if self.winning_trades > 0 else 0
        avg_loss = (self.total_losses / self.losing_trades) if self.losing_trades > 0 else 0
        profit_factor = (self.total_wins / self.total_losses) if self.total_losses > 0 else float('inf')

        return AccountState(
            initial_balance=self.initial_balance,
            balance=self.balance,
            equity=self.equity,
            open_positions=len(self.open_positions),
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            equity_hwm=self.equity_hwm,
            current_dd_pct=max(0, current_dd),
            max_dd_pct=self.max_dd_pct,
            daily_starting_equity=self.daily_starting_equity,
            daily_pnl=self.daily_pnl,
            daily_dd_pct=max(0, daily_dd),
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor
        )

    def get_open_positions(self) -> List[PaperPosition]:
        """Get list of open positions."""
        return list(self.open_positions.values())

    def get_trade_history(self) -> List[TradeRecord]:
        """Get completed trade history."""
        return self.trade_history

    def _check_pre_trade_compliance(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[bool, str]:
        """Check compliance before opening trade."""
        # Check position limit
        max_positions = self.config.get('MAX_POSITIONS', 3)
        if len(self.open_positions) >= max_positions:
            return False, f"Max positions ({max_positions}) reached"

        # Check total DD guardian
        current_dd = ((self.equity_hwm - self.equity) / self.equity_hwm * 100) if self.equity_hwm > 0 else 0
        if current_dd >= self.guardian_total_dd_pct:
            return False, f"Total DD guardian triggered ({current_dd:.1f}% >= {self.guardian_total_dd_pct}%)"

        # Check daily DD guardian
        daily_dd = ((self.daily_starting_equity - self.equity) / self.daily_starting_equity * 100) if self.daily_starting_equity > 0 else 0
        if daily_dd >= self.guardian_daily_dd_pct:
            return False, f"Daily DD guardian triggered ({daily_dd:.1f}% >= {self.guardian_daily_dd_pct}%)"

        # GFT: Check if stop would breach -2% floating loss
        if self.account_type == "GFT":
            stop_distance = abs(entry_price - stop_loss)
            potential_loss = stop_distance * size
            potential_loss_pct = (potential_loss / self.equity * 100) if self.equity > 0 else 100

            if potential_loss_pct > self.guardian_floating_pct:
                return False, f"Stop too wide for GFT ({potential_loss_pct:.1f}% > {self.guardian_floating_pct}%)"

        # Check daily profit cap (GFT)
        if self.daily_pnl >= self.daily_profit_cap * 0.9:
            return False, f"Approaching daily profit cap (${self.daily_pnl:.0f})"

        return True, "OK"

    def _check_floating_loss_breach(self, pos: PaperPosition) -> bool:
        """Check if position is approaching floating loss limit."""
        if self.account_type != "GFT":
            return False

        floating_pct = abs(pos.unrealized_pnl / self.equity * 100) if self.equity > 0 else 0

        # Only care about losses
        if pos.unrealized_pnl >= 0:
            return False

        return floating_pct >= self.guardian_floating_pct

    def _check_account_compliance(self):
        """Check account-level compliance and stop trading if needed."""
        current_dd = ((self.equity_hwm - self.equity) / self.equity_hwm * 100) if self.equity_hwm > 0 else 0

        if current_dd >= self.guardian_total_dd_pct:
            logger.critical(
                f"[{self.account_name}] TOTAL DD GUARDIAN: {current_dd:.1f}% - STOP TRADING"
            )

        daily_dd = ((self.daily_starting_equity - self.equity) / self.daily_starting_equity * 100) if self.daily_starting_equity > 0 else 0

        if daily_dd >= self.guardian_daily_dd_pct:
            logger.critical(
                f"[{self.account_name}] DAILY DD GUARDIAN: {daily_dd:.1f}% - STOP TRADING"
            )

    def _update_balance(self, pnl: float):
        """Update balance after trade closes."""
        self.balance += pnl
        self.daily_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
            self.total_wins += pnl
        else:
            self.losing_trades += 1
            self.total_losses += abs(pnl)

        self._update_equity()

    def _update_equity(self):
        """Update equity and HWM."""
        unrealized = sum(p.unrealized_pnl for p in self.open_positions.values())
        self.equity = self.balance + unrealized

        # Update HWM
        if self.equity > self.equity_hwm:
            self.equity_hwm = self.equity

        # Update max DD
        current_dd = ((self.equity_hwm - self.equity) / self.equity_hwm * 100) if self.equity_hwm > 0 else 0
        if current_dd > self.max_dd_pct:
            self.max_dd_pct = current_dd

    def _check_daily_reset(self):
        """Check if daily stats should reset (5 PM EST for GFT)."""
        today = datetime.now().date()
        if today != self.daily_date:
            logger.info(f"[{self.account_name}] Daily reset: previous P&L ${self.daily_pnl:.2f}")
            self.daily_starting_equity = self.equity
            self.daily_pnl = 0.0
            self.daily_date = today

    def _record_trade(self, pos: PaperPosition):
        """Record completed trade to history."""
        duration = (pos.exit_time - pos.entry_time).total_seconds() / 60 if pos.exit_time else 0
        pnl_pct = (pos.realized_pnl / pos.entry_price / pos.size * 100) if pos.size > 0 else 0

        record = TradeRecord(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction.value,
            size=pos.size,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time.isoformat(),
            exit_price=pos.exit_price or 0,
            exit_time=pos.exit_time.isoformat() if pos.exit_time else "",
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            realized_pnl=pos.realized_pnl,
            realized_pnl_pct=pnl_pct,
            status=pos.status.value,
            duration_minutes=duration,
            signal_info=pos.signal_info,
            comment=pos.comment
        )

        self.trade_history.append(record)
        self.closed_positions.append(pos)

    def _save_state(self):
        """Save current state to file."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

            state = {
                "account_name": self.account_name,
                "account_type": self.account_type,
                "initial_balance": self.initial_balance,
                "balance": self.balance,
                "equity": self.equity,
                "equity_hwm": self.equity_hwm,
                "max_dd_pct": self.max_dd_pct,
                "daily_starting_equity": self.daily_starting_equity,
                "daily_pnl": self.daily_pnl,
                "daily_date": self.daily_date.isoformat(),
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "trade_history": [asdict(t) for t in self.trade_history[-100:]],  # Last 100
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save paper state: {e}")

    def _load_state(self):
        """Load state from file if exists."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.balance = state.get("balance", self.initial_balance)
                self.equity = state.get("equity", self.initial_balance)
                self.equity_hwm = state.get("equity_hwm", self.initial_balance)
                self.max_dd_pct = state.get("max_dd_pct", 0.0)
                self.daily_starting_equity = state.get("daily_starting_equity", self.initial_balance)
                self.daily_pnl = state.get("daily_pnl", 0.0)
                self.total_trades = state.get("total_trades", 0)
                self.winning_trades = state.get("winning_trades", 0)
                self.losing_trades = state.get("losing_trades", 0)
                self.total_wins = state.get("total_wins", 0.0)
                self.total_losses = state.get("total_losses", 0.0)

                logger.info(f"[{self.account_name}] Loaded paper state: balance=${self.balance:.2f}")

        except Exception as e:
            logger.warning(f"Failed to load paper state: {e}")

    def reset(self):
        """Reset account to initial state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_hwm = self.initial_balance
        self.max_dd_pct = 0.0
        self.daily_starting_equity = self.initial_balance
        self.daily_pnl = 0.0
        self.daily_date = datetime.now().date()
        self.open_positions.clear()
        self.closed_positions.clear()
        self.trade_history.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_wins = 0.0
        self.total_losses = 0.0

        self._save_state()
        logger.info(f"[{self.account_name}] Paper account reset to ${self.initial_balance:.2f}")

    def print_summary(self):
        """Print account summary."""
        state = self.get_account_state()

        print(f"\n{'='*60}")
        print(f"  PAPER TRADING SUMMARY: {self.account_name}")
        print(f"{'='*60}")
        print(f"  Account Type:     {self.account_type}")
        print(f"  Initial Balance:  ${state.initial_balance:,.2f}")
        print(f"  Current Balance:  ${state.balance:,.2f}")
        print(f"  Current Equity:   ${state.equity:,.2f}")
        print(f"  Realized P&L:     ${state.realized_pnl:+,.2f}")
        print(f"  Unrealized P&L:   ${state.unrealized_pnl:+,.2f}")
        print(f"  Open Positions:   {state.open_positions}")
        print(f"{'='*60}")
        print(f"  Current DD:       {state.current_dd_pct:.2f}%")
        print(f"  Max DD:           {state.max_dd_pct:.2f}%")
        print(f"  Daily P&L:        ${state.daily_pnl:+,.2f}")
        print(f"  Daily DD:         {state.daily_dd_pct:.2f}%")
        print(f"{'='*60}")
        print(f"  Total Trades:     {state.total_trades}")
        print(f"  Win Rate:         {state.win_rate:.1f}%")
        print(f"  Avg Win:          ${state.avg_win:.2f}")
        print(f"  Avg Loss:         ${state.avg_loss:.2f}")
        print(f"  Profit Factor:    {state.profit_factor:.2f}")
        print(f"{'='*60}\n")
