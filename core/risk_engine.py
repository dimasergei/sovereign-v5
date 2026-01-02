"""
Risk Engine - Prop Firm Rule Enforcement.

CRITICAL: This is the guardian of the account.
NEVER disable or bypass these checks.

Features:
- Real-time drawdown tracking (balance-based trailing for GFT)
- Daily loss limits with automatic reset
- Position sizing based on Kelly criterion
- Consistency rule enforcement for The5ers
- Pre-trade validation and post-trade verification
- Emergency position liquidation
"""

import json
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum
from pathlib import Path

from .exceptions import RiskViolation, AccountLocked


logger = logging.getLogger(__name__)


class FirmType(Enum):
    """Supported prop firm types."""
    GFT = "goat_funded_trader"
    THE5ERS = "the5ers"


class ViolationType(Enum):
    """Types of risk violations."""
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    POSITION_SIZE_TOO_LARGE = "position_size_too_large"
    CONSISTENCY_RULE_VIOLATION = "consistency_rule_violation"
    PROHIBITED_INSTRUMENT = "prohibited_instrument"
    MAX_POSITIONS_EXCEEDED = "max_positions_exceeded"
    HEDGING_DETECTED = "hedging_detected"
    INACTIVITY_VIOLATION = "inactivity_violation"


@dataclass
class FirmRules:
    """
    Immutable prop firm rules - DO NOT MODIFY AT RUNTIME.
    """
    firm_type: FirmType
    initial_balance: float
    
    # Drawdown limits
    max_overall_drawdown_pct: float
    guardian_drawdown_pct: float  # Stop trading before actual limit
    
    # Daily limits (The5ers only)
    max_daily_loss_pct: Optional[float] = None
    guardian_daily_loss_pct: Optional[float] = None
    
    # Consistency rule (The5ers only)
    consistency_max_single_day_pct: Optional[float] = None
    
    # Position limits
    max_risk_per_trade_pct: float = 1.0
    max_open_positions: int = 5
    
    # Allowed instruments
    allowed_instruments: List[str] = field(default_factory=list)
    
    # Prohibited patterns
    allow_hedging: bool = False
    allow_martingale: bool = False
    
    # Inactivity
    max_inactivity_days: int = 30
    inactivity_warning_days: int = 25


@dataclass
class AccountRiskState:
    """
    Mutable risk state - persisted to JSON.
    """
    # Balance tracking
    initial_balance: float
    highest_balance: float  # High water mark for trailing DD
    current_balance: float
    current_equity: float
    
    # Daily tracking
    daily_starting_balance: float
    daily_pnl: float
    daily_date: str
    
    # Profit tracking
    total_realized_profit: float = 0.0
    daily_profit_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Trade tracking
    last_trade_time: Optional[str] = None
    days_since_last_trade: int = 0
    total_trades: int = 0
    
    # Lock state
    is_locked: bool = False
    lock_reason: Optional[str] = None
    lock_time: Optional[str] = None
    
    # Statistics
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0


class RiskManager:
    """
    Central risk management system.
    
    INVARIANTS (must always hold):
    1. current_drawdown < guardian_drawdown
    2. daily_loss < guardian_daily_loss (if applicable)
    3. position_risk <= max_risk_per_trade
    4. No trades on prohibited instruments
    5. No prohibited patterns (hedging, martingale)
    """
    
    def __init__(
        self,
        rules: FirmRules,
        state: AccountRiskState,
        state_file: str,
        on_violation_callback: Optional[Callable] = None
    ):
        """
        Initialize risk manager.
        
        Args:
            rules: Firm-specific trading rules
            state: Current account state
            state_file: Path to persist state
            on_violation_callback: Called when violation detected
        """
        self.rules = rules
        self.state = state
        self.state_file = Path(state_file)
        self.on_violation = on_violation_callback
        
        # Ensure state directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"RiskManager initialized for {rules.firm_type.value}: "
            f"DD limit={rules.max_overall_drawdown_pct}%, "
            f"Guardian={rules.guardian_drawdown_pct}%"
        )
    
    def update_account_state(self, balance: float, equity: float) -> Dict[str, Any]:
        """
        Update account balances and check for violations.
        
        Args:
            balance: Current account balance
            equity: Current account equity
            
        Returns:
            Dictionary with current risk metrics
        """
        self.state.current_balance = balance
        self.state.current_equity = equity
        
        # Update high water mark
        if balance > self.state.highest_balance:
            self.state.highest_balance = balance
            logger.info(f"New high water mark: ${balance:.2f}")
        
        # Check for daily reset
        today = date.today().isoformat()
        if self.state.daily_date != today:
            self._reset_daily_tracking(today)
        
        # Update daily P&L
        self.state.daily_pnl = balance - self.state.daily_starting_balance
        
        # Check for violations
        violations = self._check_all_violations()
        
        # Persist state
        self._save_state()
        
        return {
            "current_drawdown_pct": self.get_current_drawdown_pct(),
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_pct": self.get_daily_loss_pct(),
            "distance_to_guardian": self.rules.guardian_drawdown_pct - self.get_current_drawdown_pct(),
            "is_locked": self.state.is_locked,
            "violations": violations,
        }
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        symbol_info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate safe position size using Kelly criterion with risk scaling.
        
        Args:
            symbol: Trading symbol
            entry_price: Intended entry price
            stop_loss_price: Stop loss price
            account_equity: Current account equity
            symbol_info: Symbol specifications from MT5
            
        Returns:
            Tuple of (lot_size, calculation_details)
        """
        if self.state.is_locked:
            return 0.0, {"error": "Account locked", "reason": self.state.lock_reason}
        
        # Calculate base risk percentage using Kelly criterion
        base_risk_pct = self._calculate_kelly_risk()
        
        # Apply drawdown scaling - reduce risk as we approach limit
        dd_scaled_risk = self._apply_drawdown_scaling(base_risk_pct)
        
        # Apply daily loss scaling for The5ers
        if self.rules.max_daily_loss_pct:
            dd_scaled_risk = self._apply_daily_loss_scaling(dd_scaled_risk)
        
        # Calculate risk amount in dollars
        risk_amount = account_equity * (dd_scaled_risk / 100)
        
        # Calculate stop loss distance
        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance == 0:
            return 0.0, {"error": "Invalid stop loss (same as entry)"}
        
        # Calculate lot size based on risk
        tick_value = symbol_info.get("trade_tick_value", 1)
        tick_size = symbol_info.get("trade_tick_size", 0.01)
        
        if tick_size == 0:
            return 0.0, {"error": "Invalid tick size"}
        
        sl_ticks = sl_distance / tick_size
        
        if sl_ticks == 0 or tick_value == 0:
            return 0.0, {"error": "Cannot calculate position size"}
        
        lot_size = risk_amount / (sl_ticks * tick_value)
        
        # Apply lot constraints
        vol_min = symbol_info.get("volume_min", 0.01)
        vol_max = symbol_info.get("volume_max", 100)
        vol_step = symbol_info.get("volume_step", 0.01)
        
        lot_size = max(vol_min, lot_size)
        lot_size = min(vol_max, lot_size)
        
        # Round to step
        if vol_step > 0:
            lot_size = round(lot_size / vol_step) * vol_step
        
        calculation_details = {
            "account_equity": account_equity,
            "base_risk_pct": base_risk_pct,
            "dd_scaled_risk_pct": dd_scaled_risk,
            "risk_amount": risk_amount,
            "sl_distance": sl_distance,
            "sl_ticks": sl_ticks,
            "tick_value": tick_value,
            "raw_lot_size": risk_amount / (sl_ticks * tick_value) if sl_ticks * tick_value > 0 else 0,
            "final_lot_size": lot_size,
            "current_drawdown_pct": self.get_current_drawdown_pct(),
        }
        
        logger.debug(f"Position size calculated: {lot_size} lots ({calculation_details})")
        
        return lot_size, calculation_details
    
    def validate_trade(
        self,
        symbol: str,
        lot_size: float,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, Optional[ViolationType], str]:
        """
        Pre-trade validation against all rules.
        
        Args:
            symbol: Trading symbol
            lot_size: Position size in lots
            direction: "buy" or "sell"
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            
        Returns:
            Tuple of (is_valid, violation_type, message)
        """
        # Check if account is locked
        if self.state.is_locked:
            return False, ViolationType.MAX_DRAWDOWN_EXCEEDED, \
                f"Account locked: {self.state.lock_reason}"
        
        # Check instrument is allowed
        if not self._is_instrument_allowed(symbol):
            return False, ViolationType.PROHIBITED_INSTRUMENT, \
                f"Symbol {symbol} not in allowed list"
        
        # Check position count
        open_positions = self._get_open_position_count()
        if open_positions >= self.rules.max_open_positions:
            return False, ViolationType.MAX_POSITIONS_EXCEEDED, \
                f"Max {self.rules.max_open_positions} positions allowed"
        
        # Check for hedging
        if not self.rules.allow_hedging:
            if self._would_create_hedge(symbol, direction):
                return False, ViolationType.HEDGING_DETECTED, \
                    "Hedging is prohibited"
        
        # Check drawdown limit
        current_dd = self.get_current_drawdown_pct()
        if current_dd >= self.rules.guardian_drawdown_pct:
            return False, ViolationType.MAX_DRAWDOWN_EXCEEDED, \
                f"Drawdown {current_dd:.2f}% exceeds guardian limit {self.rules.guardian_drawdown_pct:.2f}%"
        
        # Check daily loss limit (The5ers)
        if self.rules.guardian_daily_loss_pct:
            daily_loss_pct = self.get_daily_loss_pct()
            if daily_loss_pct >= self.rules.guardian_daily_loss_pct:
                return False, ViolationType.DAILY_LOSS_EXCEEDED, \
                    f"Daily loss {daily_loss_pct:.2f}% exceeds guardian limit {self.rules.guardian_daily_loss_pct:.2f}%"
        
        # Check consistency rule (The5ers)
        if self.rules.consistency_max_single_day_pct:
            if not self.check_consistency_rule(0):  # Check if we can trade at all
                return False, ViolationType.CONSISTENCY_RULE_VIOLATION, \
                    "Would violate consistency rule"
        
        return True, None, "Trade validated"
    
    def record_trade_result(self, pnl: float, is_winner: bool):
        """
        Record trade result for statistics and Kelly calculation.
        
        Args:
            pnl: Profit/loss amount
            is_winner: True if trade was profitable
        """
        self.state.total_trades += 1
        
        if is_winner:
            self.state.winning_trades += 1
            self.state.total_profit += pnl
        else:
            self.state.losing_trades += 1
            self.state.total_loss += abs(pnl)
        
        self.state.last_trade_time = datetime.now().isoformat()
        self.state.days_since_last_trade = 0
        
        # Update daily profit distribution
        today = date.today().isoformat()
        current_daily = self.state.daily_profit_distribution.get(today, 0)
        self.state.daily_profit_distribution[today] = current_daily + pnl
        
        if pnl > 0:
            self.state.total_realized_profit += pnl
        
        self._save_state()
        
        logger.info(
            f"Trade recorded: PnL=${pnl:.2f}, "
            f"Win rate={self.get_win_rate():.1f}%"
        )
    
    def check_consistency_rule(self, potential_daily_profit: float) -> bool:
        """
        Check The5ers consistency rule.
        
        No single day's profit can exceed 30% of total profit.
        
        Args:
            potential_daily_profit: Additional profit that would be added today
            
        Returns:
            True if trade is allowed, False if would violate rule
        """
        if not self.rules.consistency_max_single_day_pct:
            return True
        
        # Get today's current profit
        today = date.today().isoformat()
        current_daily = self.state.daily_profit_distribution.get(today, 0)
        projected_daily = current_daily + potential_daily_profit
        
        # Calculate total profit including this trade
        total_profit = self.state.total_realized_profit + potential_daily_profit
        
        if total_profit <= 0:
            return True
        
        # Check if any single day exceeds the limit
        max_allowed = total_profit * (self.rules.consistency_max_single_day_pct / 100)
        
        return projected_daily <= max_allowed
    
    def get_current_drawdown_pct(self) -> float:
        """Calculate current drawdown from high water mark."""
        if self.state.highest_balance == 0:
            return 0.0
        
        return ((self.state.highest_balance - self.state.current_equity) / 
                self.state.highest_balance) * 100
    
    def get_daily_loss_pct(self) -> float:
        """Calculate current daily loss percentage."""
        if self.state.daily_starting_balance == 0:
            return 0.0
        
        if self.state.daily_pnl >= 0:
            return 0.0
        
        return (abs(self.state.daily_pnl) / self.state.daily_starting_balance) * 100
    
    def get_win_rate(self) -> float:
        """Calculate win rate from recorded trades."""
        total = self.state.winning_trades + self.state.losing_trades
        if total == 0:
            return 50.0  # Default assumption
        return (self.state.winning_trades / total) * 100
    
    def get_profit_factor(self) -> float:
        """Calculate profit factor."""
        if self.state.total_loss == 0:
            return float('inf') if self.state.total_profit > 0 else 1.0
        return self.state.total_profit / self.state.total_loss
    
    def get_avg_win_loss_ratio(self) -> float:
        """Calculate average win to average loss ratio."""
        avg_win = self.state.total_profit / max(1, self.state.winning_trades)
        avg_loss = self.state.total_loss / max(1, self.state.losing_trades)
        
        if avg_loss == 0:
            return 1.0
        return avg_win / avg_loss
    
    def emergency_close_all(self, reason: str):
        """
        Emergency liquidation trigger.
        
        Args:
            reason: Reason for emergency close
        """
        logger.critical(f"EMERGENCY CLOSE ALL: {reason}")
        
        self.state.is_locked = True
        self.state.lock_reason = reason
        self.state.lock_time = datetime.now().isoformat()
        
        self._save_state()
        
        if self.on_violation:
            self.on_violation(ViolationType.MAX_DRAWDOWN_EXCEEDED, reason)
    
    def unlock_account(self, confirmation: str = "CONFIRM"):
        """
        Unlock account after manual review.
        
        Args:
            confirmation: Must be "CONFIRM" to unlock
        """
        if confirmation != "CONFIRM":
            logger.warning("Account unlock requires confirmation='CONFIRM'")
            return False
        
        logger.warning("Account unlocked by user")
        self.state.is_locked = False
        self.state.lock_reason = None
        self.state.lock_time = None
        
        self._save_state()
        return True
    
    def check_inactivity(self) -> bool:
        """
        Check for inactivity and return True if trade needed.
        
        Returns:
            True if ping trade is needed to avoid inactivity violation
        """
        if self.state.last_trade_time is None:
            return True
        
        last_trade = datetime.fromisoformat(self.state.last_trade_time)
        days_inactive = (datetime.now() - last_trade).days
        self.state.days_since_last_trade = days_inactive
        
        if days_inactive >= self.rules.inactivity_warning_days:
            logger.warning(
                f"Inactivity warning: {days_inactive} days since last trade "
                f"(limit: {self.rules.max_inactivity_days})"
            )
            return True
        
        return False
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status report."""
        return {
            "firm_type": self.rules.firm_type.value,
            "is_locked": self.state.is_locked,
            "lock_reason": self.state.lock_reason,
            
            "drawdown": {
                "current_pct": round(self.get_current_drawdown_pct(), 2),
                "guardian_pct": self.rules.guardian_drawdown_pct,
                "max_pct": self.rules.max_overall_drawdown_pct,
                "distance_to_guardian": round(
                    self.rules.guardian_drawdown_pct - self.get_current_drawdown_pct(), 2
                ),
            },
            
            "daily_loss": {
                "current_pct": round(self.get_daily_loss_pct(), 2),
                "guardian_pct": self.rules.guardian_daily_loss_pct,
                "max_pct": self.rules.max_daily_loss_pct,
                "pnl_dollars": round(self.state.daily_pnl, 2),
            } if self.rules.max_daily_loss_pct else None,
            
            "balances": {
                "initial": self.state.initial_balance,
                "current": round(self.state.current_balance, 2),
                "equity": round(self.state.current_equity, 2),
                "high_water_mark": round(self.state.highest_balance, 2),
            },
            
            "statistics": {
                "total_trades": self.state.total_trades,
                "win_rate": round(self.get_win_rate(), 1),
                "profit_factor": round(self.get_profit_factor(), 2),
                "total_realized_profit": round(self.state.total_realized_profit, 2),
            },
            
            "inactivity": {
                "days_since_trade": self.state.days_since_last_trade,
                "warning_days": self.rules.inactivity_warning_days,
                "max_days": self.rules.max_inactivity_days,
            },
        }
    
    # ==================== PRIVATE METHODS ====================
    
    def _calculate_kelly_risk(self) -> float:
        """
        Calculate optimal risk fraction using Kelly Criterion.
        
        Kelly formula: f = (p * b - q) / b
        where:
            p = win probability
            b = win/loss ratio
            q = 1 - p
        """
        # Get historical statistics
        win_rate = self.get_win_rate() / 100  # Convert to decimal
        
        if self.state.total_trades < 10:
            # Not enough data, use conservative default
            return min(self.rules.max_risk_per_trade_pct, 0.5)
        
        # Calculate win/loss ratio
        win_loss_ratio = self.get_avg_win_loss_ratio()
        
        if win_loss_ratio <= 0:
            return 0.25  # Minimum risk
        
        # Kelly formula
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Use half-Kelly for safety (common practice)
        half_kelly = kelly / 2
        
        # Bound to reasonable range
        risk_pct = max(0.1, min(self.rules.max_risk_per_trade_pct, half_kelly * 100))
        
        logger.debug(
            f"Kelly calculation: win_rate={win_rate:.2f}, ratio={win_loss_ratio:.2f}, "
            f"kelly={kelly:.4f}, risk={risk_pct:.2f}%"
        )
        
        return risk_pct
    
    def _apply_drawdown_scaling(self, base_risk: float) -> float:
        """
        Reduce risk as drawdown approaches limit.
        
        Args:
            base_risk: Base risk percentage
            
        Returns:
            Scaled risk percentage
        """
        current_dd = self.get_current_drawdown_pct()
        guardian = self.rules.guardian_drawdown_pct
        
        # Start reducing at 50% of guardian limit
        reduction_start = guardian * 0.5
        
        if current_dd <= reduction_start:
            return base_risk
        
        # Linear reduction from 100% at reduction_start to 25% at guardian
        reduction_range = guardian - reduction_start
        dd_in_range = current_dd - reduction_start
        
        if reduction_range <= 0:
            return base_risk * 0.25
        
        reduction_factor = 1 - (dd_in_range / reduction_range * 0.75)
        reduction_factor = max(0.25, reduction_factor)  # Minimum 25% of normal
        
        scaled_risk = base_risk * reduction_factor
        
        if scaled_risk < base_risk:
            logger.warning(
                f"Risk reduced due to drawdown: {base_risk:.2f}% -> {scaled_risk:.2f}% "
                f"(DD: {current_dd:.2f}%)"
            )
        
        return scaled_risk
    
    def _apply_daily_loss_scaling(self, base_risk: float) -> float:
        """
        Reduce risk as daily loss approaches limit (The5ers).
        
        Args:
            base_risk: Base risk percentage
            
        Returns:
            Scaled risk percentage
        """
        if not self.rules.guardian_daily_loss_pct:
            return base_risk
        
        daily_loss = self.get_daily_loss_pct()
        guardian = self.rules.guardian_daily_loss_pct
        
        # Start reducing at 50% of daily limit
        reduction_start = guardian * 0.5
        
        if daily_loss <= reduction_start:
            return base_risk
        
        # Calculate remaining daily budget
        remaining = guardian - daily_loss
        
        if remaining <= 0:
            return 0  # No more trading today
        
        # Scale risk to remaining budget
        budget_factor = remaining / guardian
        scaled_risk = base_risk * budget_factor
        
        logger.warning(
            f"Risk reduced due to daily loss: {base_risk:.2f}% -> {scaled_risk:.2f}% "
            f"(Daily loss: {daily_loss:.2f}%, remaining: {remaining:.2f}%)"
        )
        
        return scaled_risk
    
    def _check_all_violations(self) -> List[str]:
        """Check all rules and return list of violations."""
        violations = []
        
        # Check overall drawdown
        current_dd = self.get_current_drawdown_pct()
        
        if current_dd >= self.rules.max_overall_drawdown_pct:
            self.emergency_close_all(
                f"Max drawdown BREACHED: {current_dd:.2f}% >= {self.rules.max_overall_drawdown_pct:.2f}%"
            )
            violations.append("MAX_DRAWDOWN_BREACHED")
        elif current_dd >= self.rules.guardian_drawdown_pct:
            violations.append("GUARDIAN_DRAWDOWN_EXCEEDED")
            logger.warning(f"Guardian drawdown exceeded: {current_dd:.2f}%")
        
        # Check daily loss (The5ers)
        if self.rules.max_daily_loss_pct:
            daily_loss = self.get_daily_loss_pct()
            
            if daily_loss >= self.rules.max_daily_loss_pct:
                self.emergency_close_all(
                    f"Daily loss BREACHED: {daily_loss:.2f}% >= {self.rules.max_daily_loss_pct:.2f}%"
                )
                violations.append("DAILY_LOSS_BREACHED")
            elif daily_loss >= self.rules.guardian_daily_loss_pct:
                violations.append("GUARDIAN_DAILY_LOSS_EXCEEDED")
                logger.warning(f"Guardian daily loss exceeded: {daily_loss:.2f}%")
        
        return violations
    
    def _reset_daily_tracking(self, new_date: str):
        """Reset daily counters at start of new trading day."""
        logger.info(f"Daily reset: {self.state.daily_date} -> {new_date}")
        
        # Record yesterday's P&L
        if self.state.daily_pnl != 0:
            self.state.daily_profit_distribution[self.state.daily_date] = self.state.daily_pnl
        
        # Reset for new day
        self.state.daily_date = new_date
        self.state.daily_starting_balance = self.state.current_balance
        self.state.daily_pnl = 0.0
    
    def _is_instrument_allowed(self, symbol: str) -> bool:
        """Check if instrument is in allowed list."""
        if not self.rules.allowed_instruments:
            return True  # No restrictions
        
        # Normalize symbol
        symbol_upper = symbol.upper()
        
        for allowed in self.rules.allowed_instruments:
            if allowed.upper() in symbol_upper or symbol_upper in allowed.upper():
                return True
        
        return False
    
    def _get_open_position_count(self) -> int:
        """Get current number of open positions."""
        try:
            positions = mt5.positions_get()
            return len(positions) if positions else 0
        except:
            return 0
    
    def _would_create_hedge(self, symbol: str, direction: str) -> bool:
        """Check if trade would create a hedge."""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return False
            
            for pos in positions:
                existing_dir = "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell"
                if existing_dir != direction:
                    return True
            
            return False
        except:
            return False
    
    def _save_state(self):
        """Persist state to JSON file."""
        try:
            state_dict = asdict(self.state)
            state_dict["updated_at"] = datetime.now().isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    @classmethod
    def load_state(cls, state_file: str) -> Optional[AccountRiskState]:
        """Load state from JSON file."""
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            # Remove non-dataclass fields
            data.pop("updated_at", None)
            
            return AccountRiskState(**data)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None


# ==================== FACTORY FUNCTIONS ====================

def create_gft_rules(
    initial_balance: float = 10000,
    symbols: List[str] = None
) -> FirmRules:
    """Create rules for Goat Funded Trader accounts."""
    return FirmRules(
        firm_type=FirmType.GFT,
        initial_balance=initial_balance,
        max_overall_drawdown_pct=8.0,
        guardian_drawdown_pct=7.0,
        max_daily_loss_pct=None,  # GFT doesn't have daily limit
        guardian_daily_loss_pct=None,
        consistency_max_single_day_pct=None,
        max_risk_per_trade_pct=0.8,
        max_open_positions=3,
        allowed_instruments=symbols or [
            "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "LTCUSD",
            "BNBUSD", "ADAUSD", "DOTUSD", "AVAXUSD", "MATICUSD"
        ],
        allow_hedging=False,
        allow_martingale=False,
        max_inactivity_days=30,
        inactivity_warning_days=25,
    )


def create_the5ers_rules(
    initial_balance: float = 5000,
    symbols: List[str] = None
) -> FirmRules:
    """Create rules for The5ers accounts."""
    return FirmRules(
        firm_type=FirmType.THE5ERS,
        initial_balance=initial_balance,
        max_overall_drawdown_pct=10.0,
        guardian_drawdown_pct=8.5,
        max_daily_loss_pct=5.0,
        guardian_daily_loss_pct=4.0,
        consistency_max_single_day_pct=30.0,
        max_risk_per_trade_pct=0.5,
        max_open_positions=2,
        allowed_instruments=symbols or [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
            "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
        ],
        allow_hedging=False,
        allow_martingale=False,
        max_inactivity_days=30,
        inactivity_warning_days=25,
    )
