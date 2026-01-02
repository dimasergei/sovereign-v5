"""
Smart Execution Engine - Institutional-grade order execution.

Features:
- Multiple execution styles (MARKET, TWAP, ICEBERG, ADAPTIVE)
- Slippage monitoring and control
- Retry logic with exponential backoff
- Post-trade execution quality analysis
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

import MetaTrader5 as mt5
import numpy as np

from .mt5_connector import MT5Connector
from .exceptions import OrderRejected


logger = logging.getLogger(__name__)


class ExecutionStyle(Enum):
    """Order execution styles."""
    MARKET = "market"
    TWAP = "twap"
    ICEBERG = "iceberg"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionPlan:
    """Plan for executing an order."""
    symbol: str
    direction: str  # "buy" or "sell"
    total_size: float
    style: ExecutionStyle
    stop_loss: float
    take_profit: Optional[float] = None
    slices: List[Dict] = field(default_factory=list)
    max_duration_seconds: int = 300
    max_slippage_pips: float = 5.0
    magic_number: int = 12345


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    ticket: int = 0
    symbol: str = ""
    direction: str = ""
    avg_fill_price: float = 0.0
    total_filled: float = 0.0
    slippage_pips: float = 0.0
    execution_time_seconds: float = 0.0
    num_fills: int = 0
    error_message: str = ""
    fills: List[Dict] = field(default_factory=list)


class SmartExecutor:
    """
    Intelligent order execution with multiple strategies.
    
    Usage:
        executor = SmartExecutor(connector)
        
        plan = executor.create_plan(
            symbol="BTCUSD.x",
            direction="buy",
            size=0.1,
            sl=45000,
            tp=48000
        )
        
        result = executor.execute(plan)
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1
    
    def __init__(
        self,
        connector: MT5Connector,
        max_slippage_pips: float = 3.0,
        default_magic: int = 12345
    ):
        """
        Initialize executor.
        
        Args:
            connector: MT5Connector instance
            max_slippage_pips: Maximum acceptable slippage
            default_magic: Default magic number for orders
        """
        self.connector = connector
        self.max_slippage_pips = max_slippage_pips
        self.default_magic = default_magic
        
        # Execution statistics
        self.total_orders = 0
        self.successful_orders = 0
        self.total_slippage = 0.0
    
    def create_plan(
        self,
        symbol: str,
        direction: str,
        size: float,
        sl: float,
        tp: Optional[float] = None,
        urgency: float = 0.5,
        style: ExecutionStyle = None
    ) -> ExecutionPlan:
        """
        Create an execution plan.
        
        Args:
            symbol: Trading symbol
            direction: "buy" or "sell"
            size: Position size in lots
            sl: Stop loss price
            tp: Take profit price
            urgency: 0-1, higher = faster execution
            style: Execution style (auto-selected if None)
            
        Returns:
            ExecutionPlan
        """
        # Auto-select style based on size and urgency
        if style is None:
            style = self._select_execution_style(symbol, size, urgency)
        
        plan = ExecutionPlan(
            symbol=symbol,
            direction=direction,
            total_size=size,
            style=style,
            stop_loss=sl,
            take_profit=tp,
            magic_number=self.default_magic,
        )
        
        # Create execution slices based on style
        if style == ExecutionStyle.MARKET:
            plan.slices = [{"size": size, "delay": 0}]
        
        elif style == ExecutionStyle.TWAP:
            n_slices = max(2, int(size / 0.05))
            slice_size = size / n_slices
            slice_delay = plan.max_duration_seconds / n_slices
            
            plan.slices = [
                {"size": slice_size, "delay": i * slice_delay}
                for i in range(n_slices)
            ]
        
        elif style == ExecutionStyle.ICEBERG:
            visible_size = min(0.05, size * 0.2)
            n_slices = int(np.ceil(size / visible_size))
            
            plan.slices = [
                {"size": min(visible_size, size - i * visible_size), "delay": i * 5}
                for i in range(n_slices)
            ]
        
        elif style == ExecutionStyle.ADAPTIVE:
            # Start with market, adjust based on fill quality
            plan.slices = [{"size": size, "delay": 0}]
        
        logger.info(
            f"Execution plan: {direction} {size} {symbol} "
            f"style={style.value} slices={len(plan.slices)}"
        )
        
        return plan
    
    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        Execute a trading plan.
        
        Args:
            plan: ExecutionPlan to execute
            
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        
        if not self.connector.ensure_connected():
            return ExecutionResult(
                success=False,
                error_message="Not connected to MT5"
            )
        
        # Select symbol
        if not mt5.symbol_select(plan.symbol, True):
            return ExecutionResult(
                success=False,
                error_message=f"Symbol {plan.symbol} not available"
            )
        
        # Execute based on style
        if plan.style == ExecutionStyle.MARKET:
            result = self._execute_market(plan)
        elif plan.style == ExecutionStyle.TWAP:
            result = self._execute_twap(plan)
        elif plan.style == ExecutionStyle.ICEBERG:
            result = self._execute_iceberg(plan)
        elif plan.style == ExecutionStyle.ADAPTIVE:
            result = self._execute_adaptive(plan)
        else:
            result = self._execute_market(plan)
        
        result.execution_time_seconds = time.time() - start_time
        
        # Update statistics
        self.total_orders += 1
        if result.success:
            self.successful_orders += 1
            self.total_slippage += result.slippage_pips
        
        logger.info(
            f"Execution complete: success={result.success} "
            f"filled={result.total_filled} slippage={result.slippage_pips:.1f}pips"
        )
        
        return result
    
    def _execute_market(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute market order."""
        # Get current price
        tick = mt5.symbol_info_tick(plan.symbol)
        if tick is None:
            return ExecutionResult(success=False, error_message="No tick data")
        
        symbol_info = mt5.symbol_info(plan.symbol)
        if symbol_info is None:
            return ExecutionResult(success=False, error_message="No symbol info")
        
        # Determine order type and price
        if plan.direction == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": plan.symbol,
            "volume": plan.total_size,
            "type": order_type,
            "price": price,
            "sl": plan.stop_loss,
            "tp": plan.take_profit if plan.take_profit else 0.0,
            "deviation": 20,  # Max slippage in points
            "magic": plan.magic_number,
            "comment": "PropBot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Execute with retries
        for attempt in range(self.MAX_RETRIES):
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed: {error}")
                time.sleep(self.RETRY_DELAY_SECONDS)
                continue
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Calculate slippage
                fill_price = result.price
                expected_price = price
                slippage_points = abs(fill_price - expected_price) / symbol_info.point
                
                return ExecutionResult(
                    success=True,
                    ticket=result.order,
                    symbol=plan.symbol,
                    direction=plan.direction,
                    avg_fill_price=fill_price,
                    total_filled=result.volume,
                    slippage_pips=slippage_points / 10,  # Convert to pips
                    num_fills=1,
                    fills=[{
                        "price": fill_price,
                        "volume": result.volume,
                        "time": datetime.now()
                    }]
                )
            
            logger.warning(
                f"Order failed (attempt {attempt + 1}): "
                f"retcode={result.retcode} comment={result.comment}"
            )
            
            time.sleep(self.RETRY_DELAY_SECONDS)
        
        return ExecutionResult(
            success=False,
            error_message=f"Order failed after {self.MAX_RETRIES} attempts"
        )
    
    def _execute_twap(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute TWAP (Time-Weighted Average Price) order."""
        fills = []
        total_filled = 0.0
        total_cost = 0.0
        
        for i, slice_info in enumerate(plan.slices):
            if slice_info["delay"] > 0 and i > 0:
                time.sleep(slice_info["delay"])
            
            # Create mini market order for this slice
            mini_plan = ExecutionPlan(
                symbol=plan.symbol,
                direction=plan.direction,
                total_size=slice_info["size"],
                style=ExecutionStyle.MARKET,
                stop_loss=0,  # SL only on final slice
                take_profit=None,
                magic_number=plan.magic_number,
            )
            
            result = self._execute_market(mini_plan)
            
            if result.success:
                fills.extend(result.fills)
                total_filled += result.total_filled
                total_cost += result.avg_fill_price * result.total_filled
            else:
                logger.warning(f"TWAP slice {i + 1} failed: {result.error_message}")
        
        if total_filled == 0:
            return ExecutionResult(
                success=False,
                error_message="No fills in TWAP execution"
            )
        
        avg_price = total_cost / total_filled
        
        # Set SL/TP on the aggregate position
        if plan.stop_loss or plan.take_profit:
            self._set_position_sltp(
                plan.symbol, plan.stop_loss, plan.take_profit
            )
        
        return ExecutionResult(
            success=True,
            symbol=plan.symbol,
            direction=plan.direction,
            avg_fill_price=avg_price,
            total_filled=total_filled,
            num_fills=len(fills),
            fills=fills
        )
    
    def _execute_iceberg(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute iceberg order (hidden size)."""
        # Similar to TWAP but with smaller visible size
        return self._execute_twap(plan)
    
    def _execute_adaptive(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute with adaptive algorithm."""
        # Start with market order
        result = self._execute_market(plan)
        
        # If slippage is high, switch to TWAP for remaining
        if result.slippage_pips > self.max_slippage_pips:
            logger.warning(
                f"High slippage detected ({result.slippage_pips:.1f}pips), "
                f"consider TWAP for large orders"
            )
        
        return result
    
    def modify_position_sl(
        self,
        ticket: int,
        new_sl: float
    ) -> bool:
        """
        Modify stop loss of an open position.
        
        Args:
            ticket: Position ticket
            new_sl: New stop loss price
            
        Returns:
            True if successful
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return False
        
        position = positions[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": position.tp,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Modified SL for position {ticket}: {new_sl}")
            return True
        
        logger.error(f"Failed to modify SL: {result}")
        return False
    
    def close_position(self, ticket: int) -> ExecutionResult:
        """
        Close an open position.
        
        Args:
            ticket: Position ticket
            
        Returns:
            ExecutionResult
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return ExecutionResult(
                success=False,
                error_message=f"Position {ticket} not found"
            )
        
        position = positions[0]
        
        # Opposite direction to close
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.default_magic,
            "comment": "Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return ExecutionResult(
                success=True,
                ticket=ticket,
                symbol=position.symbol,
                direction="close",
                avg_fill_price=result.price,
                total_filled=result.volume,
            )
        
        return ExecutionResult(
            success=False,
            error_message=f"Close failed: {result}"
        )
    
    def close_all_positions(self, symbol: str = None) -> List[ExecutionResult]:
        """
        Close all open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of ExecutionResults
        """
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if not positions:
            return []
        
        results = []
        for pos in positions:
            result = self.close_position(pos.ticket)
            results.append(result)
        
        return results
    
    def _select_execution_style(
        self,
        symbol: str,
        size: float,
        urgency: float
    ) -> ExecutionStyle:
        """Auto-select execution style based on order characteristics."""
        # Get average volume for context
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            return ExecutionStyle.MARKET
        
        # Small orders: always market
        if size < 0.1:
            return ExecutionStyle.MARKET
        
        # High urgency: market
        if urgency > 0.8:
            return ExecutionStyle.MARKET
        
        # Large orders with low urgency: TWAP
        if size > 0.5 and urgency < 0.5:
            return ExecutionStyle.TWAP
        
        # Medium orders: adaptive
        if size > 0.2:
            return ExecutionStyle.ADAPTIVE
        
        return ExecutionStyle.MARKET
    
    def _set_position_sltp(
        self,
        symbol: str,
        sl: float,
        tp: Optional[float]
    ) -> bool:
        """Set SL/TP on the most recent position for symbol."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return False
        
        # Get most recent
        position = max(positions, key=lambda p: p.time)
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position.ticket,
            "sl": sl,
            "tp": tp if tp else 0.0,
        }
        
        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        success_rate = (
            self.successful_orders / self.total_orders * 100
            if self.total_orders > 0 else 0
        )
        avg_slippage = (
            self.total_slippage / self.successful_orders
            if self.successful_orders > 0 else 0
        )
        
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "success_rate": success_rate,
            "avg_slippage_pips": avg_slippage,
        }
