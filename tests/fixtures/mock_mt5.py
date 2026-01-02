"""
Mock MT5 Module - Simulates MetaTrader 5 API for testing.

Provides a complete mock implementation of MT5 functions
without requiring actual MT5 installation.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class TradeAction(IntEnum):
    """MT5 trade action types."""
    DEAL_BUY = 0
    DEAL_SELL = 1


class OrderType(IntEnum):
    """MT5 order types."""
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5


class TradeRetcode(IntEnum):
    """MT5 trade return codes."""
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_ERROR = 10006
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_MARKET_CLOSED = 10018


@dataclass
class MockSymbolInfo:
    """Mock MT5 symbol info."""
    name: str
    description: str = ""
    path: str = ""

    # Price info
    bid: float = 50000.0
    ask: float = 50010.0
    last: float = 50005.0

    # Volume constraints
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01

    # Value info
    trade_tick_size: float = 0.01
    trade_tick_value: float = 0.01
    trade_contract_size: float = 1.0

    # Session info
    trade_mode: int = 0  # 0 = full
    trade_stops_level: int = 0
    trade_freeze_level: int = 0

    # Point and digits
    point: float = 0.01
    digits: int = 2

    # Spread
    spread: int = 10
    spread_float: bool = True

    # Swap
    swap_long: float = -0.01
    swap_short: float = 0.01

    # Margin
    margin_initial: float = 100.0
    margin_maintenance: float = 50.0

    visible: bool = True
    select: bool = True


@dataclass
class MockPosition:
    """Mock MT5 position."""
    ticket: int
    symbol: str
    type: int  # 0 = buy, 1 = sell
    volume: float
    price_open: float
    price_current: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    time: datetime = field(default_factory=datetime.now)
    time_msc: int = 0
    magic: int = 0
    comment: str = ""
    identifier: int = 0

    @property
    def type_description(self) -> str:
        return "buy" if self.type == 0 else "sell"


@dataclass
class MockOrder:
    """Mock MT5 order."""
    ticket: int
    symbol: str
    type: int
    volume_initial: float
    volume_current: float
    price_open: float
    price_current: float
    sl: float = 0.0
    tp: float = 0.0
    time_setup: datetime = field(default_factory=datetime.now)
    time_done: Optional[datetime] = None
    state: int = 0  # 0 = started, 1 = placed, 2 = canceled, 3 = partial, 4 = filled
    magic: int = 0
    comment: str = ""


@dataclass
class MockDeal:
    """Mock MT5 deal."""
    ticket: int
    order: int
    symbol: str
    type: int
    volume: float
    price: float
    profit: float
    commission: float = 0.0
    swap: float = 0.0
    fee: float = 0.0
    time: datetime = field(default_factory=datetime.now)
    magic: int = 0
    comment: str = ""


@dataclass
class MockAccountInfo:
    """Mock MT5 account info."""
    login: int = 12345
    server: str = "MockBroker-Demo"
    name: str = "Test Account"
    currency: str = "USD"

    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    margin_free: float = 10000.0
    margin_level: float = 0.0

    profit: float = 0.0
    credit: float = 0.0

    leverage: int = 100
    trade_mode: int = 0
    limit_orders: int = 200


@dataclass
class MockTradeResult:
    """Mock trade request result."""
    retcode: int
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    request_id: int = 0


class MockMT5:
    """
    Mock MetaTrader 5 API implementation.

    Simulates all major MT5 functions for testing.

    Usage:
        mt5 = MockMT5()
        mt5.initialize()

        # Get symbol info
        info = mt5.symbol_info("BTCUSD")

        # Place order
        result = mt5.order_send(request)

        # Get positions
        positions = mt5.positions_get()
    """

    def __init__(self):
        """Initialize mock MT5."""
        self._initialized = False
        self._symbols: Dict[str, MockSymbolInfo] = {}
        self._positions: Dict[int, MockPosition] = {}
        self._orders: Dict[int, MockOrder] = {}
        self._deals: List[MockDeal] = []
        self._account = MockAccountInfo()

        self._next_ticket = 1000
        self._last_error = 0

        # Initialize default symbols
        self._init_default_symbols()

    def _init_default_symbols(self):
        """Initialize default trading symbols."""
        crypto_symbols = [
            ("BTCUSD.x", 50000.0, 50010.0),
            ("ETHUSD.x", 3000.0, 3002.0),
            ("SOLUSD.x", 100.0, 100.05),
        ]

        for name, bid, ask in crypto_symbols:
            self._symbols[name] = MockSymbolInfo(
                name=name,
                bid=bid,
                ask=ask,
                last=(bid + ask) / 2,
                trade_tick_size=0.01,
                trade_tick_value=0.01,
            )

        forex_symbols = [
            ("EURUSD", 1.0850, 1.0852),
            ("GBPUSD", 1.2650, 1.2652),
            ("USDJPY", 150.50, 150.52),
        ]

        for name, bid, ask in forex_symbols:
            self._symbols[name] = MockSymbolInfo(
                name=name,
                bid=bid,
                ask=ask,
                last=(bid + ask) / 2,
                trade_tick_size=0.00001,
                trade_tick_value=10.0,
                point=0.00001,
                digits=5,
            )

    def initialize(self, path: str = None, login: int = None,
                   password: str = None, server: str = None,
                   timeout: int = None, portable: bool = False) -> bool:
        """Initialize MT5 connection."""
        self._initialized = True
        return True

    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        self._initialized = False

    def version(self) -> Tuple[int, int, str]:
        """Get MT5 version."""
        return (5, 0, "Mock MT5")

    def last_error(self) -> Tuple[int, str]:
        """Get last error."""
        return (self._last_error, "Mock error")

    def account_info(self) -> MockAccountInfo:
        """Get account info."""
        # Update equity based on positions
        unrealized_pnl = sum(p.profit for p in self._positions.values())
        self._account.equity = self._account.balance + unrealized_pnl
        return self._account

    def symbol_info(self, symbol: str) -> Optional[MockSymbolInfo]:
        """Get symbol info."""
        return self._symbols.get(symbol)

    def symbol_info_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick for symbol."""
        info = self._symbols.get(symbol)
        if info:
            return {
                'time': datetime.now(),
                'bid': info.bid,
                'ask': info.ask,
                'last': info.last,
                'volume': np.random.uniform(1, 100),
            }
        return None

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Enable/disable symbol."""
        if symbol in self._symbols:
            self._symbols[symbol].select = enable
            return True
        return False

    def symbols_get(self, group: str = None) -> List[MockSymbolInfo]:
        """Get all symbols."""
        return list(self._symbols.values())

    def copy_rates_from(self, symbol: str, timeframe: int,
                        date_from: datetime, count: int) -> Optional[np.ndarray]:
        """Copy historical rates."""
        from .sample_data import generate_ohlcv_data

        info = self._symbols.get(symbol)
        if not info:
            return None

        df = generate_ohlcv_data(
            symbol=symbol,
            n_bars=count,
            start_price=info.bid,
            seed=42
        )

        # Convert to numpy structured array
        dtype = [
            ('time', 'i8'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('tick_volume', 'i8'),
            ('spread', 'i4'),
            ('real_volume', 'i8'),
        ]

        result = np.zeros(len(df), dtype=dtype)
        result['time'] = [int(t.timestamp()) for t in df.index]
        result['open'] = df['open'].values
        result['high'] = df['high'].values
        result['low'] = df['low'].values
        result['close'] = df['close'].values
        result['tick_volume'] = df['volume'].values.astype(int)

        return result

    def copy_rates_range(self, symbol: str, timeframe: int,
                         date_from: datetime, date_to: datetime) -> Optional[np.ndarray]:
        """Copy historical rates in date range."""
        count = int((date_to - date_from).total_seconds() / 3600)  # Hourly bars
        return self.copy_rates_from(symbol, timeframe, date_from, max(1, count))

    def positions_total(self) -> int:
        """Get total positions count."""
        return len(self._positions)

    def positions_get(self, symbol: str = None,
                      group: str = None,
                      ticket: int = None) -> List[MockPosition]:
        """Get open positions."""
        positions = list(self._positions.values())

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        if ticket:
            positions = [p for p in positions if p.ticket == ticket]

        return positions

    def orders_total(self) -> int:
        """Get total pending orders count."""
        return len([o for o in self._orders.values() if o.state < 4])

    def orders_get(self, symbol: str = None,
                   group: str = None,
                   ticket: int = None) -> List[MockOrder]:
        """Get pending orders."""
        orders = [o for o in self._orders.values() if o.state < 4]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if ticket:
            orders = [o for o in orders if o.ticket == ticket]

        return orders

    def order_send(self, request: Dict[str, Any]) -> MockTradeResult:
        """Send trading order."""
        symbol = request.get('symbol', '')
        action = request.get('action', 0)
        order_type = request.get('type', 0)
        volume = request.get('volume', 0.0)
        price = request.get('price', 0.0)
        sl = request.get('sl', 0.0)
        tp = request.get('tp', 0.0)
        magic = request.get('magic', 0)
        comment = request.get('comment', '')

        # Validate symbol
        info = self._symbols.get(symbol)
        if not info:
            return MockTradeResult(retcode=TradeRetcode.TRADE_RETCODE_ERROR)

        # Validate volume
        if volume < info.volume_min or volume > info.volume_max:
            return MockTradeResult(retcode=TradeRetcode.TRADE_RETCODE_INVALID_VOLUME)

        # Get execution price
        if order_type in [OrderType.ORDER_TYPE_BUY]:
            exec_price = info.ask
        elif order_type in [OrderType.ORDER_TYPE_SELL]:
            exec_price = info.bid
        else:
            exec_price = price

        # Create position
        ticket = self._next_ticket
        self._next_ticket += 1

        position = MockPosition(
            ticket=ticket,
            symbol=symbol,
            type=0 if order_type == OrderType.ORDER_TYPE_BUY else 1,
            volume=volume,
            price_open=exec_price,
            price_current=exec_price,
            sl=sl,
            tp=tp,
            magic=magic,
            comment=comment,
        )

        self._positions[ticket] = position

        # Update account margin
        margin_used = volume * exec_price / self._account.leverage
        self._account.margin += margin_used
        self._account.margin_free -= margin_used

        return MockTradeResult(
            retcode=TradeRetcode.TRADE_RETCODE_DONE,
            deal=ticket,
            order=ticket,
            volume=volume,
            price=exec_price,
            bid=info.bid,
            ask=info.ask,
        )

    def order_check(self, request: Dict[str, Any]) -> MockTradeResult:
        """Check if order can be executed."""
        return MockTradeResult(retcode=TradeRetcode.TRADE_RETCODE_DONE)

    def history_deals_get(self, date_from: datetime = None,
                          date_to: datetime = None,
                          group: str = None,
                          ticket: int = None,
                          position: int = None) -> List[MockDeal]:
        """Get historical deals."""
        deals = self._deals

        if date_from:
            deals = [d for d in deals if d.time >= date_from]
        if date_to:
            deals = [d for d in deals if d.time <= date_to]
        if ticket:
            deals = [d for d in deals if d.ticket == ticket]

        return deals

    def history_orders_get(self, date_from: datetime = None,
                           date_to: datetime = None,
                           group: str = None,
                           ticket: int = None,
                           position: int = None) -> List[MockOrder]:
        """Get historical orders."""
        orders = [o for o in self._orders.values() if o.state >= 4]

        if date_from:
            orders = [o for o in orders if o.time_setup >= date_from]
        if date_to:
            orders = [o for o in orders if o.time_setup <= date_to]
        if ticket:
            orders = [o for o in orders if o.ticket == ticket]

        return orders

    # Helper methods for testing

    def set_account_balance(self, balance: float) -> None:
        """Set account balance for testing."""
        self._account.balance = balance
        self._account.equity = balance
        self._account.margin_free = balance

    def set_symbol_price(self, symbol: str, bid: float, ask: float) -> None:
        """Set symbol price for testing."""
        if symbol in self._symbols:
            self._symbols[symbol].bid = bid
            self._symbols[symbol].ask = ask
            self._symbols[symbol].last = (bid + ask) / 2

    def update_position_profit(self, ticket: int, profit: float) -> None:
        """Update position profit for testing."""
        if ticket in self._positions:
            self._positions[ticket].profit = profit

    def close_position(self, ticket: int) -> bool:
        """Close a position for testing."""
        if ticket in self._positions:
            del self._positions[ticket]
            return True
        return False

    def reset(self) -> None:
        """Reset mock to initial state."""
        self._positions.clear()
        self._orders.clear()
        self._deals.clear()
        self._account = MockAccountInfo()
        self._next_ticket = 1000
