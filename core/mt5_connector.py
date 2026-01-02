"""
MT5 Connector - Production-Grade MetaTrader 5 Connection Management.

Features:
- Thread-safe connection management
- Auto-reconnect with exponential backoff
- Heartbeat monitoring
- Connection state tracking
- Graceful degradation
"""

import MetaTrader5 as mt5
import time
import threading
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

from .exceptions import ConnectionError, ReconnectionFailed


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """MT5 connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class MT5Credentials:
    """MT5 login credentials."""
    login: int
    password: str
    server: str
    path: str  # Path to terminal64.exe
    timeout: int = 60000  # Connection timeout in ms


class MT5Connector:
    """
    Thread-safe MT5 connection manager with auto-recovery.
    
    Usage:
        connector = MT5Connector(credentials)
        connector.connect()
        
        # Auto-reconnect wrapper for any MT5 operation
        account_info = connector.get_account_info()
        
        # Or use context manager for safe operations
        with connector.safe_operation():
            positions = mt5.positions_get()
    """
    
    MAX_RECONNECT_ATTEMPTS = 5
    INITIAL_BACKOFF_SECONDS = 1
    MAX_BACKOFF_SECONDS = 60
    HEARTBEAT_INTERVAL_SECONDS = 30
    
    def __init__(
        self,
        credentials: MT5Credentials,
        on_disconnect_callback: Optional[Callable] = None,
        on_reconnect_callback: Optional[Callable] = None
    ):
        """
        Initialize MT5 connector.
        
        Args:
            credentials: MT5 login credentials
            on_disconnect_callback: Called when connection lost
            on_reconnect_callback: Called when reconnected
        """
        self.credentials = credentials
        self.on_disconnect = on_disconnect_callback
        self.on_reconnect = on_reconnect_callback
        
        self.state = ConnectionState.DISCONNECTED
        self.last_heartbeat: Optional[datetime] = None
        self.reconnect_count = 0
        self.total_disconnects = 0
        self.connected_since: Optional[datetime] = None
        
        self._lock = threading.RLock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        
        logger.info(f"MT5Connector initialized for login {credentials.login}")
    
    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal.
        
        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            self.state = ConnectionState.CONNECTING
            logger.info(f"Connecting to MT5: {self.credentials.server}")
            
            # Initialize MT5
            if not mt5.initialize(
                path=self.credentials.path,
                login=self.credentials.login,
                password=self.credentials.password,
                server=self.credentials.server,
                timeout=self.credentials.timeout
            ):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                self.state = ConnectionState.FAILED
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info after connection")
                self.state = ConnectionState.FAILED
                return False
            
            logger.info(
                f"Connected to MT5: Account #{account_info.login}, "
                f"Balance: ${account_info.balance:.2f}, "
                f"Server: {account_info.server}"
            )
            
            self.state = ConnectionState.CONNECTED
            self.last_heartbeat = datetime.now()
            self.connected_since = datetime.now()
            self.reconnect_count = 0
            
            # Start heartbeat monitoring
            self._start_heartbeat_monitor()
            
            return True
    
    def disconnect(self):
        """Gracefully disconnect from MT5."""
        with self._lock:
            logger.info("Disconnecting from MT5...")
            
            # Stop heartbeat monitor
            self._stop_heartbeat.set()
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=5)
            
            # Shutdown MT5
            mt5.shutdown()
            self.state = ConnectionState.DISCONNECTED
            self.connected_since = None
            
            logger.info("Disconnected from MT5")
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.
        
        Returns:
            True if reconnection successful, False after all attempts exhausted
        """
        with self._lock:
            self.state = ConnectionState.RECONNECTING
            self.total_disconnects += 1
            
            logger.warning(f"Connection lost. Total disconnects: {self.total_disconnects}")
            
            if self.on_disconnect:
                try:
                    self.on_disconnect(self.reconnect_count, self.total_disconnects)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")
            
            for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
                self.reconnect_count = attempt
                
                # Calculate backoff with exponential increase
                backoff = min(
                    self.INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                    self.MAX_BACKOFF_SECONDS
                )
                
                logger.warning(
                    f"Reconnection attempt {attempt}/{self.MAX_RECONNECT_ATTEMPTS} "
                    f"in {backoff}s..."
                )
                time.sleep(backoff)
                
                # Shutdown existing connection first
                try:
                    mt5.shutdown()
                except:
                    pass
                time.sleep(1)
                
                # Try to connect
                if self.connect():
                    logger.info(f"Reconnected successfully on attempt {attempt}")
                    
                    if self.on_reconnect:
                        try:
                            self.on_reconnect(attempt)
                        except Exception as e:
                            logger.error(f"Reconnect callback error: {e}")
                    
                    return True
            
            self.state = ConnectionState.FAILED
            logger.critical(
                f"Failed to reconnect after {self.MAX_RECONNECT_ATTEMPTS} attempts"
            )
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connection is alive.
        
        Returns:
            True if connected and responsive
        """
        if self.state != ConnectionState.CONNECTED:
            return False
        
        try:
            info = mt5.terminal_info()
            if info is None:
                return False
            return info.connected
        except Exception:
            return False
    
    def ensure_connected(self) -> bool:
        """
        Ensure connection is active, reconnect if needed.
        
        Returns:
            True if connected (or reconnected), False if failed
        """
        if self.is_connected():
            return True
        
        logger.warning("Connection check failed, attempting reconnect...")
        return self.reconnect()
    
    @contextmanager
    def safe_operation(self):
        """
        Context manager for safe MT5 operations with auto-reconnect.
        
        Usage:
            with connector.safe_operation():
                positions = mt5.positions_get()
        """
        if not self.ensure_connected():
            raise ReconnectionFailed("Could not establish MT5 connection")
        
        try:
            yield
        except Exception as e:
            logger.error(f"MT5 operation failed: {e}")
            if not self.is_connected():
                self.reconnect()
            raise
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information.
        
        Returns:
            Dictionary with account details or None if failed
        """
        if not self.ensure_connected():
            return None
        
        info = mt5.account_info()
        if info is None:
            return None
        
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level if info.margin > 0 else 0,
            "profit": info.profit,
            "leverage": info.leverage,
            "currency": info.currency,
            "server": info.server,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
        }
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol specifications.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSD.x")
            
        Returns:
            Dictionary with symbol details or None if not found
        """
        if not self.ensure_connected():
            return None
        
        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Symbol {symbol} not available")
            return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            "name": info.name,
            "description": info.description,
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "bid": info.bid,
            "ask": info.ask,
            "trade_mode": info.trade_mode,
        }
    
    def get_positions(self, symbol: str = None) -> list:
        """
        Get open positions.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of position dictionaries
        """
        if not self.ensure_connected():
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "swap": p.swap,
                "time": datetime.fromtimestamp(p.time),
                "magic": p.magic,
                "comment": p.comment,
            }
            for p in positions
        ]
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "state": self.state.value,
            "is_connected": self.is_connected(),
            "connected_since": self.connected_since.isoformat() if self.connected_since else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "total_disconnects": self.total_disconnects,
            "current_reconnect_attempts": self.reconnect_count,
        }
    
    def _start_heartbeat_monitor(self):
        """Start background heartbeat monitoring thread."""
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="MT5Heartbeat"
        )
        self._heartbeat_thread.start()
        logger.debug("Heartbeat monitor started")
    
    def _heartbeat_loop(self):
        """Background loop to monitor connection health."""
        while not self._stop_heartbeat.is_set():
            try:
                if self.state == ConnectionState.CONNECTED:
                    if not self.is_connected():
                        logger.warning("Heartbeat detected connection loss")
                        self.reconnect()
                    else:
                        self.last_heartbeat = datetime.now()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            # Wait for next heartbeat check
            self._stop_heartbeat.wait(self.HEARTBEAT_INTERVAL_SECONDS)
        
        logger.debug("Heartbeat monitor stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
