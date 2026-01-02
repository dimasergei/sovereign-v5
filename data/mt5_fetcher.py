"""
MT5 Data Fetcher - Retrieves market data from MetaTrader 5.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from core import MT5Connector, InvalidSymbol


logger = logging.getLogger(__name__)


# Timeframe mapping
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}


class MT5DataFetcher:
    """
    Fetches historical and real-time data from MetaTrader 5.
    
    Usage:
        fetcher = MT5DataFetcher(connector)
        df = fetcher.get_historical_bars("BTCUSD.x", "M5", 500)
    """
    
    def __init__(self, connector: MT5Connector):
        """
        Initialize data fetcher.
        
        Args:
            connector: MT5Connector instance
        """
        self.connector = connector
        self._tick_streams: Dict[str, threading.Thread] = {}
        self._stop_streams = threading.Event()
    
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        start_pos: int = 0
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            count: Number of bars to retrieve
            start_pos: Starting position (0 = current bar)
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        if not self.connector.ensure_connected():
            logger.error("Not connected to MT5")
            return pd.DataFrame()
        
        # Select symbol
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            raise InvalidSymbol(f"Symbol {symbol} not available")
        
        # Get timeframe constant
        tf = TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Fetch data
        rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        })
        
        # Ensure standard columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        return df
    
    def get_historical_bars_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical bars within a date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connector.ensure_connected():
            return pd.DataFrame()
        
        if not mt5.symbol_select(symbol, True):
            raise InvalidSymbol(f"Symbol {symbol} not available")
        
        tf = TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'tick_volume': 'volume'})
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def get_tick_data(
        self,
        symbol: str,
        count: int,
        flags: int = mt5.COPY_TICKS_ALL
    ) -> pd.DataFrame:
        """
        Get recent tick data.
        
        Args:
            symbol: Trading symbol
            count: Number of ticks to retrieve
            flags: Tick type filter
            
        Returns:
            DataFrame with tick data
        """
        if not self.connector.ensure_connected():
            return pd.DataFrame()
        
        if not mt5.symbol_select(symbol, True):
            raise InvalidSymbol(f"Symbol {symbol} not available")
        
        ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, flags)
        
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol specifications.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol info or None
        """
        return self.connector.get_symbol_info(symbol)
    
    def get_available_symbols(self, filter_str: str = None) -> List[str]:
        """
        Get list of available trading symbols.
        
        Args:
            filter_str: Optional filter string
            
        Returns:
            List of symbol names
        """
        if not self.connector.ensure_connected():
            return []
        
        if filter_str:
            symbols = mt5.symbols_get(filter_str)
        else:
            symbols = mt5.symbols_get()
        
        if symbols is None:
            return []
        
        return [s.name for s in symbols if s.visible]
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask prices.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with bid, ask, last, spread
        """
        if not self.connector.ensure_connected():
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time),
        }
    
    def stream_ticks(
        self,
        symbol: str,
        callback: Callable[[Dict], None],
        interval_ms: int = 100
    ):
        """
        Start streaming ticks for a symbol.
        
        Args:
            symbol: Trading symbol
            callback: Function to call with each tick
            interval_ms: Polling interval in milliseconds
        """
        if symbol in self._tick_streams:
            logger.warning(f"Tick stream already active for {symbol}")
            return
        
        def _stream_loop():
            logger.info(f"Starting tick stream for {symbol}")
            last_tick_time = 0
            
            while not self._stop_streams.is_set():
                try:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick and tick.time > last_tick_time:
                        last_tick_time = tick.time
                        tick_data = {
                            'symbol': symbol,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'last': tick.last,
                            'volume': tick.volume,
                            'time': datetime.fromtimestamp(tick.time),
                        }
                        callback(tick_data)
                except Exception as e:
                    logger.error(f"Tick stream error: {e}")
                
                time.sleep(interval_ms / 1000)
            
            logger.info(f"Tick stream stopped for {symbol}")
        
        thread = threading.Thread(target=_stream_loop, daemon=True)
        thread.start()
        self._tick_streams[symbol] = thread
    
    def stop_tick_stream(self, symbol: str = None):
        """
        Stop tick streaming.
        
        Args:
            symbol: Symbol to stop (None = all)
        """
        if symbol:
            if symbol in self._tick_streams:
                self._stop_streams.set()
                self._tick_streams[symbol].join(timeout=2)
                del self._tick_streams[symbol]
        else:
            self._stop_streams.set()
            for thread in self._tick_streams.values():
                thread.join(timeout=2)
            self._tick_streams.clear()
        
        self._stop_streams.clear()
    
    def get_multi_symbol_data(
        self,
        symbols: List[str],
        timeframe: str,
        count: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe string
            count: Number of bars
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        
        for symbol in symbols:
            try:
                df = self.get_historical_bars(symbol, timeframe, count)
                if not df.empty:
                    result[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return result
