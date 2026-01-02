"""
Paper Mode Data Fetcher - Fetches real market data from free APIs for paper trading.

Uses:
- CoinGecko API for crypto (BTC, ETH, SOL, XRP, LTC)
- Free forex APIs for currency pairs
- Falls back to synthetic data if APIs unavailable
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Symbol mappings to API identifiers
CRYPTO_SYMBOLS = {
    'BTCUSD.x': 'bitcoin',
    'BTCUSD': 'bitcoin',
    'BTC-USD': 'bitcoin',
    'ETHUSD.x': 'ethereum',
    'ETHUSD': 'ethereum',
    'ETH-USD': 'ethereum',
    'SOLUSD.x': 'solana',
    'SOLUSD': 'solana',
    'XRPUSD.x': 'ripple',
    'XRPUSD': 'ripple',
    'LTCUSD.x': 'litecoin',
    'LTCUSD': 'litecoin',
}

FOREX_SYMBOLS = {
    'EURUSD': ('EUR', 'USD'),
    'EURUSD.x': ('EUR', 'USD'),
    'GBPUSD': ('GBP', 'USD'),
    'GBPUSD.x': ('GBP', 'USD'),
    'USDJPY': ('USD', 'JPY'),
    'USDJPY.x': ('USD', 'JPY'),
    'AUDUSD': ('AUD', 'USD'),
    'AUDUSD.x': ('AUD', 'USD'),
    'USDCAD': ('USD', 'CAD'),
    'USDCAD.x': ('USD', 'CAD'),
    'EURGBP': ('EUR', 'GBP'),
    'EURGBP.x': ('EUR', 'GBP'),
    'EURJPY': ('EUR', 'JPY'),
    'EURJPY.x': ('EUR', 'JPY'),
}

# Gold symbol mapping
GOLD_SYMBOLS = {
    'XAUUSD': 'gold',
    'XAUUSD.x': 'gold',
    'XAU-USD': 'gold',
}

# Index symbols mapping to Yahoo Finance tickers
INDEX_SYMBOLS = {
    'NAS100': '^IXIC',      # NASDAQ Composite
    'NAS100.x': '^IXIC',
    'US30': '^DJI',         # Dow Jones Industrial Average
    'US30.x': '^DJI',
    'SPX500': '^GSPC',      # S&P 500
    'SPX500.x': '^GSPC',
    'US500': '^GSPC',
    'US500.x': '^GSPC',
}

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
    'M30': 30,
    'H1': 60,
    'H4': 240,
    'D1': 1440,
    'W1': 10080,
}


class PaperDataFetcher:
    """
    Fetches market data from free APIs for paper trading mode.

    Falls back to synthetic data generation when APIs are unavailable.
    """

    def __init__(self):
        """Initialize paper data fetcher."""
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 60  # Cache TTL in seconds

        # Base prices for synthetic data (updated periodically)
        self._base_prices = {
            # Crypto
            'BTCUSD.x': 95000.0,
            'BTCUSD': 95000.0,
            'ETHUSD.x': 3400.0,
            'ETHUSD': 3400.0,
            'SOLUSD.x': 190.0,
            'SOLUSD': 190.0,
            'XRPUSD.x': 2.20,
            'XRPUSD': 2.20,
            'LTCUSD.x': 105.0,
            'LTCUSD': 105.0,
            # Forex
            'EURUSD': 1.0450,
            'EURUSD.x': 1.0450,
            'GBPUSD': 1.2550,
            'GBPUSD.x': 1.2550,
            'USDJPY': 157.50,
            'USDJPY.x': 157.50,
            'AUDUSD': 0.6250,
            'AUDUSD.x': 0.6250,
            'USDCAD': 1.4350,
            'USDCAD.x': 1.4350,
            'EURGBP': 0.8330,
            'EURGBP.x': 0.8330,
            'EURJPY': 164.50,
            'EURJPY.x': 164.50,
            # Gold
            'XAUUSD': 2650.0,
            'XAUUSD.x': 2650.0,
            # Indices
            'NAS100': 21500.0,
            'NAS100.x': 21500.0,
            'US30': 43000.0,
            'US30.x': 43000.0,
            'SPX500': 6000.0,
            'SPX500.x': 6000.0,
            'US500': 6000.0,
            'US500.x': 6000.0,
        }

        # Volatility estimates (daily %)
        self._volatility = {
            # Crypto (high volatility)
            'BTCUSD.x': 0.03,
            'BTCUSD': 0.03,
            'ETHUSD.x': 0.04,
            'ETHUSD': 0.04,
            'SOLUSD.x': 0.05,
            'SOLUSD': 0.05,
            'XRPUSD.x': 0.05,
            'XRPUSD': 0.05,
            'LTCUSD.x': 0.04,
            'LTCUSD': 0.04,
            # Forex (low volatility)
            'EURUSD': 0.005,
            'EURUSD.x': 0.005,
            'GBPUSD': 0.006,
            'GBPUSD.x': 0.006,
            'USDJPY': 0.006,
            'USDJPY.x': 0.006,
            'AUDUSD': 0.007,
            'AUDUSD.x': 0.007,
            'USDCAD': 0.005,
            'USDCAD.x': 0.005,
            'EURGBP': 0.004,
            'EURGBP.x': 0.004,
            'EURJPY': 0.007,
            'EURJPY.x': 0.007,
            # Gold (medium volatility)
            'XAUUSD': 0.012,
            'XAUUSD.x': 0.012,
            # Indices (medium volatility)
            'NAS100': 0.015,
            'NAS100.x': 0.015,
            'US30': 0.010,
            'US30.x': 0.010,
            'SPX500': 0.010,
            'SPX500.x': 0.010,
            'US500': 0.010,
            'US500.x': 0.010,
        }

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
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1)
            count: Number of bars to retrieve
            start_pos: Starting position (0 = current bar)

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{timeframe}_{count}"

        # Check cache
        if cache_key in self._cache:
            cache_age = (datetime.now() - self._cache_time[cache_key]).total_seconds()
            if cache_age < self._cache_ttl:
                logger.debug(f"Using cached data for {symbol}")
                return self._cache[cache_key].copy()

        # Try to fetch real data
        df = self._fetch_real_data(symbol, timeframe, count)

        # Fall back to synthetic data if real data unavailable
        if df.empty:
            logger.info(f"Using synthetic data for {symbol}")
            df = self._generate_synthetic_data(symbol, timeframe, count)

        # Cache the result
        if not df.empty:
            self._cache[cache_key] = df
            self._cache_time[cache_key] = datetime.now()

        return df

    def _fetch_real_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Try to fetch real data from free APIs."""
        # Normalize symbol
        symbol_upper = symbol.upper()
        symbol_clean = symbol.replace('.x', '').upper()

        # Try crypto API (CoinGecko)
        if symbol_upper in CRYPTO_SYMBOLS or symbol in CRYPTO_SYMBOLS:
            return self._fetch_crypto_data(symbol, timeframe, count)

        # Try gold API (CoinGecko has gold data too)
        if symbol_upper in GOLD_SYMBOLS or symbol in GOLD_SYMBOLS or symbol_clean in GOLD_SYMBOLS:
            return self._fetch_gold_data(symbol, timeframe, count)

        # Try index data
        if symbol_upper in INDEX_SYMBOLS or symbol in INDEX_SYMBOLS or symbol_clean in INDEX_SYMBOLS:
            return self._fetch_index_data(symbol, timeframe, count)

        # For forex, use synthetic with current rate
        if symbol_upper in FOREX_SYMBOLS or symbol in FOREX_SYMBOLS or symbol_clean in FOREX_SYMBOLS:
            return self._fetch_forex_data(symbol, timeframe, count)

        # Unknown symbol - will fall back to synthetic
        return pd.DataFrame()

    def _fetch_crypto_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Fetch crypto data from CoinGecko."""
        try:
            coin_id = CRYPTO_SYMBOLS.get(symbol) or CRYPTO_SYMBOLS.get(symbol.upper())
            if not coin_id:
                return pd.DataFrame()

            # Calculate days needed based on timeframe
            minutes = TIMEFRAME_MINUTES.get(timeframe.upper(), 5)
            days_needed = max(1, (count * minutes) // 1440 + 1)
            days_needed = min(days_needed, 90)  # CoinGecko limit

            # CoinGecko market chart endpoint (free, no API key)
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days_needed,
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])

                if not prices:
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop('timestamp', axis=1)

                # Resample to requested timeframe
                df = df.set_index('time')
                resample_rule = self._get_resample_rule(timeframe)

                ohlc = df['price'].resample(resample_rule).ohlc()
                ohlc['volume'] = df['price'].resample(resample_rule).count() * 1000

                ohlc = ohlc.dropna().reset_index()
                ohlc.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

                # Get last 'count' bars
                if len(ohlc) > count:
                    ohlc = ohlc.tail(count)

                logger.info(f"Fetched {len(ohlc)} bars from CoinGecko for {symbol}")
                return ohlc

            elif response.status_code == 429:
                logger.warning("CoinGecko rate limit hit, using synthetic data")
                return pd.DataFrame()
            else:
                logger.warning(f"CoinGecko API error: {response.status_code}")
                return pd.DataFrame()

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch crypto data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing crypto data: {e}")
            return pd.DataFrame()

    def _fetch_forex_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Fetch forex data - uses synthetic with real current rate."""
        try:
            pair = FOREX_SYMBOLS.get(symbol.upper())
            if not pair:
                return pd.DataFrame()

            base, quote = pair

            # Try to get current rate from free API
            try:
                url = f"https://api.exchangerate-api.com/v4/latest/{base}"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    current_rate = data['rates'].get(quote, self._base_prices.get(symbol, 1.0))
                    self._base_prices[symbol] = current_rate
            except:
                pass

            # Generate synthetic data based on current rate
            return self._generate_synthetic_data(symbol, timeframe, count)

        except Exception as e:
            logger.warning(f"Failed to fetch forex data: {e}")
            return pd.DataFrame()

    def _fetch_gold_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Fetch gold data from free API."""
        try:
            # Try to get gold price from metals API or use synthetic
            # Gold is available on some free APIs
            try:
                url = "https://api.coingecko.com/api/v3/simple/price?ids=tether-gold&vs_currencies=usd"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    # Tether Gold tracks gold price
                    data = response.json()
                    if 'tether-gold' in data:
                        current_price = data['tether-gold'].get('usd', 2650.0)
                        self._base_prices[symbol] = current_price
                        self._base_prices[symbol.upper()] = current_price
            except:
                pass

            # Generate synthetic data based on current gold price
            return self._generate_synthetic_data(symbol, timeframe, count)

        except Exception as e:
            logger.warning(f"Failed to fetch gold data: {e}")
            return pd.DataFrame()

    def _fetch_index_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Fetch index data - generates synthetic data based on typical index values."""
        try:
            # Index data requires paid APIs (Yahoo Finance needs yfinance library)
            # Use synthetic data with realistic index prices
            symbol_clean = symbol.replace('.x', '').upper()

            # Update base prices with recent approximate values
            index_prices = {
                'NAS100': 21500.0,  # NASDAQ ~21500
                'US30': 43000.0,    # Dow Jones ~43000
                'SPX500': 6000.0,   # S&P 500 ~6000
                'US500': 6000.0,
            }

            if symbol_clean in index_prices:
                self._base_prices[symbol] = index_prices[symbol_clean]
                self._base_prices[symbol.upper()] = index_prices[symbol_clean]

            logger.debug(f"Using synthetic data for index {symbol}")
            return self._generate_synthetic_data(symbol, timeframe, count)

        except Exception as e:
            logger.warning(f"Failed to fetch index data: {e}")
            return pd.DataFrame()

    def _generate_synthetic_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Generate realistic synthetic OHLCV data."""
        # Get base price and volatility
        symbol_key = symbol.upper() if symbol.upper() in self._base_prices else symbol
        base_price = self._base_prices.get(symbol_key, 100.0)
        daily_vol = self._volatility.get(symbol_key, 0.02)

        # Scale volatility to timeframe
        minutes = TIMEFRAME_MINUTES.get(timeframe.upper(), 5)
        bar_vol = daily_vol * np.sqrt(minutes / 1440)

        # Generate timestamps
        end_time = datetime.now().replace(second=0, microsecond=0)
        end_time = end_time - timedelta(minutes=end_time.minute % minutes)

        timestamps = [end_time - timedelta(minutes=minutes * i) for i in range(count)]
        timestamps.reverse()

        # Generate price path using geometric Brownian motion
        np.random.seed(int(time.time()) % 10000)  # Semi-random but reproducible within second

        returns = np.random.normal(0, bar_vol, count)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from prices
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            # Add intra-bar variation
            high_mult = 1 + abs(np.random.normal(0, bar_vol * 0.5))
            low_mult = 1 - abs(np.random.normal(0, bar_vol * 0.5))

            if i == 0:
                open_price = close * (1 + np.random.normal(0, bar_vol * 0.3))
            else:
                open_price = data[-1]['close']

            high = max(open_price, close) * high_mult
            low = min(open_price, close) * low_mult

            # Random volume
            volume = int(np.random.exponential(1000) * 10)

            data.append({
                'time': ts,
                'open': round(open_price, 5 if base_price < 10 else 2),
                'high': round(high, 5 if base_price < 10 else 2),
                'low': round(low, 5 if base_price < 10 else 2),
                'close': round(close, 5 if base_price < 10 else 2),
                'volume': volume,
            })

        df = pd.DataFrame(data)
        logger.debug(f"Generated {len(df)} synthetic bars for {symbol}")
        return df

    def _get_resample_rule(self, timeframe: str) -> str:
        """Convert timeframe to pandas resample rule."""
        rules = {
            'M1': '1min',
            'M5': '5min',
            'M15': '15min',
            'M30': '30min',
            'H1': '1h',
            'H4': '4h',
            'D1': '1D',
            'W1': '1W',
        }
        return rules.get(timeframe.upper(), '5min')

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current price estimate."""
        df = self.get_historical_bars(symbol, 'M5', 1)
        if df.empty:
            return None

        last = df.iloc[-1]
        spread = last['close'] * 0.0002  # 2 pip spread estimate

        return {
            'bid': last['close'] - spread / 2,
            'ask': last['close'] + spread / 2,
            'last': last['close'],
            'spread': spread,
            'time': last['time'],
        }

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get mock symbol info for paper trading."""
        symbol_upper = symbol.upper()
        symbol_clean = symbol.replace('.x', '').upper()

        # Crypto symbols
        if symbol_upper in CRYPTO_SYMBOLS or symbol in CRYPTO_SYMBOLS:
            return {
                'symbol': symbol,
                'volume_min': 0.001,
                'volume_max': 100.0,
                'volume_step': 0.001,
                'price_digits': 2,
                'contract_size': 1.0,
                'trade_mode': 'full',
            }

        # Gold symbols
        if symbol_upper in GOLD_SYMBOLS or symbol in GOLD_SYMBOLS or symbol_clean in GOLD_SYMBOLS:
            return {
                'symbol': symbol,
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01,
                'price_digits': 2,
                'contract_size': 100.0,  # 100 oz per lot
                'trade_mode': 'full',
            }

        # Index symbols
        if symbol_upper in INDEX_SYMBOLS or symbol in INDEX_SYMBOLS or symbol_clean in INDEX_SYMBOLS:
            return {
                'symbol': symbol,
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01,
                'price_digits': 2,
                'contract_size': 1.0,
                'trade_mode': 'full',
            }

        # Forex symbols
        if symbol_upper in FOREX_SYMBOLS or symbol in FOREX_SYMBOLS or symbol_clean in FOREX_SYMBOLS:
            digits = 3 if 'JPY' in symbol_upper else 5
            return {
                'symbol': symbol,
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01,
                'price_digits': digits,
                'contract_size': 100000.0,
                'trade_mode': 'full',
            }

        # Default
        return {
            'symbol': symbol,
            'volume_min': 0.01,
            'volume_max': 100.0,
            'volume_step': 0.01,
            'price_digits': 5,
            'contract_size': 1.0,
            'trade_mode': 'full',
        }
