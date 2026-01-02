"""
Crypto Alternative Data - Comprehensive crypto market data integration.

Aggregates data from multiple sources for crypto trading signals:
- Funding rates (Binance, Coinglass)
- Fear & Greed Index
- Open Interest
- Liquidations
- Exchange flows (if available)

All signals are normalized to [-1, 1] range for easy integration.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .coinglass import CoinglassClient, FundingRate, OpenInterest
from .alternative_me import FearGreedClient

logger = logging.getLogger(__name__)


@dataclass
class CryptoSignals:
    """Aggregated crypto alternative data signals."""
    timestamp: datetime
    symbol: str

    # Raw data
    funding_rate: Optional[float] = None
    fear_greed_index: Optional[int] = None
    open_interest: Optional[float] = None
    open_interest_change_24h: Optional[float] = None
    long_short_ratio: Optional[float] = None

    # Normalized signals (-1 to 1)
    funding_signal: float = 0.0  # Positive = longs paying, bearish pressure
    sentiment_signal: float = 0.0  # Negative = extreme fear (buy signal)
    oi_signal: float = 0.0  # Rising OI with price = trend confirmation

    # Combined signal
    aggregate_signal: float = 0.0
    signal_confidence: float = 0.0

    # Signal components for ensemble
    signals_dict: Dict[str, float] = field(default_factory=dict)


class CryptoAlternativeData:
    """
    Comprehensive crypto alternative data aggregator.

    Fetches data from multiple sources and provides normalized signals
    for trading decision support.

    Usage:
        alt_data = CryptoAlternativeData()

        # Get all signals
        signals = await alt_data.get_all_signals("BTCUSD")

        # Use in trading
        if signals['aggregate_signal'] > 0.5:
            # Bullish alternative data
            pass
    """

    # Symbol mappings for different APIs
    SYMBOL_MAP = {
        "BTCUSD": {"coinglass": "BTC", "binance": "BTCUSDT"},
        "BTCUSD.x": {"coinglass": "BTC", "binance": "BTCUSDT"},
        "ETHUSD": {"coinglass": "ETH", "binance": "ETHUSDT"},
        "ETHUSD.x": {"coinglass": "ETH", "binance": "ETHUSDT"},
        "SOLUSD": {"coinglass": "SOL", "binance": "SOLUSDT"},
        "SOLUSD.x": {"coinglass": "SOL", "binance": "SOLUSDT"},
    }

    def __init__(
        self,
        coinglass_api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300  # 5 minute cache
    ):
        """
        Initialize alternative data client.

        Args:
            coinglass_api_key: API key for Coinglass
            cache_ttl_seconds: Cache TTL for API responses
        """
        self.coinglass = CoinglassClient(api_key=coinglass_api_key)
        self.fear_greed = FearGreedClient()

        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

        # Historical data for signal calibration
        self._funding_history: List[float] = []
        self._oi_history: List[float] = []

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for symbol.

        Positive rate means longs pay shorts (bearish pressure).
        Negative rate means shorts pay longs (bullish pressure).

        Args:
            symbol: Trading symbol

        Returns:
            Funding rate as decimal or None if unavailable
        """
        cache_key = f"funding_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            mapped = self._map_symbol(symbol)
            rate_data = self.coinglass.get_average_funding_rate(mapped.get('coinglass', symbol))

            if rate_data:
                self._set_cached(cache_key, rate_data)
                return rate_data

        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")

        return None

    async def get_fear_greed(self) -> Optional[int]:
        """
        Get current Fear & Greed Index (0-100).

        0-24: Extreme Fear (potential buy signal)
        25-49: Fear
        50-74: Greed
        75-100: Extreme Greed (potential sell signal)

        Returns:
            Fear & Greed index value
        """
        cache_key = "fear_greed"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            data = self.fear_greed.get_current()
            if data:
                self._set_cached(cache_key, data.value)
                return data.value

        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed index: {e}")

        return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        """
        Get current open interest in USD.

        Args:
            symbol: Trading symbol

        Returns:
            Open interest in USD
        """
        cache_key = f"oi_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            mapped = self._map_symbol(symbol)
            oi_data = self.coinglass.get_open_interest(mapped.get('coinglass', symbol))

            if oi_data:
                total_oi = sum(o.open_interest for o in oi_data)
                self._set_cached(cache_key, total_oi)
                return total_oi

        except Exception as e:
            logger.warning(f"Failed to fetch open interest for {symbol}: {e}")

        return None

    async def get_liquidations(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get recent liquidation data.

        Returns:
            Dictionary with long_liquidations, short_liquidations, net_liquidations
        """
        cache_key = f"liqs_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            mapped = self._map_symbol(symbol)
            liq_data = self.coinglass.get_liquidations(mapped.get('coinglass', symbol))

            if liq_data:
                self._set_cached(cache_key, liq_data)
                return liq_data

        except Exception as e:
            logger.warning(f"Failed to fetch liquidations for {symbol}: {e}")

        return None

    async def get_long_short_ratio(self, symbol: str) -> Optional[float]:
        """
        Get long/short ratio from top traders.

        > 1.0: More longs than shorts
        < 1.0: More shorts than longs

        Returns:
            Long/short ratio
        """
        cache_key = f"ls_ratio_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            mapped = self._map_symbol(symbol)
            ratio = self.coinglass.get_long_short_ratio(mapped.get('coinglass', symbol))

            if ratio:
                self._set_cached(cache_key, ratio)
                return ratio

        except Exception as e:
            logger.warning(f"Failed to fetch long/short ratio for {symbol}: {e}")

        return None

    async def get_all_signals(self, symbol: str) -> Dict[str, float]:
        """
        Get all alternative data signals for a symbol.

        All signals are normalized to [-1, 1] range:
        - Positive: Bullish
        - Negative: Bearish

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of normalized signals
        """
        # Fetch all data concurrently
        funding_rate, fear_greed, open_interest, ls_ratio = await asyncio.gather(
            self.get_funding_rate(symbol),
            self.get_fear_greed(),
            self.get_open_interest(symbol),
            self.get_long_short_ratio(symbol),
            return_exceptions=True
        )

        # Handle exceptions
        funding_rate = funding_rate if not isinstance(funding_rate, Exception) else None
        fear_greed = fear_greed if not isinstance(fear_greed, Exception) else None
        open_interest = open_interest if not isinstance(open_interest, Exception) else None
        ls_ratio = ls_ratio if not isinstance(ls_ratio, Exception) else None

        signals = {}
        weights = {}

        # Funding rate signal
        # High positive funding = bearish (longs paying)
        # High negative funding = bullish (shorts paying)
        if funding_rate is not None:
            # Use distribution-based normalization
            funding_signal = self._normalize_funding_rate(funding_rate)
            signals['funding_signal'] = -funding_signal  # Invert: high funding = bearish
            weights['funding_signal'] = 0.25

        # Fear & Greed signal
        # Extreme fear (0-25) = bullish (contrarian buy)
        # Extreme greed (75-100) = bearish (contrarian sell)
        if fear_greed is not None:
            # Normalize to [-1, 1]
            fg_normalized = (fear_greed - 50) / 50  # 0->-1, 50->0, 100->1
            signals['sentiment_signal'] = -fg_normalized  # Contrarian: extreme greed = bearish
            weights['sentiment_signal'] = 0.25

        # Open Interest signal
        # Rising OI in uptrend = bullish confirmation
        # Falling OI = weak trend
        if open_interest is not None:
            oi_signal = self._calculate_oi_signal(open_interest)
            signals['oi_signal'] = oi_signal
            weights['oi_signal'] = 0.2

        # Long/Short ratio signal
        # High ratio (many longs) = contrarian bearish
        # Low ratio (many shorts) = contrarian bullish
        if ls_ratio is not None:
            # Normalize around 1.0
            ls_normalized = (ls_ratio - 1.0) / max(abs(ls_ratio - 1.0), 0.5)
            ls_normalized = np.clip(ls_normalized, -1, 1)
            signals['ls_ratio_signal'] = -ls_normalized  # Contrarian
            weights['ls_ratio_signal'] = 0.15

        # Calculate aggregate signal
        if signals:
            total_weight = sum(weights.values())
            aggregate = sum(
                signals[k] * weights.get(k, 0)
                for k in signals
            ) / total_weight if total_weight > 0 else 0

            signals['aggregate_signal'] = float(np.clip(aggregate, -1, 1))
            signals['signal_confidence'] = total_weight / 0.85  # Max weight is ~0.85
        else:
            signals['aggregate_signal'] = 0.0
            signals['signal_confidence'] = 0.0

        # Add raw values for debugging
        signals['raw_funding_rate'] = funding_rate
        signals['raw_fear_greed'] = fear_greed
        signals['raw_open_interest'] = open_interest
        signals['raw_ls_ratio'] = ls_ratio

        return signals

    def _normalize_funding_rate(self, rate: float) -> float:
        """
        Normalize funding rate using historical distribution.

        Uses percentile-based normalization to avoid hardcoded thresholds.
        """
        # Add to history for calibration
        self._funding_history.append(rate)
        if len(self._funding_history) > 1000:
            self._funding_history = self._funding_history[-1000:]

        if len(self._funding_history) < 10:
            # Not enough history, use simple normalization
            # Typical funding range is -0.001 to 0.001
            return float(np.clip(rate / 0.001, -1, 1))

        # Use percentile ranking
        percentile = np.sum(np.array(self._funding_history) <= rate) / len(self._funding_history)

        # Convert to [-1, 1]
        return float((percentile - 0.5) * 2)

    def _calculate_oi_signal(self, current_oi: float) -> float:
        """
        Calculate OI signal based on historical OI changes.
        """
        self._oi_history.append(current_oi)
        if len(self._oi_history) > 100:
            self._oi_history = self._oi_history[-100:]

        if len(self._oi_history) < 2:
            return 0.0

        # Calculate OI change
        prev_oi = self._oi_history[-2]
        if prev_oi == 0:
            return 0.0

        oi_change_pct = (current_oi - prev_oi) / prev_oi

        # Normalize using historical changes
        if len(self._oi_history) >= 10:
            changes = np.diff(self._oi_history) / np.array(self._oi_history[:-1])
            std_change = np.std(changes) if len(changes) > 1 else 0.1
            if std_change > 0:
                normalized = oi_change_pct / (std_change * 2)
                return float(np.clip(normalized, -1, 1))

        # Simple normalization for small history
        return float(np.clip(oi_change_pct * 10, -1, 1))

    def _map_symbol(self, symbol: str) -> Dict[str, str]:
        """Map trading symbol to API-specific symbols."""
        return self.SYMBOL_MAP.get(symbol, {"coinglass": symbol.replace(".x", "").replace("USD", "")})

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return value
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = (datetime.now(), value)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


class MockCryptoAlternativeData(CryptoAlternativeData):
    """
    Mock implementation for testing and development.

    Generates realistic-looking alternative data signals.
    """

    def __init__(self):
        """Initialize mock client."""
        self._cache = {}
        self.cache_ttl = 300
        self._funding_history = []
        self._oi_history = []

    async def get_funding_rate(self, symbol: str) -> float:
        """Return mock funding rate."""
        # Simulate typical funding rate between -0.01% and 0.03%
        import random
        return random.gauss(0.0001, 0.0005)

    async def get_fear_greed(self) -> int:
        """Return mock Fear & Greed index."""
        import random
        return random.randint(20, 80)

    async def get_open_interest(self, symbol: str) -> float:
        """Return mock open interest."""
        import random
        base_oi = 10_000_000_000  # $10B
        return base_oi * random.uniform(0.9, 1.1)

    async def get_liquidations(self, symbol: str) -> Dict[str, float]:
        """Return mock liquidation data."""
        import random
        long_liqs = random.uniform(10_000_000, 100_000_000)
        short_liqs = random.uniform(10_000_000, 100_000_000)
        return {
            'long_liquidations': long_liqs,
            'short_liquidations': short_liqs,
            'net_liquidations': long_liqs - short_liqs
        }

    async def get_long_short_ratio(self, symbol: str) -> float:
        """Return mock long/short ratio."""
        import random
        return random.uniform(0.8, 1.2)
