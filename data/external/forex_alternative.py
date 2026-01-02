"""
Forex Alternative Data - Alternative data for forex trading.

Provides:
- Interest rate differentials
- COT (Commitment of Traders) positioning
- Risk sentiment indicators (VIX-based)
- Economic surprise indices

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

logger = logging.getLogger(__name__)


@dataclass
class ForexSignals:
    """Aggregated forex alternative data signals."""
    timestamp: datetime
    pair: str

    # Raw data
    rate_differential: Optional[float] = None  # Interest rate diff (base - quote)
    cot_net_position: Optional[float] = None  # Net speculative position
    vix_level: Optional[float] = None  # VIX for risk sentiment

    # Normalized signals (-1 to 1)
    carry_signal: float = 0.0  # Positive = favorable carry
    positioning_signal: float = 0.0  # Extreme positioning = contrarian
    risk_signal: float = 0.0  # High VIX = risk off

    # Combined signal
    aggregate_signal: float = 0.0
    signal_confidence: float = 0.0

    signals_dict: Dict[str, float] = field(default_factory=dict)


# Central bank rates (updated periodically)
# In production, these would be fetched from an API
CENTRAL_BANK_RATES = {
    "USD": 5.25,  # Fed Funds Rate
    "EUR": 4.50,  # ECB Main Refinancing Rate
    "GBP": 5.25,  # Bank of England Rate
    "JPY": -0.10,  # Bank of Japan Rate
    "CHF": 1.75,  # Swiss National Bank Rate
    "AUD": 4.35,  # Reserve Bank of Australia
    "CAD": 5.00,  # Bank of Canada
    "NZD": 5.50,  # Reserve Bank of New Zealand
}

# Currency to country mapping for COT data
CURRENCY_TO_COT = {
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND",
    "JPY": "JAPANESE YEN",
    "CHF": "SWISS FRANC",
    "AUD": "AUSTRALIAN DOLLAR",
    "CAD": "CANADIAN DOLLAR",
    "NZD": "NEW ZEALAND DOLLAR",
}


class ForexAlternativeData:
    """
    Alternative data aggregator for forex trading.

    Provides signals based on:
    - Carry trade dynamics (interest rate differentials)
    - Speculative positioning (COT data)
    - Risk sentiment (VIX)

    Usage:
        alt_data = ForexAlternativeData()
        signals = await alt_data.get_all_signals("EURUSD")
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600  # 1 hour cache (forex data updates slowly)
    ):
        """
        Initialize forex alternative data client.

        Args:
            cache_ttl_seconds: Cache TTL for API responses
        """
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

        # Historical data for calibration
        self._rate_diff_history: Dict[str, List[float]] = {}
        self._cot_history: Dict[str, List[float]] = {}
        self._vix_history: List[float] = []

        # Current central bank rates
        self._rates = CENTRAL_BANK_RATES.copy()

    async def get_rate_differential(self, pair: str) -> Optional[float]:
        """
        Get interest rate differential for currency pair.

        Positive differential means base currency has higher rate.

        Args:
            pair: Currency pair (e.g., "EURUSD")

        Returns:
            Interest rate differential in percentage points
        """
        base, quote = self._parse_pair(pair)

        if base not in self._rates or quote not in self._rates:
            logger.warning(f"Unknown currency in pair {pair}")
            return None

        differential = self._rates[base] - self._rates[quote]
        return differential

    async def get_cot_positioning(self, pair: str) -> Optional[float]:
        """
        Get COT (Commitment of Traders) net positioning.

        Positive = net long speculative position
        Negative = net short speculative position

        Note: COT data is released weekly (Tuesday for previous week)

        Args:
            pair: Currency pair

        Returns:
            Net speculative position normalized to historical range
        """
        cache_key = f"cot_{pair}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # In production, fetch from CFTC API
            # For now, return cached/simulated data
            base, quote = self._parse_pair(pair)

            # Get COT data for base currency
            if base in CURRENCY_TO_COT:
                cot_data = await self._fetch_cot_data(CURRENCY_TO_COT[base])
                if cot_data:
                    self._set_cached(cache_key, cot_data)
                    return cot_data

            return None

        except Exception as e:
            logger.warning(f"Failed to fetch COT data for {pair}: {e}")
            return None

    async def get_risk_sentiment(self) -> Optional[float]:
        """
        Get risk sentiment indicator based on VIX.

        Returns normalized signal:
        - Positive (high VIX): Risk-off (favor safe havens like USD, JPY, CHF)
        - Negative (low VIX): Risk-on (favor risk currencies like AUD, NZD)

        Returns:
            Risk sentiment signal (-1 to 1)
        """
        cache_key = "vix"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            vix = await self._fetch_vix()
            if vix is not None:
                signal = self._normalize_vix(vix)
                self._set_cached(cache_key, signal)
                return signal

            return None

        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")
            return None

    async def get_all_signals(self, pair: str) -> Dict[str, float]:
        """
        Get all alternative data signals for forex pair.

        All signals normalized to [-1, 1]:
        - Positive: Bullish for base currency
        - Negative: Bearish for base currency

        Args:
            pair: Currency pair (e.g., "EURUSD")

        Returns:
            Dictionary of normalized signals
        """
        # Fetch all data
        rate_diff, cot, risk_sentiment = await asyncio.gather(
            self.get_rate_differential(pair),
            self.get_cot_positioning(pair),
            self.get_risk_sentiment(),
            return_exceptions=True
        )

        # Handle exceptions
        rate_diff = rate_diff if not isinstance(rate_diff, Exception) else None
        cot = cot if not isinstance(cot, Exception) else None
        risk_sentiment = risk_sentiment if not isinstance(risk_sentiment, Exception) else None

        signals = {}
        weights = {}

        base, quote = self._parse_pair(pair)

        # Carry signal
        # Positive rate differential = favorable to hold base currency (bullish)
        if rate_diff is not None:
            carry_signal = self._normalize_rate_differential(rate_diff, pair)
            signals['carry_signal'] = carry_signal
            weights['carry_signal'] = 0.3

        # COT positioning signal (contrarian)
        # Extreme long positioning = bearish (crowded trade)
        # Extreme short positioning = bullish (contrarian)
        if cot is not None:
            positioning_signal = -cot  # Contrarian
            signals['positioning_signal'] = positioning_signal
            weights['positioning_signal'] = 0.3

        # Risk sentiment signal
        # Risk-off (high VIX) benefits: USD, JPY, CHF
        # Risk-on (low VIX) benefits: AUD, NZD, risk currencies
        if risk_sentiment is not None:
            risk_signal = self._apply_risk_bias(risk_sentiment, base, quote)
            signals['risk_signal'] = risk_signal
            weights['risk_signal'] = 0.25

        # Calculate aggregate
        if signals:
            total_weight = sum(weights.values())
            aggregate = sum(
                signals[k] * weights.get(k, 0)
                for k in signals
            ) / total_weight if total_weight > 0 else 0

            signals['aggregate_signal'] = float(np.clip(aggregate, -1, 1))
            signals['signal_confidence'] = total_weight / 0.85
        else:
            signals['aggregate_signal'] = 0.0
            signals['signal_confidence'] = 0.0

        # Add raw values
        signals['raw_rate_differential'] = rate_diff
        signals['raw_cot_position'] = cot
        signals['raw_risk_sentiment'] = risk_sentiment

        return signals

    def update_central_bank_rate(self, currency: str, rate: float) -> None:
        """
        Update central bank rate for a currency.

        Args:
            currency: Currency code (e.g., "USD")
            rate: Interest rate in percentage
        """
        self._rates[currency] = rate
        logger.info(f"Updated {currency} rate to {rate}%")

    def _parse_pair(self, pair: str) -> Tuple[str, str]:
        """Parse currency pair into base and quote."""
        pair = pair.upper().replace("/", "")
        if len(pair) == 6:
            return pair[:3], pair[3:]
        raise ValueError(f"Invalid currency pair: {pair}")

    def _normalize_rate_differential(
        self,
        differential: float,
        pair: str
    ) -> float:
        """
        Normalize rate differential using historical data.
        """
        if pair not in self._rate_diff_history:
            self._rate_diff_history[pair] = []

        self._rate_diff_history[pair].append(differential)
        if len(self._rate_diff_history[pair]) > 100:
            self._rate_diff_history[pair] = self._rate_diff_history[pair][-100:]

        history = self._rate_diff_history[pair]

        if len(history) < 5:
            # Simple normalization for small history
            # Typical rate differentials are -5% to +5%
            return float(np.clip(differential / 5.0, -1, 1))

        # Percentile-based normalization
        percentile = np.sum(np.array(history) <= differential) / len(history)
        return float((percentile - 0.5) * 2)

    def _normalize_vix(self, vix: float) -> float:
        """
        Normalize VIX to risk signal.

        Uses adaptive thresholds based on historical VIX.
        """
        self._vix_history.append(vix)
        if len(self._vix_history) > 252:  # ~1 year of daily data
            self._vix_history = self._vix_history[-252:]

        if len(self._vix_history) < 20:
            # Simple normalization
            # Typical VIX range: 10-40
            # Normalize so 12 = -1, 25 = 0, 38 = 1
            normalized = (vix - 25) / 13
            return float(np.clip(normalized, -1, 1))

        # Use percentile ranking
        percentile = np.sum(np.array(self._vix_history) <= vix) / len(self._vix_history)
        return float((percentile - 0.5) * 2)

    def _apply_risk_bias(
        self,
        risk_signal: float,
        base: str,
        quote: str
    ) -> float:
        """
        Apply risk bias based on currency characteristics.

        Risk-off currencies: USD, JPY, CHF
        Risk-on currencies: AUD, NZD, commodity currencies
        """
        RISK_OFF = {"USD", "JPY", "CHF"}
        RISK_ON = {"AUD", "NZD", "CAD"}

        base_is_safe = base in RISK_OFF
        quote_is_safe = quote in RISK_OFF
        base_is_risk = base in RISK_ON
        quote_is_risk = quote in RISK_ON

        # High risk signal (risk-off environment)
        # Benefits safe havens, hurts risk currencies
        if risk_signal > 0:
            if base_is_safe and not quote_is_safe:
                return risk_signal  # Bullish for base (safe haven)
            elif quote_is_safe and not base_is_safe:
                return -risk_signal  # Bearish for base (risk currency)
            elif base_is_risk and not quote_is_risk:
                return -risk_signal  # Bearish for base (risk currency)
            elif quote_is_risk and not base_is_risk:
                return risk_signal  # Bullish for base
        else:  # risk_signal < 0 (risk-on environment)
            if base_is_risk and not quote_is_risk:
                return -risk_signal  # Bullish for base (risk currency benefits)
            elif quote_is_risk and not base_is_risk:
                return risk_signal  # Bearish for base
            elif base_is_safe and not quote_is_safe:
                return risk_signal  # Bearish for base (safe haven hurts)
            elif quote_is_safe and not base_is_safe:
                return -risk_signal  # Bullish for base

        return 0.0  # Neutral for pairs with same risk profile

    async def _fetch_cot_data(self, commodity: str) -> Optional[float]:
        """
        Fetch COT data from CFTC.

        Returns normalized net speculative position.
        """
        # In production, implement actual CFTC API call
        # For now, return simulated data
        return None

    async def _fetch_vix(self) -> Optional[float]:
        """
        Fetch current VIX level.
        """
        # In production, fetch from financial data API
        # For now, return simulated data
        return None

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


class MockForexAlternativeData(ForexAlternativeData):
    """
    Mock implementation for testing.
    """

    def __init__(self):
        """Initialize mock client."""
        super().__init__()

    async def get_cot_positioning(self, pair: str) -> float:
        """Return mock COT positioning."""
        import random
        return random.uniform(-0.8, 0.8)

    async def get_risk_sentiment(self) -> float:
        """Return mock risk sentiment."""
        import random
        vix = random.uniform(12, 35)
        return self._normalize_vix(vix)

    async def _fetch_vix(self) -> float:
        """Return mock VIX."""
        import random
        return random.uniform(12, 35)
