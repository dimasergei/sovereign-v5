"""
Economic Calendar Module - News and event filtering for trading.

Provides:
- Economic event calendar
- News event detection
- Trading window recommendations based on events

Critical for prop firm compliance: avoid trading around major news events.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """Economic event impact levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4  # Fed decisions, NFP, etc.


@dataclass
class EconomicEvent:
    """Economic calendar event."""
    event_id: str
    title: str
    country: str
    currency: str
    timestamp: datetime

    impact: EventImpact
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None

    # Trading recommendations
    avoid_minutes_before: int = 15
    avoid_minutes_after: int = 15

    @property
    def surprise(self) -> Optional[float]:
        """Calculate surprise (actual vs forecast)."""
        if self.actual is not None and self.forecast is not None:
            if self.forecast != 0:
                return (self.actual - self.forecast) / abs(self.forecast)
        return None

    def is_in_avoid_window(self, current_time: datetime = None) -> bool:
        """Check if current time is in avoid window."""
        if current_time is None:
            current_time = datetime.now()

        start = self.timestamp - timedelta(minutes=self.avoid_minutes_before)
        end = self.timestamp + timedelta(minutes=self.avoid_minutes_after)

        return start <= current_time <= end


# Major economic events with their default avoid windows
MAJOR_EVENTS = {
    # US Events
    "FOMC": {"avoid_before": 60, "avoid_after": 60, "impact": EventImpact.CRITICAL},
    "NFP": {"avoid_before": 30, "avoid_after": 30, "impact": EventImpact.CRITICAL},
    "CPI": {"avoid_before": 15, "avoid_after": 30, "impact": EventImpact.HIGH},
    "GDP": {"avoid_before": 15, "avoid_after": 15, "impact": EventImpact.HIGH},
    "ISM": {"avoid_before": 10, "avoid_after": 15, "impact": EventImpact.MEDIUM},

    # ECB
    "ECB Rate Decision": {"avoid_before": 60, "avoid_after": 60, "impact": EventImpact.CRITICAL},

    # BOE
    "BOE Rate Decision": {"avoid_before": 60, "avoid_after": 60, "impact": EventImpact.CRITICAL},

    # BOJ
    "BOJ Rate Decision": {"avoid_before": 60, "avoid_after": 60, "impact": EventImpact.CRITICAL},

    # Crypto specific
    "Bitcoin Halving": {"avoid_before": 120, "avoid_after": 240, "impact": EventImpact.CRITICAL},
    "ETF Deadline": {"avoid_before": 60, "avoid_after": 120, "impact": EventImpact.HIGH},
}

# Currency to affected pairs mapping
CURRENCY_PAIRS = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
    "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD"],
    "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"],
    "CHF": ["USDCHF", "EURCHF", "GBPCHF"],
    "AUD": ["AUDUSD", "EURAUD", "AUDJPY", "AUDNZD"],
    "CAD": ["USDCAD", "EURCAD", "CADJPY"],
    "NZD": ["NZDUSD", "AUDNZD", "NZDJPY"],
}


class EconomicCalendar:
    """
    Economic calendar for trading event management.

    Provides:
    - Upcoming event listing
    - Trading window recommendations
    - Event impact assessment

    Usage:
        calendar = EconomicCalendar()

        # Check if safe to trade
        can_trade, reason = await calendar.is_safe_to_trade("EURUSD")

        # Get upcoming events
        events = await calendar.get_upcoming_events("USD", hours=24)
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 1800,  # 30 minute cache
        default_avoid_minutes: int = 15
    ):
        """
        Initialize calendar.

        Args:
            cache_ttl_seconds: Cache TTL for event data
            default_avoid_minutes: Default minutes to avoid before/after events
        """
        self.cache_ttl = cache_ttl_seconds
        self.default_avoid = default_avoid_minutes
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

        # Upcoming events store
        self._events: List[EconomicEvent] = []
        self._last_fetch: Optional[datetime] = None

    async def refresh_events(self) -> None:
        """Refresh economic calendar events."""
        try:
            events = await self._fetch_events()
            self._events = events
            self._last_fetch = datetime.now()
            logger.info(f"Refreshed economic calendar: {len(events)} events")
        except Exception as e:
            logger.error(f"Failed to refresh economic calendar: {e}")

    async def get_upcoming_events(
        self,
        currency: str = None,
        hours: int = 24,
        min_impact: EventImpact = EventImpact.MEDIUM
    ) -> List[EconomicEvent]:
        """
        Get upcoming economic events.

        Args:
            currency: Filter by currency (None for all)
            hours: Hours ahead to look
            min_impact: Minimum impact level

        Returns:
            List of upcoming events
        """
        # Refresh if stale
        if (self._last_fetch is None or
            datetime.now() - self._last_fetch > timedelta(seconds=self.cache_ttl)):
            await self.refresh_events()

        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        events = []
        for event in self._events:
            # Time filter
            if event.timestamp < now or event.timestamp > cutoff:
                continue

            # Currency filter
            if currency and event.currency != currency:
                continue

            # Impact filter
            if event.impact.value < min_impact.value:
                continue

            events.append(event)

        return sorted(events, key=lambda e: e.timestamp)

    async def is_safe_to_trade(
        self,
        symbol: str,
        check_time: datetime = None
    ) -> Tuple[bool, str]:
        """
        Check if it's safe to trade based on economic calendar.

        Args:
            symbol: Trading symbol
            check_time: Time to check (default: now)

        Returns:
            Tuple of (is_safe, reason)
        """
        if check_time is None:
            check_time = datetime.now()

        # Refresh if stale
        if (self._last_fetch is None or
            datetime.now() - self._last_fetch > timedelta(seconds=self.cache_ttl)):
            await self.refresh_events()

        # Get currencies affected by this symbol
        affected_currencies = self._get_affected_currencies(symbol)

        for event in self._events:
            # Check if event affects this symbol
            if event.currency not in affected_currencies:
                continue

            # Check if in avoid window
            if event.is_in_avoid_window(check_time):
                return False, f"Avoid trading: {event.title} at {event.timestamp}"

            # Check if approaching high-impact event
            time_to_event = (event.timestamp - check_time).total_seconds() / 60

            if event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
                if 0 < time_to_event < event.avoid_minutes_before:
                    return False, f"Approaching {event.title} in {int(time_to_event)} minutes"

        return True, "No imminent economic events"

    async def get_trading_windows(
        self,
        symbol: str,
        hours: int = 24
    ) -> List[Tuple[datetime, datetime]]:
        """
        Get safe trading windows avoiding economic events.

        Args:
            symbol: Trading symbol
            hours: Hours ahead to analyze

        Returns:
            List of (start, end) tuples for safe trading windows
        """
        now = datetime.now()
        end_time = now + timedelta(hours=hours)

        # Get affected currencies
        affected_currencies = self._get_affected_currencies(symbol)

        # Get relevant events
        events = [e for e in self._events
                  if e.currency in affected_currencies
                  and now <= e.timestamp <= end_time]

        if not events:
            return [(now, end_time)]

        # Sort by time
        events = sorted(events, key=lambda e: e.timestamp)

        # Build windows
        windows = []
        current_start = now

        for event in events:
            avoid_start = event.timestamp - timedelta(minutes=event.avoid_minutes_before)
            avoid_end = event.timestamp + timedelta(minutes=event.avoid_minutes_after)

            if current_start < avoid_start:
                # Window before this event
                windows.append((current_start, avoid_start))

            current_start = max(current_start, avoid_end)

        # Final window after last event
        if current_start < end_time:
            windows.append((current_start, end_time))

        return windows

    def add_custom_event(
        self,
        title: str,
        timestamp: datetime,
        currency: str,
        impact: EventImpact = EventImpact.HIGH,
        avoid_before: int = 15,
        avoid_after: int = 15
    ) -> EconomicEvent:
        """
        Add a custom event to the calendar.

        Args:
            title: Event title
            timestamp: Event time
            currency: Affected currency
            impact: Event impact level
            avoid_before: Minutes to avoid before
            avoid_after: Minutes to avoid after

        Returns:
            Created event
        """
        event = EconomicEvent(
            event_id=f"custom_{len(self._events)}",
            title=title,
            country="Custom",
            currency=currency,
            timestamp=timestamp,
            impact=impact,
            avoid_minutes_before=avoid_before,
            avoid_minutes_after=avoid_after
        )

        self._events.append(event)
        return event

    def _get_affected_currencies(self, symbol: str) -> List[str]:
        """Get currencies affected by a trading symbol."""
        symbol = symbol.upper().replace(".X", "").replace("/", "")

        affected = []

        # Extract base and quote currencies
        if len(symbol) >= 6:
            base = symbol[:3]
            quote = symbol[3:6]
            affected.extend([base, quote])
        else:
            # Crypto symbols
            if "BTC" in symbol:
                affected.append("BTC")
            if "ETH" in symbol:
                affected.append("ETH")
            if "USD" in symbol:
                affected.append("USD")

        return affected

    async def _fetch_events(self) -> List[EconomicEvent]:
        """
        Fetch economic calendar from external source.

        Override for actual API implementation.
        """
        # In production, fetch from:
        # - Forex Factory
        # - Investing.com
        # - TradingView Economic Calendar

        # Return sample events for development
        now = datetime.now()
        sample_events = []

        # Add some sample events
        sample_events.append(EconomicEvent(
            event_id="sample_1",
            title="Sample CPI Release",
            country="US",
            currency="USD",
            timestamp=now + timedelta(hours=4),
            impact=EventImpact.HIGH,
            avoid_minutes_before=15,
            avoid_minutes_after=30
        ))

        return sample_events


class NewsEventDetector:
    """
    Real-time news event detector.

    Monitors for breaking news that could impact markets.
    """

    # Keywords that indicate market-moving news
    CRITICAL_KEYWORDS = {
        'rate hike', 'rate cut', 'inflation', 'recession',
        'war', 'conflict', 'crisis', 'crash', 'collapse',
        'fed', 'fomc', 'ecb', 'boe', 'boj',
        'emergency', 'surprise', 'shock', 'breaking',
    }

    CRYPTO_KEYWORDS = {
        'hack', 'exploit', 'sec', 'regulation', 'ban',
        'etf', 'halving', 'fork', 'whale', 'bankruptcy',
        'exchange', 'binance', 'coinbase', 'ftx',
    }

    def __init__(self):
        """Initialize detector."""
        self._recent_alerts: List[Tuple[datetime, str]] = []

    def analyze_headline(
        self,
        headline: str,
        symbols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a news headline for trading impact.

        Args:
            headline: News headline
            symbols: Relevant trading symbols

        Returns:
            Analysis result with impact assessment
        """
        headline_lower = headline.lower()

        # Check for critical keywords
        critical_matches = [
            kw for kw in self.CRITICAL_KEYWORDS
            if kw in headline_lower
        ]

        crypto_matches = [
            kw for kw in self.CRYPTO_KEYWORDS
            if kw in headline_lower
        ]

        # Determine impact level
        if critical_matches:
            if len(critical_matches) >= 2:
                impact = EventImpact.CRITICAL
            else:
                impact = EventImpact.HIGH
        elif crypto_matches:
            impact = EventImpact.MEDIUM
        else:
            impact = EventImpact.LOW

        # Determine affected symbols
        affected_symbols = []
        if symbols:
            affected_symbols = symbols
        else:
            # Try to detect from headline
            for currency in CURRENCY_PAIRS.keys():
                if currency.lower() in headline_lower:
                    affected_symbols.extend(CURRENCY_PAIRS[currency])

        return {
            'headline': headline,
            'impact': impact,
            'critical_keywords': critical_matches,
            'crypto_keywords': crypto_matches,
            'affected_symbols': list(set(affected_symbols)),
            'should_pause_trading': impact.value >= EventImpact.HIGH.value,
            'timestamp': datetime.now().isoformat(),
        }


class MockEconomicCalendar(EconomicCalendar):
    """
    Mock calendar for testing.
    """

    async def _fetch_events(self) -> List[EconomicEvent]:
        """Return mock events."""
        now = datetime.now()
        return [
            EconomicEvent(
                event_id="mock_1",
                title="Mock FOMC",
                country="US",
                currency="USD",
                timestamp=now + timedelta(hours=2),
                impact=EventImpact.CRITICAL,
                avoid_minutes_before=60,
                avoid_minutes_after=60
            ),
            EconomicEvent(
                event_id="mock_2",
                title="Mock GDP",
                country="US",
                currency="USD",
                timestamp=now + timedelta(hours=6),
                impact=EventImpact.HIGH,
                avoid_minutes_before=15,
                avoid_minutes_after=30
            ),
        ]
