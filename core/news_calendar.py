# core/news_calendar.py
"""
Economic News Calendar for Trading Blackout Periods.

Fetches and caches high-impact economic events to enforce news blackouts.
GFT: +/- 5 min blackout (profit capped at 1% if violated)
The5ers: +/- 2 min blackout (ALL execution blocked)

Data sources:
- Primary: ForexFactory API
- Fallback: Investing.com calendar
- Cache: 15-minute refresh for intraday updates
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import logging
import json
import os

logger = logging.getLogger(__name__)


class NewsImpact(Enum):
    """News event impact level."""
    HIGH = "high"      # Red folder - NFP, FOMC, CPI
    MEDIUM = "medium"  # Orange folder
    LOW = "low"        # Yellow folder


@dataclass
class NewsEvent:
    """High-impact news event."""
    time: datetime
    name: str
    currency: str
    impact: NewsImpact
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


@dataclass
class NewsCalendarConfig:
    """Configuration for news calendar."""
    cache_duration_minutes: int = 15
    fetch_ahead_hours: int = 24
    high_impact_currencies: List[str] = field(default_factory=lambda: [
        "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"
    ])


class NewsCalendar:
    """
    Fetch and cache high-impact news events for trading blackouts.

    Usage:
        calendar = NewsCalendar()
        calendar.refresh()  # Fetch latest events

        # Check if in blackout for GFT (5 min window)
        if calendar.is_blackout_period("GFT"):
            # Block trade or cap profit

        # Get upcoming events
        events = calendar.get_upcoming_events(minutes=30)
    """

    # High-impact event keywords (red folder events)
    HIGH_IMPACT_KEYWORDS = [
        "Non-Farm Payroll", "NFP", "Nonfarm",
        "FOMC", "Federal Reserve", "Interest Rate Decision",
        "CPI", "Consumer Price Index", "Inflation",
        "GDP", "Gross Domestic Product",
        "ECB", "European Central Bank",
        "BOE", "Bank of England",
        "BOJ", "Bank of Japan",
        "RBA", "Reserve Bank of Australia",
        "Retail Sales",
        "Unemployment Rate", "Employment Change",
        "PMI",  # Purchasing Managers Index
        "Trade Balance",
    ]

    # Static high-impact events (US-focused, update monthly)
    KNOWN_HIGH_IMPACT = [
        # Format: (day_of_week, week_of_month, hour_utc, name, currency)
        # US NFP - First Friday of month at 13:30 UTC (8:30 AM EST)
        ("Friday", 1, 13, 30, "Non-Farm Payrolls", "USD"),
        # FOMC - 8 times per year, Wednesday 19:00 UTC (2:00 PM EST)
        # CPI - Second week of month, 13:30 UTC
    ]

    def __init__(self, config: Optional[NewsCalendarConfig] = None):
        """
        Initialize news calendar.

        Args:
            config: Calendar configuration
        """
        self.config = config or NewsCalendarConfig()
        self.events: List[NewsEvent] = []
        self.last_fetch: Optional[datetime] = None
        self.cache_file = "storage/cache/news_calendar.json"

        # Load cached events
        self._load_cache()

    def refresh(self, force: bool = False) -> bool:
        """
        Refresh news calendar from sources.

        Args:
            force: Force refresh even if cache is fresh

        Returns:
            True if refresh successful
        """
        now = datetime.utcnow()

        # Check cache validity
        if not force and self.last_fetch:
            cache_age = (now - self.last_fetch).total_seconds() / 60
            if cache_age < self.config.cache_duration_minutes:
                logger.debug(f"News cache still fresh ({cache_age:.1f} min old)")
                return True

        try:
            # Try ForexFactory first
            events = self._fetch_forexfactory()

            if not events:
                # Fallback to static schedule
                events = self._get_static_events()

            self.events = events
            self.last_fetch = now
            self._save_cache()

            logger.info(f"News calendar refreshed: {len(events)} high-impact events")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh news calendar: {e}")
            return False

    def is_blackout_period(self, account_type: str = "GFT") -> bool:
        """
        Check if currently in news blackout period.

        Args:
            account_type: "GFT" (5 min) or "THE5ERS" (2 min)

        Returns:
            True if in blackout period
        """
        blackout_minutes = 5 if account_type.upper() == "GFT" else 2
        return self._check_blackout(blackout_minutes)

    def get_blackout_info(self, account_type: str = "GFT") -> Optional[Dict[str, Any]]:
        """
        Get information about current blackout if any.

        Args:
            account_type: "GFT" or "THE5ERS"

        Returns:
            Dict with event info if in blackout, None otherwise
        """
        blackout_minutes = 5 if account_type.upper() == "GFT" else 2
        now = datetime.utcnow()
        blackout_seconds = blackout_minutes * 60

        for event in self.events:
            time_diff = (event.time - now).total_seconds()

            # Check if within blackout window (before or after event)
            if -blackout_seconds <= time_diff <= blackout_seconds:
                return {
                    "event": event.name,
                    "currency": event.currency,
                    "event_time": event.time.isoformat(),
                    "seconds_to_event": time_diff,
                    "blackout_minutes": blackout_minutes,
                }

        return None

    def get_upcoming_events(self, minutes: int = 60) -> List[NewsEvent]:
        """
        Get high-impact events in the next N minutes.

        Args:
            minutes: Lookahead window in minutes

        Returns:
            List of upcoming NewsEvent objects
        """
        now = datetime.utcnow()
        cutoff = now + timedelta(minutes=minutes)

        return [
            event for event in self.events
            if now <= event.time <= cutoff and event.impact == NewsImpact.HIGH
        ]

    def get_next_event(self) -> Optional[NewsEvent]:
        """
        Get the next high-impact event.

        Returns:
            Next NewsEvent or None
        """
        now = datetime.utcnow()

        future_events = [e for e in self.events if e.time > now]
        if future_events:
            return min(future_events, key=lambda e: e.time)

        return None

    def _check_blackout(self, blackout_minutes: int) -> bool:
        """Check if within blackout window of any high-impact event."""
        now = datetime.utcnow()
        blackout_seconds = blackout_minutes * 60

        for event in self.events:
            if event.impact != NewsImpact.HIGH:
                continue

            time_diff = abs((now - event.time).total_seconds())

            if time_diff <= blackout_seconds:
                logger.warning(
                    f"News blackout: {event.name} ({event.currency}) "
                    f"at {event.time.strftime('%H:%M UTC')}"
                )
                return True

        return False

    def _fetch_forexfactory(self) -> List[NewsEvent]:
        """
        Fetch high-impact events from ForexFactory.

        Note: ForexFactory requires web scraping or unofficial API.
        This is a placeholder - implement actual scraping or use
        a paid API service for production.
        """
        # TODO: Implement ForexFactory scraping or API integration
        # For now, return static events
        return self._get_static_events()

    def _get_static_events(self) -> List[NewsEvent]:
        """
        Get static high-impact events based on known schedule.

        This is a fallback when API fetching fails.
        Updates monthly with known scheduled events.
        """
        events = []
        now = datetime.utcnow()

        # Add NFP (first Friday of each month, 13:30 UTC)
        nfp_date = self._get_first_friday(now.year, now.month)
        nfp_time = datetime(nfp_date.year, nfp_date.month, nfp_date.day, 13, 30)

        # If NFP already passed this month, get next month's
        if nfp_time < now - timedelta(hours=1):
            next_month = now.month + 1 if now.month < 12 else 1
            next_year = now.year if now.month < 12 else now.year + 1
            nfp_date = self._get_first_friday(next_year, next_month)
            nfp_time = datetime(nfp_date.year, nfp_date.month, nfp_date.day, 13, 30)

        events.append(NewsEvent(
            time=nfp_time,
            name="Non-Farm Payrolls",
            currency="USD",
            impact=NewsImpact.HIGH
        ))

        # Add CPI (typically mid-month, 13:30 UTC)
        cpi_date = datetime(now.year, now.month, 12, 13, 30)
        if cpi_date < now - timedelta(hours=1):
            next_month = now.month + 1 if now.month < 12 else 1
            next_year = now.year if now.month < 12 else now.year + 1
            cpi_date = datetime(next_year, next_month, 12, 13, 30)

        events.append(NewsEvent(
            time=cpi_date,
            name="CPI m/m",
            currency="USD",
            impact=NewsImpact.HIGH
        ))

        return events

    def _get_first_friday(self, year: int, month: int) -> datetime:
        """Get the first Friday of a given month."""
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7  # Friday = 4
        return first_day + timedelta(days=days_until_friday)

    def _load_cache(self):
        """Load cached events from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)

                self.events = [
                    NewsEvent(
                        time=datetime.fromisoformat(e['time']),
                        name=e['name'],
                        currency=e['currency'],
                        impact=NewsImpact(e['impact'])
                    )
                    for e in data.get('events', [])
                ]

                if data.get('last_fetch'):
                    self.last_fetch = datetime.fromisoformat(data['last_fetch'])

                logger.debug(f"Loaded {len(self.events)} cached news events")

        except Exception as e:
            logger.warning(f"Failed to load news cache: {e}")

    def _save_cache(self):
        """Save events to cache file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            data = {
                'events': [
                    {
                        'time': e.time.isoformat(),
                        'name': e.name,
                        'currency': e.currency,
                        'impact': e.impact.value
                    }
                    for e in self.events
                ],
                'last_fetch': self.last_fetch.isoformat() if self.last_fetch else None
            }

            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save news cache: {e}")


# Singleton instance for global access
_calendar_instance: Optional[NewsCalendar] = None


def get_news_calendar() -> NewsCalendar:
    """Get the global news calendar instance."""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = NewsCalendar()
    return _calendar_instance
