"""
External Data Sources - Alternative data APIs.

These modules require optional dependencies (aiohttp, etc.)
They are lazily imported to avoid breaking core functionality.
"""

import logging

logger = logging.getLogger(__name__)

# Track available modules
_AVAILABLE_MODULES = {}

# Try importing coinglass (requires aiohttp)
try:
    from .coinglass import CoinglassClient, FundingRate, OpenInterest, MockCoinglassClient
    _AVAILABLE_MODULES['coinglass'] = True
except ImportError as e:
    logger.debug(f"Coinglass module not available: {e}")
    CoinglassClient = None
    FundingRate = None
    OpenInterest = None
    MockCoinglassClient = None
    _AVAILABLE_MODULES['coinglass'] = False

# Try importing alternative_me (requires aiohttp)
try:
    from .alternative_me import FearGreedClient, FearGreedData, SentimentAggregator
    _AVAILABLE_MODULES['alternative_me'] = True
except ImportError as e:
    logger.debug(f"Alternative.me module not available: {e}")
    FearGreedClient = None
    FearGreedData = None
    SentimentAggregator = None
    _AVAILABLE_MODULES['alternative_me'] = False

# Try importing crypto_alternative (requires aiohttp)
try:
    from .crypto_alternative import (
        CryptoAlternativeData,
        CryptoSignals,
        MockCryptoAlternativeData,
    )
    _AVAILABLE_MODULES['crypto_alternative'] = True
except ImportError as e:
    logger.debug(f"Crypto alternative module not available: {e}")
    CryptoAlternativeData = None
    CryptoSignals = None
    MockCryptoAlternativeData = None
    _AVAILABLE_MODULES['crypto_alternative'] = False

# Try importing forex_alternative (requires aiohttp)
try:
    from .forex_alternative import (
        ForexAlternativeData,
        ForexSignals,
        MockForexAlternativeData,
        CENTRAL_BANK_RATES,
    )
    _AVAILABLE_MODULES['forex_alternative'] = True
except ImportError as e:
    logger.debug(f"Forex alternative module not available: {e}")
    ForexAlternativeData = None
    ForexSignals = None
    MockForexAlternativeData = None
    CENTRAL_BANK_RATES = {}
    _AVAILABLE_MODULES['forex_alternative'] = False

# Try importing sentiment (requires aiohttp and optional NLP libs)
try:
    from .sentiment import (
        SentimentAnalyzer,
        SentimentData,
        SentimentLexicon,
        MockSentimentAnalyzer,
    )
    _AVAILABLE_MODULES['sentiment'] = True
except ImportError as e:
    logger.debug(f"Sentiment module not available: {e}")
    SentimentAnalyzer = None
    SentimentData = None
    SentimentLexicon = None
    MockSentimentAnalyzer = None
    _AVAILABLE_MODULES['sentiment'] = False

# Try importing economic_calendar
try:
    from .economic_calendar import (
        EconomicCalendar,
        EconomicEvent,
        EventImpact,
        NewsEventDetector,
        MockEconomicCalendar,
    )
    _AVAILABLE_MODULES['economic_calendar'] = True
except ImportError as e:
    logger.debug(f"Economic calendar module not available: {e}")
    EconomicCalendar = None
    EconomicEvent = None
    EventImpact = None
    NewsEventDetector = None
    MockEconomicCalendar = None
    _AVAILABLE_MODULES['economic_calendar'] = False


def is_available(module_name: str) -> bool:
    """Check if a module is available."""
    return _AVAILABLE_MODULES.get(module_name, False)


__all__ = [
    # Coinglass
    'CoinglassClient',
    'FundingRate',
    'OpenInterest',
    'MockCoinglassClient',
    # Fear & Greed
    'FearGreedClient',
    'FearGreedData',
    'SentimentAggregator',
    # Crypto Alternative Data
    'CryptoAlternativeData',
    'CryptoSignals',
    'MockCryptoAlternativeData',
    # Forex Alternative Data
    'ForexAlternativeData',
    'ForexSignals',
    'MockForexAlternativeData',
    'CENTRAL_BANK_RATES',
    # Sentiment Analysis
    'SentimentAnalyzer',
    'SentimentData',
    'SentimentLexicon',
    'MockSentimentAnalyzer',
    # Economic Calendar
    'EconomicCalendar',
    'EconomicEvent',
    'EventImpact',
    'NewsEventDetector',
    'MockEconomicCalendar',
    # Utility
    'is_available',
]
