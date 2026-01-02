"""
External Data Sources - Alternative data APIs.
"""

from .coinglass import CoinglassClient, FundingRate, OpenInterest, MockCoinglassClient
from .alternative_me import FearGreedClient, FearGreedData, SentimentAggregator

from .crypto_alternative import (
    CryptoAlternativeData,
    CryptoSignals,
    MockCryptoAlternativeData,
)

from .forex_alternative import (
    ForexAlternativeData,
    ForexSignals,
    MockForexAlternativeData,
    CENTRAL_BANK_RATES,
)

from .sentiment import (
    SentimentAnalyzer,
    SentimentData,
    SentimentLexicon,
    MockSentimentAnalyzer,
)

from .economic_calendar import (
    EconomicCalendar,
    EconomicEvent,
    EventImpact,
    NewsEventDetector,
    MockEconomicCalendar,
)


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
]
