"""
Data Module - Data fetching, processing, and feature engineering.
"""

from .mt5_fetcher import MT5DataFetcher, TIMEFRAMES
from .feature_engineer import FeatureEngineer

# External data sources are optional (require aiohttp, etc.)
# They are imported but may be None if dependencies are missing
from .external import (
    CoinglassClient,
    FundingRate,
    OpenInterest,
    MockCoinglassClient,
    FearGreedClient,
    FearGreedData,
    SentimentAggregator,
    is_available as external_is_available,
)


__all__ = [
    'MT5DataFetcher',
    'TIMEFRAMES',
    'FeatureEngineer',
    'CoinglassClient',
    'FundingRate',
    'OpenInterest',
    'MockCoinglassClient',
    'FearGreedClient',
    'FearGreedData',
    'SentimentAggregator',
    'external_is_available',
]
