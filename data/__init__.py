"""
Data Module - Data fetching, processing, and feature engineering.
"""

from .mt5_fetcher import MT5DataFetcher, TIMEFRAMES
from .feature_engineer import FeatureEngineer
from .external import (
    CoinglassClient,
    FundingRate,
    OpenInterest,
    MockCoinglassClient,
    FearGreedClient,
    FearGreedData,
    SentimentAggregator,
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
]
