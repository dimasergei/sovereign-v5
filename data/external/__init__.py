"""
External Data Sources - Alternative data APIs.
"""

from .coinglass import CoinglassClient, FundingRate, OpenInterest, MockCoinglassClient
from .alternative_me import FearGreedClient, FearGreedData, SentimentAggregator


__all__ = [
    'CoinglassClient',
    'FundingRate',
    'OpenInterest',
    'MockCoinglassClient',
    'FearGreedClient',
    'FearGreedData',
    'SentimentAggregator',
]
