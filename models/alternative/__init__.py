"""
Alternative Data Models - Models using non-price data.
"""

from .funding_rate_model import FundingRateModel, FundingArbitrageStrategy
from .sentiment_model import SentimentModel, SentimentAnalyzer


__all__ = [
    'FundingRateModel',
    'FundingArbitrageStrategy',
    'SentimentModel',
    'SentimentAnalyzer',
]
