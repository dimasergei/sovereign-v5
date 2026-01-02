"""
Models Module - Prediction models for the trading system.
"""

from .base import BaseModel, ModelPrediction
from .ensemble import EnsembleMetaLearner, EnsemblePrediction
from .statistical import MeanReversionModel, RegimeDetector, LeadLagDetector
from .temporal import (
    LSTMAttentionModel,
    TemporalTransformer,
    TCNModel,
    NBEATSModel,
    create_sequences,
)
from .reinforcement import (
    PPOAgent,
    A2CAgent,
    TradingEnvironment,
    TradingConfig,
    MultiAssetEnvironment,
)
from .regime import (
    HMMRegimeModel,
    MarketRegime,
    RegimeAwareStrategy,
    GARCHModel,
    VolatilityRegimeFilter,
)
from .alternative import (
    FundingRateModel,
    FundingArbitrageStrategy,
    SentimentModel,
    SentimentAnalyzer,
)


__all__ = [
    'BaseModel',
    'ModelPrediction',
    'EnsembleMetaLearner',
    'EnsemblePrediction',
    'MeanReversionModel',
    'RegimeDetector',
    'LeadLagDetector',
    # Temporal models
    'LSTMAttentionModel',
    'TemporalTransformer',
    'TCNModel',
    'NBEATSModel',
    'create_sequences',
    # Reinforcement learning
    'PPOAgent',
    'A2CAgent',
    'TradingEnvironment',
    'TradingConfig',
    'MultiAssetEnvironment',
    # Regime models
    'HMMRegimeModel',
    'MarketRegime',
    'RegimeAwareStrategy',
    'GARCHModel',
    'VolatilityRegimeFilter',
    # Alternative data models
    'FundingRateModel',
    'FundingArbitrageStrategy',
    'SentimentModel',
    'SentimentAnalyzer',
]
