"""
Models Module - Prediction models for the trading system.

Some models require optional dependencies (tensorflow, arch, etc.)
"""

import logging

logger = logging.getLogger(__name__)

# Core models (always available)
from .base import BaseModel, ModelPrediction
from .ensemble import EnsembleMetaLearner, EnsemblePrediction
from .statistical_models import MeanReversionModel, RegimeDetector, LeadLagDetector

# Temporal models (require tensorflow - may be None)
from .temporal import (
    LSTMAttentionModel,
    TemporalTransformer,
    TCNModel,
    NBEATSModel,
    create_sequences,
)

# Reinforcement learning models (require optional deps)
try:
    from .reinforcement import (
        PPOAgent,
        A2CAgent,
        TradingEnvironment,
        TradingConfig,
        MultiAssetEnvironment,
    )
except (ImportError, NameError) as e:
    logger.debug(f"Reinforcement learning models not available: {e}")
    PPOAgent = None
    A2CAgent = None
    TradingEnvironment = None
    TradingConfig = None
    MultiAssetEnvironment = None

# Regime models (may require arch package)
try:
    from .regime import (
        HMMRegimeModel,
        MarketRegime,
        RegimeAwareStrategy,
        GARCHModel,
        VolatilityRegimeFilter,
    )
except (ImportError, NameError) as e:
    logger.debug(f"Regime models not fully available: {e}")
    HMMRegimeModel = None
    MarketRegime = None
    RegimeAwareStrategy = None
    GARCHModel = None
    VolatilityRegimeFilter = None

# Alternative data models
try:
    from .alternative import (
        FundingRateModel,
        FundingArbitrageStrategy,
        SentimentModel,
        SentimentAnalyzer,
    )
except (ImportError, NameError) as e:
    logger.debug(f"Alternative data models not available: {e}")
    FundingRateModel = None
    FundingArbitrageStrategy = None
    SentimentModel = None
    SentimentAnalyzer = None


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
