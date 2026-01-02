"""
Temporal Models - Deep learning models for time series.

These models require TensorFlow/Keras which is optional.
"""

import logging

logger = logging.getLogger(__name__)

# Track available models
_AVAILABLE_MODELS = {}

# Try importing LSTM model (requires tensorflow)
try:
    from .lstm_attention import LSTMAttentionModel, create_sequences
    _AVAILABLE_MODELS['lstm'] = True
except (ImportError, NameError) as e:
    logger.debug(f"LSTM model not available: {e}")
    LSTMAttentionModel = None
    create_sequences = None
    _AVAILABLE_MODELS['lstm'] = False

# Try importing Transformer model (requires tensorflow)
try:
    from .transformer import TemporalTransformer
    _AVAILABLE_MODELS['transformer'] = True
except (ImportError, NameError) as e:
    logger.debug(f"Transformer model not available: {e}")
    TemporalTransformer = None
    _AVAILABLE_MODELS['transformer'] = False

# Try importing TCN model (requires tensorflow)
try:
    from .tcn import TCNModel
    _AVAILABLE_MODELS['tcn'] = True
except (ImportError, NameError) as e:
    logger.debug(f"TCN model not available: {e}")
    TCNModel = None
    _AVAILABLE_MODELS['tcn'] = False

# Try importing N-BEATS model (requires tensorflow)
try:
    from .nbeats import NBEATSModel
    _AVAILABLE_MODELS['nbeats'] = True
except (ImportError, NameError) as e:
    logger.debug(f"N-BEATS model not available: {e}")
    NBEATSModel = None
    _AVAILABLE_MODELS['nbeats'] = False


def is_available(model_name: str) -> bool:
    """Check if a model is available."""
    return _AVAILABLE_MODELS.get(model_name, False)


__all__ = [
    'LSTMAttentionModel',
    'TemporalTransformer',
    'TCNModel',
    'NBEATSModel',
    'create_sequences',
    'is_available',
]
