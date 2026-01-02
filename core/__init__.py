"""
Core Module - Foundation components for the prop trading system.
"""

from .exceptions import (
    PropBotException,
    ConnectionError,
    ReconnectionFailed,
    RiskViolation,
    InsufficientDataError,
    NotCalibratedError,
    OrderRejected,
    InvalidSymbol,
    AccountLocked,
    ConfigurationError,
    ModelNotTrainedError,
    DataQualityError,
)

from .mt5_connector import (
    MT5Connector,
    MT5Credentials,
    ConnectionState,
)

from .risk_engine import (
    RiskManager,
    FirmRules,
    AccountRiskState,
    FirmType,
    ViolationType,
    create_gft_rules,
    create_the5ers_rules,
)


__all__ = [
    # Exceptions
    'PropBotException',
    'ConnectionError',
    'ReconnectionFailed',
    'RiskViolation',
    'InsufficientDataError',
    'NotCalibratedError',
    'OrderRejected',
    'InvalidSymbol',
    'AccountLocked',
    'ConfigurationError',
    'ModelNotTrainedError',
    'DataQualityError',
    
    # MT5
    'MT5Connector',
    'MT5Credentials',
    'ConnectionState',
    
    # Risk
    'RiskManager',
    'FirmRules',
    'AccountRiskState',
    'FirmType',
    'ViolationType',
    'create_gft_rules',
    'create_the5ers_rules',
]
