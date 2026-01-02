"""
Core Exceptions for Prop Trading System.

Custom exceptions for handling various error conditions.
"""


class PropBotException(Exception):
    """Base exception for all prop bot errors."""
    pass


class ConnectionError(PropBotException):
    """MT5 connection failed."""
    pass


class ReconnectionFailed(PropBotException):
    """Failed to reconnect after multiple attempts."""
    pass


class RiskViolation(PropBotException):
    """Trade would violate risk limits."""
    
    def __init__(self, message: str, violation_type: str = None):
        super().__init__(message)
        self.violation_type = violation_type


class InsufficientDataError(PropBotException):
    """Not enough data for calibration or analysis."""
    pass


class NotCalibratedError(PropBotException):
    """Parameter not yet calibrated from market data."""
    pass


class OrderRejected(PropBotException):
    """Order was rejected by broker."""
    
    def __init__(self, message: str, retcode: int = None):
        super().__init__(message)
        self.retcode = retcode


class InvalidSymbol(PropBotException):
    """Symbol not found or not tradeable."""
    pass


class AccountLocked(PropBotException):
    """Account is locked due to risk violation."""
    pass


class ConfigurationError(PropBotException):
    """Invalid configuration."""
    pass


class ModelNotTrainedError(PropBotException):
    """ML model not yet trained."""
    pass


class DataQualityError(PropBotException):
    """Data quality check failed."""
    pass
