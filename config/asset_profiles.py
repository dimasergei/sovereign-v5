"""
Asset-Specific Trading Profiles.

CRITICAL: Different assets require different parameters.
- Crypto: Higher volatility, larger moves, wider stops
- Indices: Cleaner trends, moderate volatility
- Forex: Tighter ranges, need precision
- Gold: Strong trends but volatile
"""

from typing import Dict, Any


ASSET_PROFILES: Dict[str, Dict[str, Any]] = {
    # ================== CRYPTO ==================
    # Higher volatility, larger moves, wider stops
    'BTCUSD': {
        'atr_multiplier_stop': 2.0,      # Wider stops for crypto volatility
        'atr_multiplier_target': 4.0,    # Larger targets
        'min_pullback_pct': 2.0,         # Need bigger pullback before entry
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 0.8,         # Max 80% of normal size (volatile)
        'ma_proximity_atr': 1.5,         # How close to MA for entry (in ATR)
    },
    'ETHUSD': {
        'atr_multiplier_stop': 2.0,
        'atr_multiplier_target': 4.0,
        'min_pullback_pct': 2.5,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 0.8,
        'ma_proximity_atr': 1.5,
    },
    'SOLUSD': {
        'atr_multiplier_stop': 2.2,
        'atr_multiplier_target': 4.5,
        'min_pullback_pct': 3.0,
        'rsi_oversold': 28,
        'rsi_overbought': 72,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 0.7,
        'ma_proximity_atr': 1.8,
    },

    # ================== INDICES ==================
    # Cleaner trends, moderate volatility
    'NAS100': {
        'atr_multiplier_stop': 1.5,
        'atr_multiplier_target': 3.0,
        'min_pullback_pct': 1.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 1.0,
        'ma_proximity_atr': 1.2,
    },
    'US30': {
        'atr_multiplier_stop': 1.5,
        'atr_multiplier_target': 3.0,
        'min_pullback_pct': 1.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 1.0,
        'ma_proximity_atr': 1.2,
    },
    'SPX500': {
        'atr_multiplier_stop': 1.5,
        'atr_multiplier_target': 3.0,
        'min_pullback_pct': 1.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 1.0,
        'ma_proximity_atr': 1.2,
    },

    # ================== GOLD ==================
    # Strong trends but volatile
    'XAUUSD': {
        'atr_multiplier_stop': 1.8,
        'atr_multiplier_target': 3.5,
        'min_pullback_pct': 1.5,
        'rsi_oversold': 32,
        'rsi_overbought': 68,
        'trend_ma_fast': 21,
        'trend_ma_slow': 55,
        'max_position_pct': 0.9,
        'ma_proximity_atr': 1.3,
    },

    # ================== FOREX ==================
    # Tighter ranges, need precision
    'EURUSD': {
        'atr_multiplier_stop': 1.2,
        'atr_multiplier_target': 2.0,
        'min_pullback_pct': 0.5,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 1.0,
        'ma_proximity_atr': 1.0,
    },
    'GBPUSD': {
        'atr_multiplier_stop': 1.3,
        'atr_multiplier_target': 2.2,
        'min_pullback_pct': 0.6,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 1.0,
        'ma_proximity_atr': 1.0,
    },
    'USDJPY': {
        'atr_multiplier_stop': 1.2,
        'atr_multiplier_target': 2.0,
        'min_pullback_pct': 0.5,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 1.0,
        'ma_proximity_atr': 1.0,
    },
    'USDCHF': {
        'atr_multiplier_stop': 1.2,
        'atr_multiplier_target': 2.0,
        'min_pullback_pct': 0.5,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trend_ma_fast': 20,
        'trend_ma_slow': 50,
        'max_position_pct': 1.0,
        'ma_proximity_atr': 1.0,
    },
}

# Default profile for unknown symbols
DEFAULT_PROFILE: Dict[str, Any] = {
    'atr_multiplier_stop': 1.5,
    'atr_multiplier_target': 3.0,
    'min_pullback_pct': 1.0,
    'rsi_oversold': 33,
    'rsi_overbought': 67,
    'trend_ma_fast': 20,
    'trend_ma_slow': 50,
    'max_position_pct': 1.0,
    'ma_proximity_atr': 1.2,
}


def get_profile(symbol: str) -> Dict[str, Any]:
    """
    Get asset profile, stripping any broker suffix.

    Args:
        symbol: Trading symbol (e.g., "BTCUSD.x", "BTCUSD")

    Returns:
        Asset-specific parameter dictionary
    """
    # Clean symbol - remove common broker suffixes
    clean_symbol = symbol.upper()
    for suffix in ['.X', '.x', '.PRO', '.pro', '.STD', '.std']:
        clean_symbol = clean_symbol.replace(suffix, '')

    return ASSET_PROFILES.get(clean_symbol, DEFAULT_PROFILE)


def get_asset_class(symbol: str) -> str:
    """
    Determine asset class for a symbol.

    Returns: 'crypto', 'index', 'forex', 'commodity', or 'unknown'
    """
    clean_symbol = symbol.upper().replace('.X', '').replace('.x', '')

    crypto = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 'LTCUSD', 'BNBUSD',
              'ADAUSD', 'DOTUSD', 'AVAXUSD', 'MATICUSD']
    indices = ['NAS100', 'US30', 'SPX500', 'US500', 'GER40', 'UK100']
    forex = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
             'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']
    commodities = ['XAUUSD', 'XAGUSD', 'USOIL', 'UKOIL']

    if clean_symbol in crypto:
        return 'crypto'
    elif clean_symbol in indices:
        return 'index'
    elif clean_symbol in forex:
        return 'forex'
    elif clean_symbol in commodities:
        return 'commodity'
    else:
        return 'unknown'
