# config/trading_params.py
"""
Renaissance-style trading parameters for high-frequency multi-alpha trading.

Key principle: Generate MORE trades with slight edge.
Many small bets compound to large returns.

Target: 150-200 trades/year per symbol with 52% win rate and 2:1 R:R
"""

PARAMS = {
    # Risk per trade: 0.5% base (was 1.0%)
    # Allows more positions simultaneously
    'base_risk_pct': 0.5,
    'max_risk_pct': 0.8,
    'min_risk_pct': 0.2,

    # Position limits
    'max_positions': 4,
    'max_same_direction': 3,  # Max 3 longs or 3 shorts at once

    # Signal thresholds - LOWER to generate more trades
    'min_confidence': 0.40,  # Was 0.50

    # Mean reversion triggers - LOWER to trade more
    'mr_zscore_entry': 1.5,  # Was 2.0
    'mr_rsi_oversold': 35,   # Was 30
    'mr_rsi_overbought': 65, # Was 70

    # Trend thresholds - LOWER
    'trend_roc_threshold': 0.5,  # Was 1.0%
    'trend_ema_tolerance': 0.02,  # Within 2% of EMA is OK

    # Breakout parameters
    'breakout_lookback': 20,
    'vol_compression_ratio': 0.8,

    # Lead-lag parameters
    'lead_lag_threshold': 1.0,  # 1% divergence triggers signal

    # R:R targets
    'min_rr_ratio': 2.0,
    'target_rr_ratio': 2.5,
    'stop_atr_multiplier': 1.5,  # Default stop distance

    # Guardian and risk management
    'guardian_dd_pct': 4.0,     # Stop new trades at 4% DD (2% buffer from 6% limit)
    'guardian_buffer': 1.5,     # Reduce size when within 1.5% of guardian
    'max_daily_trades': 10,     # Prevent overtrading

    # Strategy weights
    'weight_trend': 0.30,
    'weight_mean_reversion': 0.35,
    'weight_breakout': 0.15,
    'weight_lead_lag': 0.20,
}


# Asset-specific overrides
ASSET_OVERRIDES = {
    'BTCUSD': {
        'base_risk_pct': 0.4,  # Slightly smaller for volatile crypto
        'mr_zscore_entry': 1.8,  # Crypto needs bigger move
        'stop_atr_multiplier': 2.0,  # Wider stops
    },
    'XAUUSD': {
        'base_risk_pct': 0.5,
        'mr_zscore_entry': 1.5,
        'stop_atr_multiplier': 1.8,
    },
    'NAS100': {
        'base_risk_pct': 0.5,
        'mr_zscore_entry': 1.4,  # Tech is fast
        'stop_atr_multiplier': 1.5,
    },
    'EURUSD': {
        'base_risk_pct': 0.6,  # Can size up for less volatile forex
        'mr_zscore_entry': 1.3,
        'stop_atr_multiplier': 1.2,
    },
}


def get_params(symbol: str = None) -> dict:
    """Get trading parameters, with optional asset-specific overrides."""
    params = PARAMS.copy()

    if symbol:
        clean_symbol = symbol.upper().replace('.X', '').replace('.x', '')
        if clean_symbol in ASSET_OVERRIDES:
            params.update(ASSET_OVERRIDES[clean_symbol])

    return params
