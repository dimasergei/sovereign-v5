# The5ers Account Configuration
# The5ers High Stakes - $5K Account

MT5_LOGIN = 25858994
MT5_PASSWORD = "uylwRMPA~~11"
MT5_SERVER = "FivePercentOnline-Real"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
ACCOUNT_NAME = "The5ers_5K"
ACCOUNT_SIZE = 5000
INITIAL_BALANCE = 5000
FIRM = "THE5ERS"
ACCOUNT_TYPE = "THE5ERS"

# Telegram Alerts
TELEGRAM_TOKEN = "8044940173:AAE6eEz3NjXxWaHkZq3nP903m2LHZAhuYjM"
TELEGRAM_CHAT_IDS = [7898079111]

# Elite Portfolio - Top 6 by Sharpe ratio (projected +63% annual)
# Plus BTCUSD and ETHUSD for 24/7 crypto coverage
SYMBOLS = ["XAUUSD", "XAGUSD", "NAS100", "UK100", "SPX500", "EURUSD"]
CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD"]  # 24/7 trading
ALL_SYMBOLS = SYMBOLS + CRYPTO_SYMBOLS

TIMEFRAME = "M15"
CRYPTO_TIMEFRAME = "H1"  # Use 1H for crypto (less noise)
SCAN_INTERVAL = 60

# ============================================================================
# THE5ERS COMPLIANCE LIMITS - DO NOT MODIFY
# ============================================================================
# Actual limits (breach = account closure)
MAX_DAILY_LOSS_PCT = 4.0      # 4% daily loss (triggers 1-day trading pause)
MAX_TOTAL_DD_PCT = 6.0        # 6% total DD (account termination)
PROFIT_TARGET_PCT = 8.0       # 8% to pass evaluation

# Guardian limits (stop trading before breach)
GUARDIAN_DAILY_LOSS_PCT = 3.5 # Stop at 3.5% daily loss
GUARDIAN_TOTAL_DD_PCT = 5.5   # Stop at 5.5% total DD

# News blackout (stricter than GFT)
NEWS_BLACKOUT_MINUTES = 2     # Block ALL trades +/- 2 min of high-impact news

# Leverage limits by asset class
LEVERAGE = {
    "forex": 30,
    "crypto": 30,
    "indices": 20,
    "commodities": 20
}

# ============================================================================
# CRYPTO-SPECIFIC PARAMETERS
# ============================================================================
CRYPTO_CONFIG = {
    "max_risk_per_trade_pct": 0.5,    # 0.5% risk per trade (half of forex)
    "max_leverage_usage": 0.5,         # Use max 50% of available leverage
    "atr_stop_multiplier": 2.0,        # Wider stops for volatility
    "atr_target_multiplier": 4.0,      # Larger targets
    "min_confluence_score": 0.70,      # 70% of checks must pass
    "min_rr_ratio": 2.0,               # Minimum 2:1 reward:risk
    "regime_filter": True,             # Skip RANGING_VOLATILE regime
    "mtf_required": True,              # Require 4H + 1H alignment
}

# Prohibited strategies (The5ers rules)
PROHIBITED_STRATEGIES = [
    "martingale",
    "grid",
    "arbitrage",
    "hedging_cross_account"
]

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
MAX_DRAWDOWN_PERCENT = 5.5    # Guardian total DD (same as GUARDIAN_TOTAL_DD_PCT)
DAILY_LOSS_PERCENT = 3.5      # Guardian daily loss (same as GUARDIAN_DAILY_LOSS_PCT)
MAX_POSITIONS = 3
MAX_CRYPTO_POSITIONS = 2      # Max 2 crypto positions at once
MAX_RISK_PER_TRADE = 1.0      # 1% max risk per trade
MAX_CRYPTO_RISK_PER_TRADE = 0.5  # 0.5% max risk for crypto
CONSISTENCY_RULE_PERCENT = 30.0  # No single day > 30% of total profits
DEFAULT_EXECUTION_STYLE = "MARKET"
SLIPPAGE_TOLERANCE = 0.0005
MAX_RETRIES = 3