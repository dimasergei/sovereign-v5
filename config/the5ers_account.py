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
SYMBOLS = ["XAUUSD", "XAGUSD", "NAS100", "UK100", "SPX500", "EURUSD"]
TIMEFRAME = "M15"
SCAN_INTERVAL = 60

# ============================================================================
# THE5ERS COMPLIANCE LIMITS - DO NOT MODIFY
# ============================================================================
# Actual limits (breach = account closure)
MAX_DAILY_LOSS_PCT = 5.0      # 5% daily loss (from previous day's max)
MAX_TOTAL_DD_PCT = 10.0       # 10% total DD (STATIC from initial balance)

# Guardian limits (stop trading before breach)
GUARDIAN_DAILY_LOSS_PCT = 4.0 # Stop at 4% daily loss
GUARDIAN_TOTAL_DD_PCT = 8.0   # Stop at 8% total DD

# News blackout (stricter than GFT)
NEWS_BLACKOUT_MINUTES = 2     # Block ALL trades +/- 2 min of high-impact news

# Leverage limits (higher than GFT)
LEVERAGE = {"forex": 100, "indices": 25, "metals": 15}

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
MAX_DRAWDOWN_PERCENT = 8.0    # Guardian total DD (same as GUARDIAN_TOTAL_DD_PCT)
DAILY_LOSS_PERCENT = 4.0      # Guardian daily loss (same as GUARDIAN_DAILY_LOSS_PCT)
MAX_POSITIONS = 3
MAX_RISK_PER_TRADE = 1.5
CONSISTENCY_RULE_PERCENT = 30.0
DEFAULT_EXECUTION_STYLE = "MARKET"
SLIPPAGE_TOLERANCE = 0.0005
MAX_RETRIES = 3