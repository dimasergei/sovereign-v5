# GFT Account 2 Configuration
# GFT Instant GOAT Model - $10K Account

MT5_LOGIN = 314329148
MT5_PASSWORD = "4yWrZf#Chq"
MT5_SERVER = "GoatFunded-Server"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
ACCOUNT_NAME = "GFT_10K_2"
ACCOUNT_SIZE = 10000
INITIAL_BALANCE = 10000
FIRM = "GFT"
ACCOUNT_TYPE = "GFT"

# Telegram Alerts
TELEGRAM_TOKEN = "8044940173:AAE6eEz3NjXxWaHkZq3nP903m2LHZAhuYjM"
TELEGRAM_CHAT_IDS = [7898079111]

# Elite Portfolio - Top 6 by Sharpe ratio (projected +63% annual)
SYMBOLS = ["XAUUSD.x", "XAGUSD.x", "NAS100.x", "UK100.x", "SPX500.x", "EURUSD.x"]
TIMEFRAME = "M15"
SCAN_INTERVAL = 60

# ============================================================================
# GFT COMPLIANCE LIMITS - DO NOT MODIFY
# ============================================================================
MAX_DAILY_DD_PCT = 3.0
MAX_TOTAL_DD_PCT = 6.0
MAX_FLOATING_LOSS_PCT = 2.0
MAX_RISK_PER_TRADE_PCT = 2.0
DAILY_PROFIT_CAP = 3000
GUARDIAN_DAILY_DD_PCT = 2.5
GUARDIAN_TOTAL_DD_PCT = 5.0
GUARDIAN_FLOATING_PCT = 1.8
NEWS_BLACKOUT_MINUTES = 5
LEVERAGE = {"forex": 50, "indices": 10, "commodities": 10}

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
MAX_DRAWDOWN_PERCENT = 5.0
DAILY_LOSS_PERCENT = 2.5
MAX_POSITIONS = 3
MAX_RISK_PER_TRADE = 1.5
DEFAULT_EXECUTION_STYLE = "MARKET"
SLIPPAGE_TOLERANCE = 0.001
MAX_RETRIES = 3