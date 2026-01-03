# GFT Account 1 Configuration
# GFT Instant GOAT Model - $10K Account

MT5_LOGIN = 314329147
MT5_PASSWORD = "e#sBIdI0sV"
MT5_SERVER = "GoatFunded-Server"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
ACCOUNT_NAME = "GFT_10K_1"
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
# Actual limits (breach = account closure)
MAX_DAILY_DD_PCT = 3.0        # 3% daily DD limit (trailing)
MAX_TOTAL_DD_PCT = 6.0        # 6% total DD limit (trailing from equity HWM)
MAX_FLOATING_LOSS_PCT = 2.0   # 2% per position (HARD BREACH - INSTANT CLOSURE!)
MAX_RISK_PER_TRADE_PCT = 2.0  # 2% max risk per trade
DAILY_PROFIT_CAP = 3000       # $3,000 daily profit cap

# Guardian limits (stop trading before breach)
GUARDIAN_DAILY_DD_PCT = 2.5   # Stop at 2.5% daily DD
GUARDIAN_TOTAL_DD_PCT = 5.0   # Stop at 5% total DD
GUARDIAN_FLOATING_PCT = 1.8   # Close position at 1.8% floating loss

# News blackout
NEWS_BLACKOUT_MINUTES = 5     # Block trades +/- 5 min of high-impact news

# Trade duration
MIN_TRADE_DURATION_SECONDS = 120  # 2 min minimum - profits deducted if closed earlier!

# Leverage limits
LEVERAGE = {"forex": 50, "indices": 10, "commodities": 10}

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
MAX_DRAWDOWN_PERCENT = 5.0    # Guardian total DD (same as GUARDIAN_TOTAL_DD_PCT)
DAILY_LOSS_PERCENT = 2.5      # Guardian daily DD (same as GUARDIAN_DAILY_DD_PCT)
MAX_POSITIONS = 3
MAX_RISK_PER_TRADE = 1.5      # Conservative (limit is 2%)
DEFAULT_EXECUTION_STYLE = "MARKET"
SLIPPAGE_TOLERANCE = 0.001
MAX_RETRIES = 3