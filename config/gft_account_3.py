"""
GFT Account 3 Configuration.

Fill in your actual credentials and settings below.
"""

# Account identifier
ACCOUNT_NAME = "GFT_Account_3"
ACCOUNT_SIZE = 10000.0
FIRM = "GFT"

# MT5 Credentials
MT5_LOGIN = 12345680  # Your MT5 login number
MT5_PASSWORD = "your_password"  # Your MT5 password
MT5_SERVER = "GoatFundedTrader-Server"  # Your broker server
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Telegram notifications
TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHAT_IDS = [123456789]  # Your Telegram chat ID(s)

# Trading symbols
SYMBOLS = [
    "BTCUSD.x",
    "ETHUSD.x",
    "SOLUSD.x",
    "XRPUSD.x",
    "LTCUSD.x",
]

# Timing
TIMEFRAME = "M5"
SCAN_INTERVAL = 60  # seconds

# Risk Management (guardian limits - more conservative than firm limits)
MAX_DRAWDOWN_PERCENT = 5.0  # Guardian: 5% (firm limit: 6%)
DAILY_LOSS_PERCENT = 2.5  # Guardian: 2.5% (firm limit: 3%)
MAX_POSITIONS = 3
MAX_RISK_PER_TRADE = 1.5  # Guardian: 1.5% per trade floating loss (firm limit: 2%)

# Execution settings
DEFAULT_EXECUTION_STYLE = "market"  # market, limit, scaled
SLIPPAGE_TOLERANCE = 10  # points
MAX_RETRIES = 3
