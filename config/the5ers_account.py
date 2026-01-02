"""
The5ers Account Configuration.

Fill in your actual credentials and settings below.
"""

# Account identifier
ACCOUNT_NAME = "The5ers_Account"
ACCOUNT_SIZE = 5000.0
FIRM = "The5ers"

# MT5 Credentials
MT5_LOGIN = 87654321  # Your MT5 login number
MT5_PASSWORD = "your_password"  # Your MT5 password
MT5_SERVER = "The5ers-Server"  # Your broker server
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Telegram notifications
TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHAT_IDS = [123456789]  # Your Telegram chat ID(s)

# Trading symbols (forex majors/minors)
SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "EURGBP",
    "EURJPY",
]

# Timing
TIMEFRAME = "M5"
SCAN_INTERVAL = 60  # seconds

# Risk Management (guardian limits - more conservative than firm limits)
MAX_DRAWDOWN_PERCENT = 8.5  # Guardian: 8.5% (firm limit: 10%)
DAILY_LOSS_PERCENT = 4.0  # Guardian: 4% (firm limit: 5%)
MAX_POSITIONS = 5
MAX_RISK_PER_TRADE = 1.0  # 1% risk per trade

# Execution settings
DEFAULT_EXECUTION_STYLE = "market"  # market, limit, scaled
SLIPPAGE_TOLERANCE = 5  # points
MAX_RETRIES = 3
