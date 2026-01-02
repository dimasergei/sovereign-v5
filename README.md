# Institutional Prop Trading Bot System

An autonomous trading system implementing institutional-grade methodologies for funded proprietary trading accounts.

## 🎯 Philosophy: The Lossless Principle

**Every trading parameter is derived from market data observation - NO hardcoded magic numbers.**

Traditional approaches use fixed values like `RSI_PERIOD = 14` or `OVERBOUGHT = 70`. This system derives ALL parameters dynamically:

| Traditional (❌) | Lossless (✅) |
|-----------------|---------------|
| `RSI_PERIOD = 14` | Period from spectral analysis |
| `OVERBOUGHT = 70` | Threshold from RSI distribution percentiles |
| `ATR_MULT = 2.0` | Multiplier from adverse excursion analysis |
| `RISK = 1%` | Kelly criterion from win rate and edge |

## 🏦 Supported Accounts

### Goat Funded Trader (GFT)
- **Max Drawdown**: 8% trailing (Guardian: 7%)
- **Daily Limit**: None
- **Instruments**: Crypto CFDs
- **Special Rules**: No hedging, 30-day activity requirement

### The5ers
- **Max Drawdown**: 10% (Guardian: 8.5%)
- **Daily Limit**: 5% (Guardian: 4%)
- **Consistency**: No single day > 30% of total profit
- **Instruments**: Forex majors/minors

## 📁 Project Structure

```
prop_bots/
├── core/
│   ├── __init__.py
│   ├── exceptions.py         # Custom exceptions
│   ├── mt5_connector.py      # MT5 connection management
│   ├── risk_engine.py        # Risk management & validation
│   ├── execution.py          # Order execution
│   └── lossless/
│       ├── __init__.py
│       ├── parameter.py      # LosslessParameter class
│       ├── calibrator.py     # Market-derived parameters
│       ├── entropy.py        # Entropy analysis
│       ├── spectral.py       # FFT cycle detection
│       ├── fractal.py        # Fractal dimension
│       └── hurst.py          # Hurst exponent
├── data/
│   ├── __init__.py
│   ├── mt5_fetcher.py        # Data retrieval
│   └── feature_engineer.py   # Technical indicators
├── models/
│   ├── __init__.py
│   ├── base.py               # Base model class
│   ├── ensemble.py           # Meta-learner
│   └── statistical.py        # Mean reversion, regime
├── signals/
│   ├── __init__.py
│   └── generator.py          # Signal generation
├── monitoring/
│   ├── __init__.py
│   └── telegram_bot.py       # Alerts & control
├── bots/
│   ├── __init__.py
│   ├── base_bot.py           # Base bot class
│   ├── gft_bot.py            # GFT crypto bot
│   └── the5ers_bot.py        # The5ers forex bot
├── tests/
│   └── test_risk_engine.py
├── storage/
│   ├── models/               # Saved ML models
│   ├── state/                # Account state persistence
│   └── logs/                 # Log files
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

Edit `bots/gft_bot.py` or `bots/the5ers_bot.py`:

```python
# MT5 Credentials
MT5_LOGIN = 12345678  # Your login
MT5_PASSWORD = "your_password"
MT5_SERVER = "YourBroker-Server"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Telegram (optional)
TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHAT_IDS = [123456789]
```

### 3. Run Tests

```bash
# Run risk engine tests
python -m pytest tests/test_risk_engine.py -v

# Run bot-specific tests
python bots/gft_bot.py --test
python bots/the5ers_bot.py --test
```

### 4. Start Trading

```bash
# GFT Crypto Bot
python bots/gft_bot.py

# The5ers Forex Bot
python bots/the5ers_bot.py

# Debug mode
python bots/gft_bot.py --debug
```

## 🔒 Risk Management

### Guardian Limits

The system implements "guardian limits" - safety buffers below actual prop firm limits:

| Firm | Actual Limit | Guardian Limit | Buffer |
|------|--------------|----------------|--------|
| GFT DD | 8% | 7% | 1% |
| The5ers DD | 10% | 8.5% | 1.5% |
| The5ers Daily | 5% | 4% | 1% |

### Dynamic Risk Scaling

Risk automatically reduces as drawdown increases:

```
DD < 50% of guardian → Full risk
DD 50-100% of guardian → Linear reduction to 25%
DD >= guardian → Trading stops
```

### Kelly Criterion Position Sizing

Position sizes are calculated using:

```
Kelly% = (win_rate × win_ratio - loss_rate) / win_ratio
Actual% = Kelly% × 0.5  (half-Kelly for safety)
```

## 📊 Market Calibration

The MarketCalibrator derives all parameters from price data:

```python
from core.lossless import MarketCalibrator

calibrator = MarketCalibrator()
params = calibrator.calibrate_all(df)

# Derived parameters:
params.fast_period        # From spectral analysis
params.slow_period        # From dominant cycle
params.overbought_threshold  # From RSI distribution
params.stop_loss_atr_multiple  # From adverse excursion
params.hurst_exponent     # Market regime
params.fractal_dimension  # Trend vs mean-reversion
```

### Calibration Methods

| Parameter | Derivation Method |
|-----------|-------------------|
| Fast Period | Shortest significant spectral cycle |
| Slow Period | Dominant spectral cycle |
| ATR Period | Volatility clustering autocorrelation |
| RSI Thresholds | Distribution percentiles before reversals |
| Stop Loss | Maximum adverse excursion (75th percentile) |
| Take Profit | Expected value optimization |
| Risk Fraction | Kelly criterion from trade history |

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Account balance, equity, drawdown |
| `/risk` | Current risk metrics |
| `/performance` | Win rate, profit factor |
| `/trades` | Recent trade history |
| `/pause` | Pause trading |
| `/resume` | Resume trading |
| `/stop CONFIRM` | Emergency close all |
| `/unlock CONFIRM` | Unlock after review |

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_risk_engine.py -v

# With coverage
pytest --cov=core tests/
```

## ⚠️ Important Warnings

1. **NEVER modify risk limits** below prop firm requirements
2. **Guardian limits are safety buffers**, not targets
3. **Paper trade minimum 2 weeks** before live
4. **State persistence is critical** - never lose track of positions/PnL
5. **ML models can fail** - risk limits are the ultimate backstop

## 🔧 Configuration

### Environment Variables (Optional)

```bash
export MT5_LOGIN=12345678
export MT5_PASSWORD=your_password
export TELEGRAM_TOKEN=your_bot_token
```

### Multiple Accounts

Run multiple instances for different accounts:

```bash
# Terminal 1 - GFT Account 1
python bots/gft_bot.py

# Terminal 2 - GFT Account 2
# (Edit ACCOUNT_NAME and MT5_LOGIN first)
python bots/gft_bot.py

# Terminal 3 - The5ers
python bots/the5ers_bot.py
```

## 📈 Performance Targets

| Metric | Target |
|--------|--------|
| Annual Return | > 40% |
| Sharpe Ratio | > 2.0 |
| Max Drawdown | < 6% |
| Win Rate | > 55% |
| Profit Factor | > 1.5 |

## 🛠️ Development

### Adding New Indicators

```python
# In data/feature_engineer.py
def _add_custom_indicator(self, df):
    df['custom'] = your_calculation(df, self.fast_period)
    return df
```

### Adding New Models

```python
# In models/
class MyModel(BaseModel):
    def fit(self, X, y):
        # Training logic
        return self
    
    def predict(self, X):
        return ModelPrediction(
            model_name=self.name,
            direction=prediction,
            magnitude=expected_move,
            confidence=confidence
        )
```

## 📄 License

Private/Proprietary - Not for distribution.

## 🆘 Support

For issues:
1. Check logs in `storage/logs/`
2. Run tests to isolate problems
3. Use `--debug` flag for verbose output
