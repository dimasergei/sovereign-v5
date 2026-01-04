# SOVEREIGN V5 - Multi-Account Prop Firm Trading System

## System Blueprint

**Version:** 5.0
**Author:** Sovereign Trading Systems
**Last Updated:** January 2026

A production-grade autonomous trading system for managing multiple prop firm accounts with institutional-grade risk management, compliance enforcement, and ML-driven signal generation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Prop Firm Compliance](#prop-firm-compliance)
4. [Trading Strategy](#trading-strategy)
5. [Paper Trading](#paper-trading)
6. [Live Deployment](#live-deployment)
7. [Module Reference](#module-reference)
8. [Configuration](#configuration)
9. [Telegram Integration](#telegram-integration)
10. [Troubleshooting](#troubleshooting)

---

## System Overview

### What It Does

Sovereign V5 autonomously trades a portfolio of 6 elite symbols across multiple prop firm accounts, plus cryptocurrency for 24/7 coverage:

| Symbol | Type | Why Selected |
|--------|------|--------------|
| XAUUSD | Gold | High volatility, strong trends |
| XAGUSD | Silver | Correlated hedge, mean-reversion |
| NAS100 | Index | Tech momentum, clean technicals |
| UK100 | Index | European session coverage |
| SPX500 | Index | US session, high liquidity |
| EURUSD | Forex | Most liquid pair, tight spreads |
| BTCUSD | Crypto | 24/7 trading, The5ers only |
| ETHUSD | Crypto | 24/7 trading, The5ers only |

### Account Structure

| Account | Firm | Size | Daily DD | Total DD | Special Rules |
|---------|------|------|----------|----------|---------------|
| GFT_1 | GFT Instant GOAT | $10,000 | 3% | 6% trailing | 2% floating loss = instant closure |
| GFT_2 | GFT Instant GOAT | $10,000 | 3% | 6% trailing | 2-min minimum trade duration |
| GFT_3 | GFT Instant GOAT | $10,000 | 3% | 6% trailing | 15% consistency rule |
| THE5ERS_1 | The5ers High Stakes | $5,000 | 5% | 10% static | Crypto 24/7 trading enabled |

**Total Capital Under Management:** $35,000

### Projected Performance

Based on backtesting the Elite 6 portfolio:
- **Annual Return:** +67.37%
- **Sharpe Ratio:** 2.14
- **Max Drawdown:** 4.8%
- **Win Rate:** 58%

---

## Architecture

```
sovereign-v5/
├── bots/                      # Trading bot implementations
│   ├── base_bot.py           # Base class for all bots
│   ├── gft_bot.py            # GFT live trading bot
│   └── the5ers_bot.py        # The5ers live trading bot
│
├── config/                    # Account configurations
│   ├── gft_account_1.py      # GFT Account 1 settings
│   ├── gft_account_2.py      # GFT Account 2 settings
│   ├── gft_account_3.py      # GFT Account 3 settings
│   └── the5ers_account.py    # The5ers settings
│
├── core/                      # Core trading infrastructure
│   ├── mt5_connector.py      # MT5 connection management
│   ├── execution.py          # Smart order execution (MARKET/TWAP/ICEBERG)
│   ├── paper_executor.py     # Paper trading simulation
│   ├── position_sizer.py     # Kelly-based position sizing
│   ├── news_calendar.py      # Economic event tracking
│   ├── risk_engine.py        # Risk validation engine
│   ├── compliance/
│   │   ├── gft_compliance.py     # GFT rule enforcement
│   │   └── the5ers_compliance.py # The5ers rule enforcement
│   └── lossless/             # Market-derived parameters
│       ├── calibrator.py     # Parameter derivation
│       ├── spectral.py       # FFT cycle detection
│       ├── hurst.py          # Trend/mean-reversion detection
│       └── entropy.py        # Market randomness analysis
│
├── data/                      # Data management
│   ├── mt5_fetcher.py        # Real-time MT5 data
│   ├── paper_fetcher.py      # Paper trading data (API + synthetic)
│   ├── feature_engineer.py   # Technical indicator calculation
│   └── external/             # External data sources
│       ├── economic_calendar.py
│       ├── sentiment.py
│       └── coinglass.py
│
├── signals/                   # Signal generation
│   ├── generator.py          # Main signal generator
│   ├── trend_filter.py       # Trend direction filter
│   ├── quality.py            # Signal quality scoring
│   ├── confluence/
│   │   └── multi_timeframe.py
│   └── microstructure/       # Order flow analysis
│       ├── order_flow.py
│       ├── vpin.py
│       └── trade_imbalance.py
│
├── crypto/                    # Cryptocurrency trading (The5ers 24/7)
│   ├── __init__.py           # Module exports
│   ├── crypto_strategy.py    # Main crypto strategy engine
│   ├── regime_detector.py    # Market regime classification
│   ├── liquidity_hunter.py   # Stop hunt detection
│   └── crypto_position_sizer.py  # Volatility-adjusted sizing
│
├── models/                    # ML models
│   ├── base.py               # Base model interface
│   ├── ensemble.py           # Model ensemble/voting
│   ├── momentum/
│   │   └── trend_following.py
│   ├── regime/
│   │   ├── hmm.py            # Hidden Markov Model
│   │   └── volatility_regime.py
│   ├── temporal/
│   │   ├── lstm_attention.py
│   │   ├── transformer.py
│   │   └── tcn.py
│   └── reinforcement/
│       ├── ppo_trader.py
│       └── actor_critic.py
│
├── portfolio/                 # Portfolio management
│   ├── construction/
│   │   ├── black_litterman.py
│   │   └── hierarchical_risk.py
│   └── optimization/
│       ├── solver.py
│       └── constraints.py
│
├── backtesting/               # Backtesting engine
│   ├── engine/
│   │   └── vectorized.py     # Fast vectorized backtester
│   ├── analysis/
│   │   ├── metrics.py        # Performance metrics
│   │   ├── drawdown_analysis.py
│   │   └── regime_analysis.py
│   ├── validation/
│   │   └── walk_forward.py   # Walk-forward optimization
│   └── reporting/
│       └── tearsheet.py      # Performance reports
│
├── monitoring/                # System monitoring
│   ├── telegram_bot.py       # Telegram notifications
│   ├── health/
│   │   ├── heartbeat.py      # Connection monitoring
│   │   ├── watchdog.py       # Process monitoring
│   │   └── diagnostics.py    # System diagnostics
│   └── metrics/
│       └── prometheus.py     # Metrics export
│
├── scripts/                   # Utility scripts
│   ├── start_paper_trading.py    # Paper trading runner
│   └── view_status.py            # Quick status viewer
│
├── storage/                   # Persistent storage
│   ├── state/                # Account state files (JSON)
│   ├── models/               # Saved ML models
│   └── logs/                 # Log files
│
├── logs/                      # Daily reports
│   └── daily_report_*.json
│
├── tests/                     # Test suite
│   ├── unit/
│   ├── integration/
│   └── stress/
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

**Codebase Stats:**
- 112 Python files
- ~36,000 lines of code
- Full test coverage for critical paths

---

## Prop Firm Compliance

### GFT Instant GOAT Rules (2026)

| Rule | Limit | Guardian | Action if Breached |
|------|-------|----------|-------------------|
| **Daily Loss** | 3% | 2.5% | Stop trading for day |
| **Total Drawdown** | 6% (trailing) | 5% | Stop all trading |
| **Floating Loss** | 2% combined | 1.8% | **INSTANT ACCOUNT CLOSURE** |
| **Trade Duration** | 2 minutes minimum | - | Profits deducted if <2min |
| **Risk Per Trade** | 2% per instrument | 1.5% | Block trade |
| **Consistency** | No day >15% of profits | - | Blocks payout only |
| **Inactivity** | 30 days | 25 days | Account closure |
| **Leverage** | 1:30 all assets | - | - |

**CRITICAL:** The 2% floating loss rule causes INSTANT account termination. The system monitors this every tick.

### The5ers High Stakes Rules

| Rule | Limit | Guardian | Notes |
|------|-------|----------|-------|
| **Daily Loss** | 5% | 4% | From previous day's max equity |
| **Total Drawdown** | 10% (STATIC) | 8% | Measured from initial balance |
| **Floating Loss** | No limit | - | More flexible than GFT |
| **News Blackout** | +/- 2 minutes | - | All execution blocked |

### Compliance Architecture

```
Trade Signal
    │
    ▼
┌─────────────────────────┐
│ Pre-Trade Compliance    │
│ ├─ Check daily DD       │
│ ├─ Check total DD       │
│ ├─ Check position limit │
│ ├─ Check risk per trade │
│ └─ Check news blackout  │
└─────────────────────────┘
    │ PASS
    ▼
┌─────────────────────────┐
│ Position Open           │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Real-Time Monitoring    │
│ ├─ Floating loss check  │ ← Every tick (GFT critical!)
│ ├─ Daily DD tracking    │
│ └─ Guardian triggers    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Position Close          │
│ ├─ Check 2-min duration │
│ └─ Update P&L tracking  │
└─────────────────────────┘
```

---

## Trading Strategy

### Signal Generation

The system uses a multi-model ensemble approach:

1. **Trend Following** - Momentum-based entries on confirmed trends
2. **Mean Reversion** - Counter-trend entries at statistical extremes
3. **Regime Detection** - HMM-based market state classification
4. **Order Flow Analysis** - VPIN and trade imbalance signals

### Entry Conditions

```python
Signal Generated When:
1. Confidence >= 0.60 (60%)
2. Trend filter aligned OR mean-reversion extreme
3. No existing position in symbol
4. Within position limit (max 3)
5. Passes all compliance checks
6. Not in signal cooldown (5 min per symbol)
```

### Stop Loss & Take Profit

All stops/targets are ATR-based (no fixed pips):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Stop Loss | 2.5 ATR | Wide enough to avoid noise |
| Take Profit | 5.0 ATR | 2:1 reward-to-risk |
| Min Confidence | 0.60 | Only high-quality signals |
| Signal Cooldown | 5 minutes | Prevent flip-flopping |

### Position Sizing

Uses modified Kelly Criterion:

```
Base Kelly = (Win Rate × Avg Win / Avg Loss - Loss Rate) / (Avg Win / Avg Loss)
Actual Size = Base Kelly × 0.25 × Risk Scalar

Risk Scalar:
- DD < 50% of guardian → 1.0 (full risk)
- DD 50-100% of guardian → Linear reduction to 0.25
- DD >= guardian → 0.0 (no new trades)
```

---

## Cryptocurrency Trading

### Overview

The5ers account trades BTCUSD and ETHUSD 24/7 using a specialized crypto strategy engine with:

- **Regime Detection** - Identifies trending vs ranging markets
- **Liquidity Hunt Detection** - Spots stop hunts for reversal entries
- **Multi-Timeframe Alignment** - 4H + 1H trend confirmation
- **Volatility-Adjusted Sizing** - Reduces size in high volatility

### Crypto Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Timeframe | 1H | Less noise than M15 |
| Max Risk | 0.5% | Half of forex (more volatile) |
| Min R:R | 2:1 | Larger targets for volatility |
| Max Positions | 2 | Limit crypto exposure |
| Regime Filter | Yes | Skip RANGING_VOLATILE |

### Symbol Suffix Handling

**Important:** Symbol suffixes differ between paper and live trading:

| Mode | MT5 Instance | Crypto Symbols | Forex/Indices |
|------|--------------|----------------|---------------|
| Paper Trading | GFT Account 1 | BTCUSD.x, ETHUSD.x | XAUUSD.x, etc. |
| Live (The5ers) | The5ers MT5 | BTCUSD, ETHUSD | XAUUSD, etc. |
| Live (GFT) | GFT MT5 | N/A | XAUUSD.x, etc. |

Paper trading uses GFT's MT5 for data, which requires the `.x` suffix for all symbols including crypto. Live trading on The5ers uses their own MT5 instance without the suffix.

---

## Paper Trading

### Quick Start

```cmd
# 1. Ensure MT5 is running and logged in

# 2. Delete old state (fresh start)
del storage\state\*.json

# 3. Start paper trading
python scripts/start_paper_trading.py
```

### What Paper Trading Does

- Connects to MT5 for **real market data**
- Simulates trades locally (no real orders)
- Tracks P&L in `storage/state/*.json`
- Sends Telegram notifications
- Saves daily reports to `logs/`

### Paper Trading Configuration

Edit `scripts/start_paper_trading.py`:

```python
# MT5 Connection (for data only)
USE_MT5_DATA = True
MT5_LOGIN = 314329147
MT5_PASSWORD = "your_password"
MT5_SERVER = "GoatFunded-Server"

# Risk Parameters
STOP_LOSS_ATR_MULT = 2.5      # Wider stops
TAKE_PROFIT_ATR_MULT = 5.0    # 2:1 R:R
MIN_SIGNAL_CONFIDENCE = 0.60  # Quality filter
SIGNAL_COOLDOWN_SECONDS = 300 # 5 min cooldown
```

### Monitoring Paper Trading

**Telegram Commands:**
- `/status` - Account balances and P&L
- `/positions` - Open positions
- `/trades` - Recent trade history
- `/help` - All commands

**Log Files:**
- Console output (real-time)
- `logs/daily_report_YYYYMMDD.json`

### Validation Criteria

Before going live, ensure:

```
□ Win rate > 40%
□ Profit factor > 1.2
□ Max drawdown < guardian limits
□ No 2-minute duration violations
□ Signals are consistent (not flip-flopping)
□ Run for minimum 2-3 weeks
```

---

## Live Deployment

### Pre-Deployment Checklist

```
□ Paper trading profitable for 2+ weeks
□ All compliance rules verified
□ MT5 credentials tested
□ Telegram notifications working
□ VPS has auto-restart configured
□ Backup of config files saved
```

### Starting Live Trading

**Single Account:**
```cmd
python bots/gft_bot.py --config config/gft_account_1.py
```

**All Accounts (parallel):**
```cmd
python run_all_accounts.py
```

**With Paper Mode First:**
```cmd
python bots/gft_bot.py --config config/gft_account_1.py --paper
```

### Scaling Strategy

```
Week 1: 1 GFT account live, others paper
Week 2: Add 2nd GFT if profitable
Week 3: Add 3rd GFT + The5ers
```

### Windows Task Scheduler (Auto-Restart)

1. Open Task Scheduler
2. Create Basic Task → "Sovereign V5"
3. Trigger: At startup
4. Action: Start a program
5. Program: `python.exe`
6. Arguments: `C:\Users\Administrator\sovereign-v5\run_all_accounts.py`
7. Start in: `C:\Users\Administrator\sovereign-v5`

---

## Module Reference

### Core Modules

| Module | Purpose |
|--------|---------|
| `core/mt5_connector.py` | Thread-safe MT5 connection with auto-reconnect |
| `core/execution.py` | Smart order execution (MARKET/TWAP/ICEBERG/ADAPTIVE) |
| `core/paper_executor.py` | Paper trading simulation engine |
| `core/position_sizer.py` | Kelly criterion position sizing |
| `core/news_calendar.py` | Economic event calendar |
| `core/risk_engine.py` | Pre-trade risk validation |
| `core/compliance/gft_compliance.py` | GFT rule enforcement |
| `core/compliance/the5ers_compliance.py` | The5ers rule enforcement |

### Data Modules

| Module | Purpose |
|--------|---------|
| `data/mt5_fetcher.py` | Real-time OHLCV from MT5 |
| `data/paper_fetcher.py` | Multi-source data for paper trading |
| `data/feature_engineer.py` | 50+ technical indicators |

### Signal Modules

| Module | Purpose |
|--------|---------|
| `signals/generator.py` | Main signal generation logic |
| `signals/trend_filter.py` | Trend direction classification |
| `signals/quality.py` | Signal quality scoring |

### Model Modules

| Module | Purpose |
|--------|---------|
| `models/ensemble.py` | Model voting/ensemble |
| `models/regime/hmm.py` | Market regime detection |
| `models/momentum/trend_following.py` | Trend signals |
| `models/temporal/transformer.py` | Deep learning forecasts |

---

## Configuration

### Account Config Files

Each account has a config file in `config/`:

```python
# config/gft_account_1.py

# MT5 Credentials
MT5_LOGIN = 314329147
MT5_PASSWORD = "your_password"
MT5_SERVER = "GoatFunded-Server"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Account Settings
ACCOUNT_NAME = "GFT_10K_1"
ACCOUNT_SIZE = 10000
INITIAL_BALANCE = 10000
ACCOUNT_TYPE = "GFT"

# Telegram
TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHAT_IDS = [your_chat_id]

# Symbols (with broker suffix)
SYMBOLS = ["XAUUSD.x", "XAGUSD.x", "NAS100.x", "UK100.x", "SPX500.x", "EURUSD.x"]

# Compliance Limits
MAX_DAILY_DD_PCT = 3.0
MAX_TOTAL_DD_PCT = 6.0
MAX_FLOATING_LOSS_PCT = 2.0
MIN_TRADE_DURATION_SECONDS = 120

# Guardian Limits (stop before breach)
GUARDIAN_DAILY_DD_PCT = 2.5
GUARDIAN_TOTAL_DD_PCT = 5.0
GUARDIAN_FLOATING_PCT = 1.8
```

---

## Telegram Integration

### Setup

1. Create bot via @BotFather
2. Get token and chat ID
3. Add to config files

### Available Commands

| Command | Description |
|---------|-------------|
| `/status` | Account summary (balance, equity, DD) |
| `/positions` | Open positions list |
| `/trades` | Recent trade history |
| `/help` | Command list |
| `/ping` | Check bot is alive |

### Notification Types

- **Trade Opened** - Entry price, size, SL/TP
- **Trade Closed** - P&L, close reason
- **Status Update** - Every 30 minutes
- **Compliance Warning** - Guardian triggers
- **Startup/Shutdown** - System status

---

## Troubleshooting

### Common Issues

**"MT5 not available"**
```cmd
pip install MetaTrader5
# Ensure MT5 terminal is running
```

**"Using synthetic data for..."**
- Symbol not available on connected broker
- Check symbol names (may need `.x` suffix)
- Market may be closed

**"Daily DD guardian triggered"**
- Account hit 2.5% daily loss (guardian limit, actual is 3%)
- Trading paused until next day (5 PM EST reset)
- Reset: `del storage\state\*.json`

**"Floating loss breach"**
- Position approaching -2% limit
- Guardian auto-closes at 1.8%
- This is working as intended!

**Unicode errors on Windows**
- Already fixed (uses ASCII dashes)
- If persists: `set PYTHONIOENCODING=utf-8`

### Log Locations

| Log | Location |
|-----|----------|
| Paper trading | Console + `logs/daily_report_*.json` |
| Live trading | `logs/gft_bot_*.log` |
| State files | `storage/state/*.json` |

### Reset Accounts

```cmd
# Delete all state (fresh start)
del storage\state\*.json

# Delete specific account
del storage\state\GFT_1_*.json
```

---

## Security Notes

1. **Never commit credentials** to git
2. **Use environment variables** for production
3. **Restrict VPS access** (firewall, SSH keys)
4. **Monitor Telegram** for unauthorized commands
5. **Backup state files** regularly

---

## Support

- Check logs first: `logs/` and console output
- Run with `--debug` for verbose output
- Review state files: `storage/state/*.json`

---

**DISCLAIMER:** Trading involves substantial risk of loss. This system is for educational purposes. Past performance does not guarantee future results. Only trade with capital you can afford to lose.
