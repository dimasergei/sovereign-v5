# INSTITUTIONAL-GRADE AUTONOMOUS TRADING SYSTEM

## MISSION BRIEF

You are a principal quantitative researcher and systems architect who has worked at Renaissance Technologies, Two Sigma, DE Shaw, and Citadel. You are tasked with building a fully autonomous trading system that implements institutional-grade methodologies adapted for funded proprietary trading accounts.

**Performance Target**: Compete with Medallion Fund's risk-adjusted returns (~66% annual gross, Sharpe >2.0) while operating within prop firm constraints.

**Philosophy**: This system follows the "Lossless Principle" - NO hardcoded strategy parameters. Every threshold, period, multiplier, and signal weight must be derived dynamically from market data observation. Magic numbers are forbidden. The system must be able to run for 12+ years without parameter updates, adapting entirely through market observation.

---

## PART I: THEORETICAL FOUNDATION

### 1.1 The Alpha Generation Framework

Renaissance's edge comes from finding thousands of small, uncorrelated alpha signals. Each signal alone may be weak (51-52% accuracy), but combined through optimal portfolio construction, they compound into extraordinary returns.

**Alpha Categories to Implement:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ALPHA SIGNAL TAXONOMY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 1: MICROSTRUCTURE SIGNALS (Tick/Second Resolution)                   │
│  ├── Order flow imbalance (bid-ask volume differential)                     │
│  ├── Trade arrival intensity (Hawkes process modeling)                      │
│  ├── Spread dynamics and mean-reversion                                     │
│  ├── Quote stuffing detection / liquidity withdrawal                        │
│  ├── Iceberg order detection                                                │
│  └── Toxic flow identification (adverse selection)                          │
│                                                                              │
│  LEVEL 2: STATISTICAL ARBITRAGE SIGNALS (Minute Resolution)                 │
│  ├── Cross-asset correlation breakdowns                                     │
│  ├── Lead-lag relationships (BTC leads ETH, EUR leads GBP)                 │
│  ├── Cointegration residual mean-reversion                                  │
│  ├── Principal component analysis (PCA) factor residuals                    │
│  ├── Pairs trading z-score divergence                                       │
│  └── Basket vs constituent mispricing                                       │
│                                                                              │
│  LEVEL 3: REGIME-AWARE SIGNALS (Hourly Resolution)                          │
│  ├── Hidden Markov Model state transitions                                  │
│  ├── Volatility regime clustering (GARCH family)                            │
│  ├── Correlation regime shifts                                              │
│  ├── Trend vs mean-reversion regime classification                          │
│  ├── Liquidity regime detection                                             │
│  └── Market stress indicators                                               │
│                                                                              │
│  LEVEL 4: FUNDAMENTAL/ALTERNATIVE DATA (Daily Resolution)                   │
│  ├── Funding rate arbitrage (crypto perpetuals vs spot)                     │
│  ├── On-chain metrics (exchange flows, whale movements)                     │
│  ├── Social sentiment aggregation (NLP on Twitter, Reddit)                  │
│  ├── News event impact modeling                                             │
│  ├── Fear & Greed index derivatives                                         │
│  └── Cross-exchange price discrepancies                                     │
│                                                                              │
│  LEVEL 5: MARKET-DERIVED PARAMETERS (Adaptive)                              │
│  ├── Volatility-adjusted lookback windows                                   │
│  ├── Entropy-based period selection                                         │
│  ├── Fractal dimension for trend strength                                   │
│  ├── Hurst exponent for mean-reversion timing                               │
│  ├── Market-implied thresholds from distribution analysis                   │
│  └── Self-calibrating signal weights via online learning                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Lossless Principle (No Magic Numbers)

Every parameter must be derived from market data. Here's how:

| Traditional Approach (FORBIDDEN) | Lossless Approach (REQUIRED) |
|----------------------------------|------------------------------|
| `RSI_PERIOD = 14` | Period = f(market_entropy, volatility_clustering) |
| `OVERBOUGHT = 70` | Threshold = percentile_rank(current_RSI, rolling_distribution) |
| `ATR_MULTIPLIER = 2.0` | Multiplier = f(realized_vs_implied_vol, regime_state) |
| `EMA_FAST = 12, EMA_SLOW = 26` | Periods = spectral_analysis(price_series) |
| `RISK_PER_TRADE = 0.8%` | Risk = kelly_fraction(edge, variance) * regime_scalar |
| `TAKE_PROFIT = 2x SL` | TP = expected_move(volatility, momentum, support_resistance) |

**Implementation Pattern:**

```python
class LosslessParameter:
    """
    Self-calibrating parameter that derives its value from market observation.
    
    INVARIANT: No hardcoded default values allowed. Initial value must come
    from historical data analysis during initialization.
    """
    
    def __init__(self, derivation_function: Callable, min_samples: int = 100):
        self.derive = derivation_function
        self.min_samples = min_samples
        self.value = None
        self.confidence = 0.0
        self.last_calibration = None
        self.history = deque(maxlen=1000)
    
    def calibrate(self, market_data: pd.DataFrame) -> float:
        """
        Derive parameter value from market data.
        Must be called before any trading decisions.
        """
        if len(market_data) < self.min_samples:
            raise InsufficientDataError(f"Need {self.min_samples} samples, got {len(market_data)}")
        
        self.value = self.derive(market_data)
        self.confidence = self._compute_confidence(market_data)
        self.last_calibration = datetime.now()
        self.history.append((self.last_calibration, self.value, self.confidence))
        
        return self.value
    
    def get(self) -> float:
        """Get current value. Raises if not calibrated."""
        if self.value is None:
            raise NotCalibratedError("Parameter not yet calibrated from market data")
        return self.value
```

### 1.3 Multi-Model Ensemble Architecture

Top quant funds don't rely on a single model. They use ensembles of diverse models that vote on signals.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE ARCHITECTURE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌──────────────────┐                                │
│                         │  META-LEARNER    │                                │
│                         │  (Stacking)      │                                │
│                         └────────┬─────────┘                                │
│                                  │                                          │
│         ┌────────────────────────┼────────────────────────┐                │
│         │                        │                        │                │
│         ▼                        ▼                        ▼                │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │ TEMPORAL    │         │ STATISTICAL │         │ ALTERNATIVE │          │
│  │ MODELS      │         │ MODELS      │         │ DATA MODELS │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
│         │                        │                        │                │
│    ┌────┴────┐             ┌────┴────┐             ┌────┴────┐            │
│    │         │             │         │             │         │            │
│    ▼         ▼             ▼         ▼             ▼         ▼            │
│ ┌─────┐  ┌─────┐       ┌─────┐  ┌─────┐       ┌─────┐  ┌─────┐          │
│ │LSTM │  │Trans│       │Stat │  │Pairs│       │Sent-│  │Fund-│          │
│ │Attn │  │form-│       │Arb  │  │Trade│       │iment│  │ing  │          │
│ │     │  │er   │       │     │  │     │       │NLP  │  │Rate │          │
│ └─────┘  └─────┘       └─────┘  └─────┘       └─────┘  └─────┘          │
│    │         │             │         │             │         │            │
│ ┌─────┐  ┌─────┐       ┌─────┐  ┌─────┐       ┌─────┐  ┌─────┐          │
│ │WaveN│  │N-BE-│       │Mean │  │Lead-│       │On-  │  │Order│          │
│ │et   │  │ATS  │       │Rev  │  │Lag  │       │Chain│  │Flow │          │
│ └─────┘  └─────┘       └─────┘  └─────┘       └─────┘  └─────┘          │
│                                                                          │
│  VOTING MECHANISM:                                                       │
│  - Each model outputs: (direction, magnitude, confidence)                │
│  - Weights updated via online gradient descent on realized PnL           │
│  - Disagreement → reduce position size or abstain                        │
│  - Agreement → full conviction position                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART II: ACCOUNT SPECIFICATIONS & CONSTRAINTS

### 2.1 Goat Funded Trader (GFT) - Three $10,000 Accounts

```yaml
account_type: "instant_funded"
initial_balance: 10000
currency: "USD"

risk_limits:
  max_overall_drawdown:
    value: 8.0
    type: "percentage"
    reference: "trailing_high_water_mark"
    # GUARDIAN: Hard stop at 7.0% to ensure NEVER breach
    guardian_threshold: 7.0
  
  daily_loss_limit: null  # GFT doesn't have daily limit
  
  inactivity_limit:
    days: 30
    action: "auto_ping_trade"
    guardian_days: 25  # Ping at 25 days

prohibited_activities:
  - hedging: "No opposite positions on same instrument"
  - martingale: "No position size increase after loss"
  - grid_trading: "No systematic grid entries"
  - high_frequency: "No trades < 1 minute apart"
  - news_trading: "Avoid 5 min before/after major news"

allowed_instruments:
  crypto_cfds:
    - "BTCUSD.x"
    - "ETHUSD.x"
    - "SOLUSD.x"
    - "XRPUSD.x"
    - "LTCUSD.x"
    - "BNBUSD.x"
    - "ADAUSD.x"
    - "DOTUSD.x"
    - "AVAXUSD.x"
    - "MATICUSD.x"

trading_hours:
  type: "24/7"
  session_preferences:
    high_volatility: ["00:00-04:00 UTC", "12:00-16:00 UTC"]  # Asia/US overlap
    low_volatility: ["08:00-12:00 UTC"]  # European morning

leverage:
  max_available: 100
  recommended_use: 20-50  # Conservative
  dynamic: true  # Reduce in high volatility

position_sizing:
  method: "adaptive_kelly"
  base_risk_fraction: "derived_from_sharpe"  # NOT hardcoded
  max_risk_per_trade: "derived_from_volatility_regime"
  concentration_limit: 0.5  # Max 50% of risk in single position
```

### 2.2 The5ers - One $5,000 High Stakes Account

```yaml
account_type: "high_stakes"
initial_balance: 5000
currency: "USD"

risk_limits:
  max_daily_loss:
    value: 5.0
    type: "percentage"
    reference: "day_start_balance"
    reset_time: "00:00 UTC"
    guardian_threshold: 4.0  # Stop trading at 4%
  
  max_overall_loss:
    value: 10.0
    type: "percentage"
    reference: "initial_balance"
    guardian_threshold: 8.5
  
  consistency_rule:
    description: "No single day > 30% of total profit"
    enforcement: "pre_trade_check"
    action: "reduce_position_size"
  
  profit_target:
    phase_1: 8.0  # 8% to pass
    phase_2: 5.0  # If applicable

prohibited_activities:
  - hedging: true
  - martingale: true
  - weekend_holding: "check_firm_rules"  # May need Friday closeout
  - overnight_gaps: "manage_with_reduced_size"

allowed_instruments:
  forex_majors:
    - "EURUSD"
    - "GBPUSD"
    - "USDJPY"
    - "USDCHF"
    - "AUDUSD"
    - "USDCAD"
    - "NZDUSD"
  forex_minors:
    - "EURGBP"
    - "EURJPY"
    - "GBPJPY"
    - "AUDJPY"
    - "EURAUD"
    - "EURCHF"
  # Avoid exotic pairs (wide spreads, low liquidity)

trading_hours:
  primary_sessions:
    london: "08:00-17:00 UTC"
    new_york: "13:00-22:00 UTC"
    overlap: "13:00-17:00 UTC"  # Highest liquidity
  avoid:
    - "22:00-00:00 UTC"  # Low liquidity
    - "Friday 20:00+ UTC"  # Weekend risk
  
session_optimization:
  london: ["EURUSD", "GBPUSD", "EURGBP"]
  new_york: ["EURUSD", "USDJPY", "USDCAD"]
  tokyo: ["USDJPY", "AUDJPY", "EURJPY"]

position_sizing:
  method: "adaptive_kelly_with_daily_limit"
  daily_risk_budget: "derived_from_remaining_daily_limit"
  max_concurrent_positions: 2
  correlation_adjustment: true  # Reduce if positions correlated
```

---

## PART III: SYSTEM ARCHITECTURE

### 3.1 Complete Directory Structure

```
prop_trading_system/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .env.template                    # Environment variables template
├── docker-compose.yml               # Container orchestration
├── Makefile                         # Build automation
│
├── config/
│   ├── __init__.py
│   ├── base.py                      # Base configuration class
│   ├── accounts/
│   │   ├── gft_account1.yaml
│   │   ├── gft_account2.yaml
│   │   ├── gft_account3.yaml
│   │   └── the5ers.yaml
│   ├── models/
│   │   ├── lstm_config.yaml
│   │   ├── transformer_config.yaml
│   │   ├── ensemble_config.yaml
│   │   └── rl_config.yaml
│   └── trading/
│       ├── crypto_strategy.yaml
│       └── forex_strategy.yaml
│
├── core/
│   ├── __init__.py
│   ├── lossless/
│   │   ├── __init__.py
│   │   ├── parameter.py             # LosslessParameter class
│   │   ├── calibrator.py            # Market-based calibration
│   │   ├── entropy.py               # Entropy-based period selection
│   │   ├── spectral.py              # Spectral analysis for cycles
│   │   ├── fractal.py               # Fractal dimension calculator
│   │   └── hurst.py                 # Hurst exponent calculator
│   ├── connection/
│   │   ├── __init__.py
│   │   ├── mt5_connector.py         # MT5 with circuit breaker
│   │   ├── websocket_feed.py        # Real-time data streaming
│   │   └── connection_pool.py       # Connection management
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── order_router.py          # Smart order routing
│   │   ├── execution_algo.py        # TWAP, VWAP, Iceberg
│   │   ├── slippage_model.py        # Slippage prediction
│   │   ├── market_impact.py         # Impact estimation
│   │   └── fill_analyzer.py         # Post-trade analysis
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_engine.py           # Central risk orchestrator
│   │   ├── position_sizer.py        # Kelly + regime adjustment
│   │   ├── drawdown_monitor.py      # Real-time DD tracking
│   │   ├── correlation_manager.py   # Cross-position correlation
│   │   ├── var_calculator.py        # Value at Risk
│   │   ├── stress_tester.py         # Scenario analysis
│   │   └── firm_rules.py            # Prop firm rule enforcement
│   ├── state/
│   │   ├── __init__.py
│   │   ├── state_machine.py         # Trading state management
│   │   ├── persistence.py           # JSON/SQLite persistence
│   │   ├── recovery.py              # Crash recovery
│   │   └── checkpointing.py         # Model checkpoints
│   └── utils/
│       ├── __init__.py
│       ├── time_utils.py
│       ├── math_utils.py
│       ├── validation.py
│       └── decorators.py
│
├── data/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── mt5_fetcher.py           # MT5 historical data
│   │   ├── tick_collector.py        # Tick-by-tick collection
│   │   ├── orderbook_collector.py   # Order book snapshots
│   │   └── alternative_data.py      # External data sources
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── cleaner.py               # Data cleaning
│   │   ├── normalizer.py            # Normalization
│   │   ├── resampler.py             # Multi-timeframe
│   │   └── validator.py             # Data quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py             # Technical indicators
│   │   ├── microstructure.py        # Order flow features
│   │   ├── cross_asset.py           # Cross-asset features
│   │   ├── regime.py                # Regime indicators
│   │   ├── sentiment.py             # Sentiment features
│   │   └── factory.py               # Feature factory
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── timeseries_db.py         # Time series storage
│   │   ├── feature_store.py         # Feature caching
│   │   └── trade_journal.py         # Trade history
│   └── external/
│       ├── __init__.py
│       ├── coinglass.py             # Funding rates
│       ├── glassnode.py             # On-chain metrics
│       ├── lunarcrush.py            # Social sentiment
│       ├── alternative_me.py        # Fear & Greed
│       └── news_api.py              # News feeds
│
├── models/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_model.py            # Abstract base
│   │   ├── online_learner.py        # Online learning interface
│   │   └── ensemble_member.py       # Ensemble interface
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── lstm_attention.py        # LSTM with attention
│   │   ├── transformer.py           # Temporal Transformer
│   │   ├── wavenet.py               # WaveNet for sequences
│   │   ├── nbeats.py                # N-BEATS architecture
│   │   └── tcn.py                   # Temporal CNN
│   ├── statistical/
│   │   ├── __init__.py
│   │   ├── mean_reversion.py        # Ornstein-Uhlenbeck
│   │   ├── pairs_trading.py         # Cointegration
│   │   ├── lead_lag.py              # Lead-lag detection
│   │   └── pca_residual.py          # PCA factor model
│   ├── regime/
│   │   ├── __init__.py
│   │   ├── hmm.py                   # Hidden Markov Model
│   │   ├── jump_diffusion.py        # Jump detection
│   │   ├── volatility_regime.py     # Vol clustering
│   │   └── correlation_regime.py    # Correlation shifts
│   ├── alternative/
│   │   ├── __init__.py
│   │   ├── sentiment_model.py       # NLP sentiment
│   │   ├── funding_rate_model.py    # Funding arbitrage
│   │   ├── flow_model.py            # Order flow
│   │   └── onchain_model.py         # On-chain signals
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── stacking.py              # Stacking meta-learner
│   │   ├── voting.py                # Weighted voting
│   │   ├── bayesian_combination.py  # Bayesian model averaging
│   │   └── dynamic_weighting.py     # Online weight updates
│   └── reinforcement/
│       ├── __init__.py
│       ├── policy_gradient.py       # Policy optimization
│       ├── actor_critic.py          # A2C/A3C
│       ├── ppo_trader.py            # PPO for trading
│       └── reward_shaping.py        # Reward engineering
│
├── signals/
│   ├── __init__.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── microstructure.py        # Tick-level signals
│   │   ├── momentum.py              # Trend signals
│   │   ├── mean_reversion.py        # Mean reversion signals
│   │   ├── breakout.py              # Breakout detection
│   │   ├── support_resistance.py    # S/R levels (counting-based)
│   │   └── volume_profile.py        # Volume analysis
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── regime_filter.py         # Regime-based filtering
│   │   ├── correlation_filter.py    # Correlation-based
│   │   ├── volatility_filter.py     # Vol-adjusted
│   │   └── quality_filter.py        # Signal quality scoring
│   ├── combination/
│   │   ├── __init__.py
│   │   ├── signal_aggregator.py     # Combine signals
│   │   ├── conflict_resolver.py     # Handle disagreements
│   │   └── confidence_scorer.py     # Overall confidence
│   └── validation/
│       ├── __init__.py
│       ├── signal_decay.py          # Signal decay analysis
│       ├── alpha_persistence.py     # Alpha half-life
│       └── backtest_validator.py    # Out-of-sample validation
│
├── portfolio/
│   ├── __init__.py
│   ├── construction/
│   │   ├── __init__.py
│   │   ├── kelly_sizing.py          # Kelly criterion
│   │   ├── mean_variance.py         # MVO optimization
│   │   ├── risk_parity.py           # Risk parity
│   │   ├── black_litterman.py       # Black-Litterman
│   │   └── hierarchical_risk.py     # HRP
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── constraints.py           # Portfolio constraints
│   │   ├── objective.py             # Objective functions
│   │   └── solver.py                # Optimization solver
│   └── rebalancing/
│       ├── __init__.py
│       ├── trigger.py               # Rebalance triggers
│       ├── transaction_cost.py      # Cost-aware rebalancing
│       └── scheduler.py             # Rebalance scheduling
│
├── monitoring/
│   ├── __init__.py
│   ├── telegram/
│   │   ├── __init__.py
│   │   ├── bot.py                   # Telegram bot
│   │   ├── commands.py              # Command handlers
│   │   ├── alerts.py                # Alert system
│   │   └── reports.py               # Report generation
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── structured_logger.py     # JSON logging
│   │   ├── trade_logger.py          # Trade-specific logs
│   │   └── performance_logger.py    # Performance metrics
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── real_time.py             # Real-time metrics
│   │   ├── aggregator.py            # Metric aggregation
│   │   └── prometheus.py            # Prometheus export
│   ├── health/
│   │   ├── __init__.py
│   │   ├── heartbeat.py             # Heartbeat monitor
│   │   ├── watchdog.py              # Process watchdog
│   │   └── diagnostics.py           # System diagnostics
│   └── dashboard/
│       ├── __init__.py
│       ├── api.py                   # REST API
│       └── websocket.py             # WebSocket updates
│
├── backtesting/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── vectorized.py            # Fast vectorized backtest
│   │   ├── event_driven.py          # Event-driven backtest
│   │   └── tick_replay.py           # Tick-by-tick replay
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Performance metrics
│   │   ├── attribution.py           # Return attribution
│   │   ├── regime_analysis.py       # Regime performance
│   │   └── drawdown_analysis.py     # Drawdown decomposition
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── walk_forward.py          # Walk-forward analysis
│   │   ├── monte_carlo.py           # Monte Carlo simulation
│   │   ├── cross_validation.py      # Time series CV
│   │   └── robustness.py            # Parameter sensitivity
│   └── reporting/
│       ├── __init__.py
│       ├── html_report.py           # HTML reports
│       ├── pdf_report.py            # PDF reports
│       └── tearsheet.py             # Quantstats tearsheet
│
├── bots/
│   ├── __init__.py
│   ├── gft_bot_account1.py          # Standalone GFT bot 1
│   ├── gft_bot_account2.py          # Standalone GFT bot 2
│   ├── gft_bot_account3.py          # Standalone GFT bot 3
│   └── the5ers_bot.py               # Standalone The5ers bot
│
├── scripts/
│   ├── install_service.ps1          # Windows service
│   ├── install_service.sh           # Linux systemd
│   ├── deploy.py                    # Deployment script
│   ├── train_models.py              # Model training
│   ├── run_backtest.py              # Backtest runner
│   ├── optimize_hyperparams.py      # Hyperparameter optimization
│   ├── calibrate_parameters.py      # Lossless calibration
│   └── generate_report.py           # Report generation
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_risk_engine.py
│   │   ├── test_position_sizer.py
│   │   ├── test_lossless_params.py
│   │   ├── test_signals.py
│   │   └── test_execution.py
│   ├── integration/
│   │   ├── test_mt5_integration.py
│   │   ├── test_data_pipeline.py
│   │   └── test_model_pipeline.py
│   ├── stress/
│   │   ├── test_drawdown_scenarios.py
│   │   ├── test_connection_failures.py
│   │   └── test_extreme_volatility.py
│   └── fixtures/
│       ├── sample_data.py
│       └── mock_mt5.py
│
└── storage/
    ├── models/                      # Trained model weights
    ├── state/                       # Persistent state
    ├── logs/                        # Log files
    ├── data/                        # Historical data cache
    ├── checkpoints/                 # Training checkpoints
    └── reports/                     # Generated reports
```

### 3.2 Dependencies (requirements.txt)

```
# =============================================================================
# CORE TRADING
# =============================================================================
MetaTrader5==5.0.45

# =============================================================================
# DATA PROCESSING
# =============================================================================
pandas==2.1.0
numpy==1.24.3
scipy==1.11.2
polars==0.19.0                  # Fast DataFrame operations
pyarrow==13.0.0                 # Parquet/Arrow support

# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================
ta==0.10.2
pandas-ta==0.3.14b
talib-binary==0.4.24            # TA-Lib (faster C implementation)

# =============================================================================
# MACHINE LEARNING - DEEP LEARNING
# =============================================================================
tensorflow==2.13.0
keras==2.13.1
torch==2.0.1                    # PyTorch for Transformers
transformers==4.33.0            # Hugging Face Transformers

# =============================================================================
# MACHINE LEARNING - CLASSICAL
# =============================================================================
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
catboost==1.2

# =============================================================================
# MACHINE LEARNING - SPECIALIZED
# =============================================================================
hmmlearn==0.3.0                 # Hidden Markov Models
arch==6.1.0                     # GARCH models
statsmodels==0.14.0             # Statistical models
pykalman==0.9.5                 # Kalman filters
ruptures==1.1.8                 # Change point detection

# =============================================================================
# REINFORCEMENT LEARNING
# =============================================================================
stable-baselines3==2.1.0
gymnasium==0.29.1

# =============================================================================
# OPTIMIZATION
# =============================================================================
optuna==3.3.0
hyperopt==0.2.7
cvxpy==1.4.1                    # Convex optimization
pypfopt==1.5.5                  # Portfolio optimization

# =============================================================================
# ALTERNATIVE DATA
# =============================================================================
ccxt==4.0.0                     # Crypto exchanges
tweepy==4.14.0                  # Twitter API
praw==7.7.1                     # Reddit API
newsapi-python==0.2.7           # News API
textblob==0.17.1                # Sentiment analysis
transformers==4.33.0            # BERT for NLP

# =============================================================================
# NETWORKING
# =============================================================================
aiohttp==3.8.5
websockets==11.0.3
requests==2.31.0
httpx==0.24.1

# =============================================================================
# TELEGRAM
# =============================================================================
python-telegram-bot==20.5

# =============================================================================
# DATABASE
# =============================================================================
sqlalchemy==2.0.21
redis==5.0.0
influxdb-client==1.37.0         # Time series database

# =============================================================================
# UTILITIES
# =============================================================================
python-dotenv==1.0.0
pyyaml==6.0.1
schedule==1.2.0
pytz==2023.3
colorlog==6.7.0
tqdm==4.66.1
joblib==1.3.2
dill==0.3.7                     # Advanced pickling

# =============================================================================
# PERFORMANCE
# =============================================================================
numba==0.58.0                   # JIT compilation
cython==3.0.2                   # C extensions
bottleneck==1.3.7               # Fast NumPy operations

# =============================================================================
# WINDOWS SERVICE
# =============================================================================
pywin32==306; sys_platform == 'win32'

# =============================================================================
# TESTING
# =============================================================================
pytest==7.4.2
pytest-asyncio==0.21.1
pytest-cov==4.1.0
hypothesis==6.87.0              # Property-based testing

# =============================================================================
# VISUALIZATION (for reports)
# =============================================================================
matplotlib==3.8.0
seaborn==0.12.2
plotly==5.17.0
quantstats==0.0.62              # Tearsheets
```

---

## PART IV: CORE IMPLEMENTATIONS

### 4.1 Lossless Parameter Calibration System

```python
"""
core/lossless/calibrator.py

Market-Derived Parameter Calibration System

This is the heart of the lossless philosophy. Every trading parameter
is derived from market observation, never hardcoded.
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from .entropy import market_entropy, optimal_lookback_from_entropy
from .spectral import dominant_cycle_period, spectral_density
from .fractal import fractal_dimension, box_counting_dimension
from .hurst import hurst_exponent, regime_from_hurst


class CalibrationMethod(Enum):
    """Methods for deriving parameters from market data."""
    ENTROPY = "entropy"
    SPECTRAL = "spectral"
    FRACTAL = "fractal"
    STATISTICAL = "statistical"
    REGIME = "regime"
    VOLATILITY = "volatility"
    DISTRIBUTION = "distribution"


@dataclass
class CalibratedParameter:
    """A parameter with its derived value and metadata."""
    name: str
    value: float
    method: CalibrationMethod
    confidence: float
    valid_until: Optional[pd.Timestamp]
    market_context: Dict[str, Any]


class MarketCalibrator:
    """
    Derives all trading parameters from market data observation.
    
    PRINCIPLE: The market tells us what parameters to use, we don't impose
    them. This allows the system to adapt to any market regime without
    manual intervention.
    
    Example derivations:
    - RSI period → Dominant cycle from spectral analysis
    - Overbought threshold → 95th percentile of recent RSI distribution
    - ATR multiplier → Ratio of realized to implied volatility
    - EMA periods → Spectral peaks in price series
    - Breakout threshold → 2σ of recent range distribution
    """
    
    def __init__(
        self,
        min_calibration_bars: int = 500,
        recalibration_threshold: float = 0.15,  # Recalibrate if market regime shift >15%
        logger: Optional[logging.Logger] = None
    ):
        self.min_bars = min_calibration_bars
        self.recalibration_threshold = recalibration_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self.cached_parameters: Dict[str, CalibratedParameter] = {}
        self.last_regime_state: Optional[Dict] = None
    
    def calibrate_all(self, df: pd.DataFrame) -> Dict[str, CalibratedParameter]:
        """
        Perform full calibration of all trading parameters.
        
        Args:
            df: OHLCV DataFrame with at least self.min_bars rows
            
        Returns:
            Dictionary of all calibrated parameters
        """
        if len(df) < self.min_bars:
            raise ValueError(f"Need at least {self.min_bars} bars for calibration")
        
        params = {}
        
        # 1. Derive lookback periods from entropy and spectral analysis
        params['fast_period'] = self._derive_fast_period(df)
        params['slow_period'] = self._derive_slow_period(df)
        params['signal_period'] = self._derive_signal_period(df)
        
        # 2. Derive momentum thresholds from distribution analysis
        params['overbought_threshold'] = self._derive_overbought(df)
        params['oversold_threshold'] = self._derive_oversold(df)
        
        # 3. Derive volatility parameters
        params['atr_period'] = self._derive_atr_period(df)
        params['volatility_scalar'] = self._derive_volatility_scalar(df)
        
        # 4. Derive mean-reversion parameters
        params['mean_reversion_halflife'] = self._derive_halflife(df)
        params['mean_reversion_threshold'] = self._derive_mr_threshold(df)
        
        # 5. Derive breakout parameters
        params['breakout_threshold'] = self._derive_breakout_threshold(df)
        params['consolidation_threshold'] = self._derive_consolidation_threshold(df)
        
        # 6. Derive risk parameters
        params['optimal_risk_fraction'] = self._derive_kelly_fraction(df)
        params['stop_loss_atr_multiple'] = self._derive_sl_multiple(df)
        params['take_profit_atr_multiple'] = self._derive_tp_multiple(df)
        
        # 7. Derive regime parameters
        params['regime_lookback'] = self._derive_regime_lookback(df)
        params['trend_strength_threshold'] = self._derive_trend_threshold(df)
        
        self.cached_parameters = params
        return params
    
    def _derive_fast_period(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive fast EMA period from spectral analysis.
        
        The fast period corresponds to the shortest significant cycle
        in the price series.
        """
        close = df['close'].values
        
        # Compute power spectral density
        freqs, psd = spectral_density(close)
        
        # Find peaks in PSD (significant cycles)
        peak_indices = signal.find_peaks(psd, prominence=np.std(psd))[0]
        
        if len(peak_indices) == 0:
            # Fallback: use entropy-based estimation
            period = optimal_lookback_from_entropy(close, min_period=5, max_period=50)
        else:
            # Convert highest frequency significant peak to period
            peak_freqs = freqs[peak_indices]
            highest_freq = np.max(peak_freqs[peak_freqs > 0])
            period = int(1 / highest_freq)
        
        # Bound to reasonable range (market-derived bounds)
        min_period = max(3, int(len(close) * 0.005))  # At least 0.5% of data
        max_period = int(len(close) * 0.1)  # At most 10% of data
        period = np.clip(period, min_period, max_period)
        
        return CalibratedParameter(
            name="fast_period",
            value=float(period),
            method=CalibrationMethod.SPECTRAL,
            confidence=self._compute_spectral_confidence(psd, peak_indices),
            valid_until=None,
            market_context={"dominant_cycles": peak_indices.tolist()}
        )
    
    def _derive_slow_period(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive slow EMA period from spectral analysis.
        
        The slow period corresponds to the dominant (highest power) cycle.
        """
        close = df['close'].values
        freqs, psd = spectral_density(close)
        
        # Find dominant cycle (highest power)
        dominant_idx = np.argmax(psd[1:]) + 1  # Skip DC component
        dominant_freq = freqs[dominant_idx]
        
        if dominant_freq > 0:
            period = int(1 / dominant_freq)
        else:
            # Fallback: use Hurst exponent to determine regime-appropriate period
            H = hurst_exponent(close)
            if H > 0.6:  # Trending
                period = int(len(close) * 0.15)
            else:  # Mean-reverting
                period = int(len(close) * 0.05)
        
        # Ensure slow > fast
        fast_period = self._derive_fast_period(df).value
        period = max(period, int(fast_period * 2))
        
        return CalibratedParameter(
            name="slow_period",
            value=float(period),
            method=CalibrationMethod.SPECTRAL,
            confidence=psd[dominant_idx] / np.sum(psd),
            valid_until=None,
            market_context={"dominant_frequency": dominant_freq}
        )
    
    def _derive_overbought(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive overbought threshold from RSI distribution.
        
        Instead of hardcoding 70, we use the percentile that historically
        preceded reversals.
        """
        # Calculate RSI with derived period
        rsi_period = self._derive_signal_period(df).value
        rsi = self._calculate_rsi(df['close'], int(rsi_period))
        
        # Find RSI values that preceded down moves
        future_returns = df['close'].pct_change(5).shift(-5)  # 5-bar forward return
        down_move_threshold = np.percentile(future_returns.dropna(), 25)
        
        rsi_before_down = rsi[future_returns < down_move_threshold]
        
        if len(rsi_before_down) > 20:
            # Threshold is 75th percentile of RSI values before down moves
            threshold = np.percentile(rsi_before_down.dropna(), 75)
        else:
            # Fallback: 90th percentile of all RSI values
            threshold = np.percentile(rsi.dropna(), 90)
        
        return CalibratedParameter(
            name="overbought_threshold",
            value=float(threshold),
            method=CalibrationMethod.DISTRIBUTION,
            confidence=len(rsi_before_down) / len(rsi) if len(rsi) > 0 else 0.5,
            valid_until=None,
            market_context={"sample_size": len(rsi_before_down)}
        )
    
    def _derive_oversold(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive oversold threshold from RSI distribution (mirror of overbought)."""
        rsi_period = self._derive_signal_period(df).value
        rsi = self._calculate_rsi(df['close'], int(rsi_period))
        
        future_returns = df['close'].pct_change(5).shift(-5)
        up_move_threshold = np.percentile(future_returns.dropna(), 75)
        
        rsi_before_up = rsi[future_returns > up_move_threshold]
        
        if len(rsi_before_up) > 20:
            threshold = np.percentile(rsi_before_up.dropna(), 25)
        else:
            threshold = np.percentile(rsi.dropna(), 10)
        
        return CalibratedParameter(
            name="oversold_threshold",
            value=float(threshold),
            method=CalibrationMethod.DISTRIBUTION,
            confidence=len(rsi_before_up) / len(rsi) if len(rsi) > 0 else 0.5,
            valid_until=None,
            market_context={"sample_size": len(rsi_before_up)}
        )
    
    def _derive_volatility_scalar(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive volatility scaling factor from realized vs implied volatility ratio.
        
        This replaces hardcoded ATR multipliers with a market-derived scalar.
        """
        # Calculate realized volatility
        returns = df['close'].pct_change().dropna()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Estimate "implied" volatility from high-low range (Parkinson estimator)
        hl_ratio = np.log(df['high'] / df['low'])
        parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * (hl_ratio ** 2).rolling(20).mean()) * np.sqrt(252)
        
        # Scalar is ratio of recent to long-term volatility
        recent_vol = realized_vol.iloc[-20:].mean()
        long_term_vol = realized_vol.mean()
        
        if long_term_vol > 0:
            scalar = recent_vol / long_term_vol
        else:
            scalar = 1.0
        
        # Bound to prevent extreme values
        scalar = np.clip(scalar, 0.5, 2.0)
        
        return CalibratedParameter(
            name="volatility_scalar",
            value=float(scalar),
            method=CalibrationMethod.VOLATILITY,
            confidence=0.8 if 0.7 < scalar < 1.5 else 0.5,
            valid_until=None,
            market_context={
                "realized_vol": float(recent_vol),
                "long_term_vol": float(long_term_vol)
            }
        )
    
    def _derive_halflife(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive mean-reversion half-life using Ornstein-Uhlenbeck estimation.
        
        This is critical for mean-reversion strategies to know how long
        to hold positions.
        """
        close = df['close'].values
        
        # Fit AR(1) model to log prices
        log_prices = np.log(close)
        y = log_prices[1:]
        x = log_prices[:-1]
        
        # OLS regression
        n = len(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        
        # Half-life from beta
        if beta < 1 and beta > 0:
            halflife = -np.log(2) / np.log(beta)
        else:
            # Not mean-reverting, use volatility-based estimate
            halflife = len(close) * 0.1
        
        # Bound to reasonable range
        halflife = np.clip(halflife, 1, len(close) * 0.5)
        
        return CalibratedParameter(
            name="mean_reversion_halflife",
            value=float(halflife),
            method=CalibrationMethod.STATISTICAL,
            confidence=1.0 - abs(beta) if beta < 1 else 0.3,
            valid_until=None,
            market_context={"ar1_coefficient": float(beta)}
        )
    
    def _derive_kelly_fraction(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive optimal risk fraction using Kelly Criterion on historical performance.
        
        This replaces hardcoded "risk X% per trade" with mathematically optimal sizing.
        """
        returns = df['close'].pct_change().dropna()
        
        # Separate winning and losing periods
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            kelly = 0.01  # Minimum
        else:
            # Win probability
            p = len(wins) / len(returns)
            
            # Average win/loss ratio
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            
            if avg_loss > 0:
                b = avg_win / avg_loss  # Win/loss ratio
                
                # Kelly formula
                kelly = (p * b - (1 - p)) / b
            else:
                kelly = 0.01
        
        # Use half-Kelly for safety (common practice)
        kelly = kelly / 2
        
        # Bound to reasonable range (never risk more than 2% even if Kelly says so)
        # This is a GUARDIAN limit, not a strategy parameter
        kelly = np.clip(kelly, 0.001, 0.02)
        
        return CalibratedParameter(
            name="optimal_risk_fraction",
            value=float(kelly),
            method=CalibrationMethod.STATISTICAL,
            confidence=min(1.0, len(returns) / 1000),  # More data = more confidence
            valid_until=None,
            market_context={
                "win_rate": float(p) if 'p' in dir() else 0.5,
                "win_loss_ratio": float(b) if 'b' in dir() else 1.0
            }
        )
    
    def _derive_sl_multiple(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive stop-loss ATR multiple from adverse excursion analysis.
        
        Analyze how far losing trades typically go against us before recovering
        or being stopped out. The optimal SL is the point where recovery
        probability drops significantly.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate ATR
        atr = self._calculate_atr(df, period=14)
        
        # Maximum adverse excursion for each bar (how far price moved against entry)
        # Assuming long entries at close
        mae_long = (close[:-1] - np.minimum.accumulate(low[1:])) / atr[:-1]
        
        # Find the excursion level where most losing trades could have been saved
        # (i.e., price eventually came back above this level)
        recovery_threshold = np.percentile(mae_long[mae_long > 0], 75)
        
        # Bound to reasonable range (market-derived bounds based on volatility regime)
        vol_regime = self._derive_volatility_scalar(df).value
        min_sl = 1.0 * vol_regime
        max_sl = 4.0 * vol_regime
        
        sl_multiple = np.clip(recovery_threshold, min_sl, max_sl)
        
        return CalibratedParameter(
            name="stop_loss_atr_multiple",
            value=float(sl_multiple),
            method=CalibrationMethod.STATISTICAL,
            confidence=0.7,
            valid_until=None,
            market_context={"recovery_threshold_pct": float(recovery_threshold)}
        )
    
    def _derive_tp_multiple(self, df: pd.DataFrame) -> CalibratedParameter:
        """
        Derive take-profit ATR multiple from favorable excursion analysis.
        
        Analyze how far winning trades typically go in our favor. The optimal TP
        is where expected value (probability * magnitude) is maximized.
        """
        high = df['high'].values
        close = df['close'].values
        
        atr = self._calculate_atr(df, period=14)
        
        # Maximum favorable excursion (how far price moved in our favor)
        mfe_long = (np.maximum.accumulate(high[1:]) - close[:-1]) / atr[:-1]
        
        # Find optimal TP level that maximizes expected value
        # Test different TP levels and compute expected return
        best_ev = 0
        best_tp = 2.0
        
        for tp_level in np.arange(1.0, 5.0, 0.25):
            # Probability of reaching this level
            prob_reach = np.mean(mfe_long >= tp_level)
            # Expected value at this level
            ev = prob_reach * tp_level
            
            if ev > best_ev:
                best_ev = ev
                best_tp = tp_level
        
        return CalibratedParameter(
            name="take_profit_atr_multiple",
            value=float(best_tp),
            method=CalibrationMethod.STATISTICAL,
            confidence=best_ev / best_tp if best_tp > 0 else 0.5,
            valid_until=None,
            market_context={"optimal_ev": float(best_ev)}
        )
    
    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI without hardcoded parameters."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate ATR."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = pd.Series(tr).rolling(period).mean().values
        return atr
    
    def _derive_signal_period(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive signal line period (for MACD-like indicators)."""
        # Use entropy to find optimal smoothing
        close = df['close'].values
        period = optimal_lookback_from_entropy(close, min_period=3, max_period=20)
        
        return CalibratedParameter(
            name="signal_period",
            value=float(period),
            method=CalibrationMethod.ENTROPY,
            confidence=0.7,
            valid_until=None,
            market_context={}
        )
    
    def _derive_atr_period(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive ATR period from volatility clustering."""
        returns = df['close'].pct_change().dropna().values
        
        # Find volatility clustering period using autocorrelation of squared returns
        sq_returns = returns ** 2
        acf = np.correlate(sq_returns - np.mean(sq_returns), 
                          sq_returns - np.mean(sq_returns), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        
        # Find first significant drop in autocorrelation
        period = np.argmax(acf < 0.5) if np.any(acf < 0.5) else 14
        period = max(5, min(50, period))
        
        return CalibratedParameter(
            name="atr_period",
            value=float(period),
            method=CalibrationMethod.VOLATILITY,
            confidence=0.7,
            valid_until=None,
            market_context={}
        )
    
    def _derive_mr_threshold(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive mean-reversion entry threshold from z-score distribution."""
        close = df['close'].values
        
        # Calculate z-scores
        mean = pd.Series(close).rolling(50).mean()
        std = pd.Series(close).rolling(50).std()
        zscore = (close - mean) / std
        zscore = zscore.dropna()
        
        # Find z-score level that historically preceded reversals
        future_returns = pd.Series(close).pct_change(10).shift(-10)
        
        # For z > 0 (price above mean), look for subsequent down moves
        high_z = zscore[zscore > 1]
        if len(high_z) > 20:
            high_z_returns = future_returns.iloc[high_z.index]
            reversal_rate = (high_z_returns < 0).mean()
            
            # Find optimal z threshold
            best_z = 2.0
            best_edge = 0
            
            for z in np.arange(1.0, 3.5, 0.1):
                mask = zscore > z
                if mask.sum() > 10:
                    returns_at_z = future_returns[mask]
                    edge = -(returns_at_z.mean())  # Negative because we'd short
                    if edge > best_edge:
                        best_edge = edge
                        best_z = z
        else:
            best_z = 2.0
        
        return CalibratedParameter(
            name="mean_reversion_threshold",
            value=float(best_z),
            method=CalibrationMethod.DISTRIBUTION,
            confidence=0.6,
            valid_until=None,
            market_context={}
        )
    
    def _derive_breakout_threshold(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive breakout detection threshold from range analysis."""
        high = df['high'].values
        low = df['low'].values
        
        # Calculate rolling range as percentage
        range_pct = (high - low) / low
        
        # Breakout threshold is the 90th percentile of range
        threshold = np.percentile(range_pct, 90)
        
        return CalibratedParameter(
            name="breakout_threshold",
            value=float(threshold),
            method=CalibrationMethod.DISTRIBUTION,
            confidence=0.7,
            valid_until=None,
            market_context={}
        )
    
    def _derive_consolidation_threshold(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive consolidation detection threshold."""
        high = df['high'].values
        low = df['low'].values
        
        range_pct = (high - low) / low
        
        # Consolidation threshold is 25th percentile of range
        threshold = np.percentile(range_pct, 25)
        
        return CalibratedParameter(
            name="consolidation_threshold",
            value=float(threshold),
            method=CalibrationMethod.DISTRIBUTION,
            confidence=0.7,
            valid_until=None,
            market_context={}
        )
    
    def _derive_regime_lookback(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive optimal lookback for regime detection."""
        close = df['close'].values
        
        # Use Hurst exponent stability to find regime lookback
        hurst_values = []
        lookbacks = range(50, min(500, len(close) // 2), 25)
        
        for lookback in lookbacks:
            H = hurst_exponent(close[-lookback:])
            hurst_values.append(H)
        
        # Find lookback where Hurst is most stable
        hurst_std = pd.Series(hurst_values).rolling(3).std()
        if len(hurst_std.dropna()) > 0:
            best_idx = hurst_std.dropna().idxmin()
            best_lookback = list(lookbacks)[best_idx]
        else:
            best_lookback = 100
        
        return CalibratedParameter(
            name="regime_lookback",
            value=float(best_lookback),
            method=CalibrationMethod.REGIME,
            confidence=0.6,
            valid_until=None,
            market_context={}
        )
    
    def _derive_trend_threshold(self, df: pd.DataFrame) -> CalibratedParameter:
        """Derive trend strength threshold from fractal dimension."""
        close = df['close'].values
        
        # Calculate fractal dimension
        fd = fractal_dimension(close)
        
        # FD = 1.5 is random walk, < 1.5 is trending, > 1.5 is mean-reverting
        # Trend threshold is how far from 1.5 we need to be
        threshold = abs(1.5 - fd) * 2  # Scale to reasonable range
        threshold = np.clip(threshold, 0.1, 0.5)
        
        return CalibratedParameter(
            name="trend_strength_threshold",
            value=float(threshold),
            method=CalibrationMethod.FRACTAL,
            confidence=0.7,
            valid_until=None,
            market_context={"fractal_dimension": float(fd)}
        )
    
    def _compute_spectral_confidence(
        self, 
        psd: np.ndarray, 
        peak_indices: np.ndarray
    ) -> float:
        """Compute confidence in spectral analysis based on peak clarity."""
        if len(peak_indices) == 0:
            return 0.3
        
        # Confidence based on how much power is concentrated in peaks
        peak_power = np.sum(psd[peak_indices])
        total_power = np.sum(psd)
        
        if total_power > 0:
            concentration = peak_power / total_power
        else:
            concentration = 0.5
        
        return float(np.clip(concentration, 0.3, 1.0))
```

### 4.2 Entropy and Spectral Analysis Modules

```python
"""
core/lossless/entropy.py

Information-theoretic approach to parameter derivation.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Tuple


def market_entropy(prices: np.ndarray, bins: int = 50) -> float:
    """
    Calculate Shannon entropy of price distribution.
    
    Higher entropy = more random/unpredictable
    Lower entropy = more structured/predictable
    """
    returns = np.diff(np.log(prices))
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros for log
    return scipy_entropy(hist)


def optimal_lookback_from_entropy(
    prices: np.ndarray,
    min_period: int = 5,
    max_period: int = 200
) -> int:
    """
    Find optimal lookback period by minimizing prediction entropy.
    
    The optimal period is where past data best predicts future data,
    measured by conditional entropy.
    """
    returns = np.diff(np.log(prices))
    
    best_period = min_period
    min_entropy = float('inf')
    
    for period in range(min_period, min(max_period, len(returns) // 3)):
        # Split into past (conditioning) and future (prediction)
        past = returns[:period]
        future = returns[period:period*2] if len(returns) >= period*2 else returns[period:]
        
        if len(future) < 10:
            continue
        
        # Calculate conditional entropy H(future|past)
        # Approximated by joint entropy - past entropy
        joint = np.concatenate([past, future])
        H_joint = market_entropy(np.exp(np.cumsum(joint)))
        H_past = market_entropy(np.exp(np.cumsum(past)))
        
        conditional_entropy = H_joint - H_past
        
        if conditional_entropy < min_entropy:
            min_entropy = conditional_entropy
            best_period = period
    
    return best_period


def sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r: float = 0.2
) -> float:
    """
    Calculate Sample Entropy (regularity measure).
    
    Lower SampEn = more regular/predictable
    Higher SampEn = more random
    """
    N = len(data)
    r = r * np.std(data)  # Tolerance as fraction of std
    
    def _count_matches(template_length):
        count = 0
        for i in range(N - template_length):
            for j in range(i + 1, N - template_length):
                if np.max(np.abs(data[i:i+template_length] - data[j:j+template_length])) < r:
                    count += 1
        return count
    
    A = _count_matches(m + 1)
    B = _count_matches(m)
    
    if B == 0:
        return 0.0
    
    return -np.log(A / B) if A > 0 else 0.0
```

```python
"""
core/lossless/spectral.py

Spectral analysis for cycle detection and period derivation.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple


def spectral_density(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density of price series.
    
    Returns:
        freqs: Frequency array
        psd: Power spectral density
    """
    # Detrend to focus on cycles
    detrended = signal.detrend(prices)
    
    # Apply window to reduce spectral leakage
    window = signal.windows.hann(len(detrended))
    windowed = detrended * window
    
    # Compute FFT
    n = len(windowed)
    yf = fft(windowed)
    xf = fftfreq(n, 1)[:n//2]
    
    psd = 2.0/n * np.abs(yf[0:n//2])
    
    return xf, psd


def dominant_cycle_period(prices: np.ndarray) -> int:
    """
    Find the dominant cycle period in the price series.
    """
    freqs, psd = spectral_density(prices)
    
    # Find peak (excluding DC component)
    psd[0] = 0
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    
    if peak_freq > 0:
        return int(1 / peak_freq)
    else:
        return len(prices) // 4  # Default to quarter of data


def find_all_cycles(
    prices: np.ndarray,
    min_period: int = 5,
    significance_threshold: float = 0.05
) -> list:
    """
    Find all significant cycles in the price series.
    
    Returns list of (period, power) tuples.
    """
    freqs, psd = spectral_density(prices)
    
    # Find peaks
    peaks, properties = signal.find_peaks(psd, prominence=np.std(psd))
    
    cycles = []
    total_power = np.sum(psd)
    
    for peak in peaks:
        freq = freqs[peak]
        power = psd[peak] / total_power
        
        if freq > 0 and power > significance_threshold:
            period = int(1 / freq)
            if period >= min_period:
                cycles.append((period, power))
    
    return sorted(cycles, key=lambda x: x[1], reverse=True)
```

```python
"""
core/lossless/fractal.py

Fractal analysis for regime detection.
"""

import numpy as np


def fractal_dimension(prices: np.ndarray) -> float:
    """
    Calculate fractal dimension using box-counting method.
    
    FD interpretation:
    - FD ≈ 1.0: Straight line (strong trend)
    - FD ≈ 1.5: Random walk
    - FD ≈ 2.0: Space-filling (choppy, mean-reverting)
    """
    # Normalize prices to [0, 1]
    normalized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)
    
    # Box counting at different scales
    scales = []
    counts = []
    
    for k in range(1, 10):
        box_size = 1.0 / (2 ** k)
        if box_size * len(normalized) < 1:
            break
        
        # Count boxes
        time_boxes = int(np.ceil(len(normalized) * box_size))
        price_boxes = int(np.ceil(1.0 / box_size))
        
        grid = np.zeros((time_boxes, price_boxes), dtype=bool)
        
        for i, p in enumerate(normalized):
            t_idx = min(int(i * box_size), time_boxes - 1)
            p_idx = min(int(p / box_size), price_boxes - 1)
            grid[t_idx, p_idx] = True
        
        scales.append(box_size)
        counts.append(np.sum(grid))
    
    # Linear regression on log-log plot
    if len(scales) > 2:
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        # Slope is the fractal dimension
        slope, _ = np.polyfit(log_scales, log_counts, 1)
        return -slope
    
    return 1.5  # Default to random walk


def box_counting_dimension(prices: np.ndarray, num_scales: int = 10) -> float:
    """
    Alternative box-counting implementation.
    """
    return fractal_dimension(prices)
```

```python
"""
core/lossless/hurst.py

Hurst exponent for regime classification.
"""

import numpy as np


def hurst_exponent(prices: np.ndarray) -> float:
    """
    Calculate Hurst exponent using R/S analysis.
    
    Interpretation:
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk
    - H > 0.5: Trending (persistent)
    """
    n = len(prices)
    if n < 20:
        return 0.5
    
    # Calculate returns
    returns = np.diff(np.log(prices))
    
    # R/S analysis for different time scales
    max_k = min(int(np.log2(n)) - 1, 8)
    rs_values = []
    ns = []
    
    for k in range(2, max_k + 1):
        subset_size = int(2 ** k)
        num_subsets = len(returns) // subset_size
        
        if num_subsets == 0:
            continue
        
        rs_list = []
        
        for i in range(num_subsets):
            subset = returns[i * subset_size:(i + 1) * subset_size]
            
            # Mean-adjusted cumulative sum
            mean_adj = subset - np.mean(subset)
            cumsum = np.cumsum(mean_adj)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(subset, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if len(rs_list) > 0:
            rs_values.append(np.mean(rs_list))
            ns.append(subset_size)
    
    # Linear regression on log-log plot
    if len(rs_values) > 2:
        log_n = np.log(ns)
        log_rs = np.log(rs_values)
        
        H, _ = np.polyfit(log_n, log_rs, 1)
        return np.clip(H, 0, 1)
    
    return 0.5


def regime_from_hurst(H: float) -> str:
    """
    Classify market regime from Hurst exponent.
    """
    if H < 0.4:
        return "strong_mean_reversion"
    elif H < 0.5:
        return "mild_mean_reversion"
    elif H < 0.55:
        return "random_walk"
    elif H < 0.65:
        return "mild_trending"
    else:
        return "strong_trending"
```

### 4.3 Multi-Model Ensemble System

```python
"""
models/ensemble/stacking.py

Meta-learner that combines predictions from multiple diverse models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import deque

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class ModelPrediction:
    """Standardized prediction output from any model."""
    model_name: str
    direction: float      # -1 to 1 (short to long)
    magnitude: float      # Expected % move
    confidence: float     # 0 to 1
    timestamp: pd.Timestamp
    features_used: List[str]
    model_state: Dict[str, Any]


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble."""
    direction: float
    magnitude: float
    confidence: float
    action: str           # "long", "short", "neutral"
    position_size_scalar: float  # 0 to 1
    model_agreement: float  # 0 to 1
    contributing_models: List[str]
    disagreeing_models: List[str]
    timestamp: pd.Timestamp


class BaseModel(ABC):
    """Abstract base class for all trading models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Generate prediction."""
        pass
    
    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Online update with new data."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        pass


class EnsembleMetaLearner:
    """
    Stacking meta-learner that combines diverse model predictions.
    
    Features:
    - Dynamic weight adjustment based on recent performance
    - Disagreement-aware position sizing
    - Regime-conditional model selection
    - Online learning of optimal weights
    """
    
    def __init__(
        self,
        models: Dict[str, BaseModel],
        min_confidence: float = 0.5,
        agreement_threshold: float = 0.6,
        weight_decay: float = 0.95,
        performance_window: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        self.models = models
        self.min_confidence = min_confidence
        self.agreement_threshold = agreement_threshold
        self.weight_decay = weight_decay
        self.performance_window = performance_window
        self.logger = logger or logging.getLogger(__name__)
        
        # Model weights (start equal)
        self.weights = {name: 1.0 / len(models) for name in models}
        
        # Performance tracking
        self.prediction_history: deque = deque(maxlen=performance_window)
        self.actual_returns: deque = deque(maxlen=performance_window)
        
        # Meta-learner (learns optimal combination)
        self.meta_model = Ridge(alpha=1.0)
        self.meta_scaler = StandardScaler()
        self.meta_trained = False
    
    def predict(self, features: Dict[str, np.ndarray]) -> EnsemblePrediction:
        """
        Generate ensemble prediction from all models.
        
        Args:
            features: Dictionary of feature arrays for each model type
            
        Returns:
            Combined prediction with confidence and position sizing
        """
        predictions = []
        
        # Collect predictions from all models
        for name, model in self.models.items():
            try:
                if name in features:
                    pred = model.predict(features[name])
                    predictions.append((name, pred))
            except Exception as e:
                self.logger.error(f"Model {name} prediction failed: {e}")
        
        if not predictions:
            return self._neutral_prediction()
        
        # Calculate weighted ensemble
        weighted_direction = 0.0
        weighted_magnitude = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for name, pred in predictions:
            w = self.weights[name] * pred.confidence
            weighted_direction += w * pred.direction
            weighted_magnitude += w * pred.magnitude
            weighted_confidence += w * pred.confidence
            total_weight += w
        
        if total_weight > 0:
            ensemble_direction = weighted_direction / total_weight
            ensemble_magnitude = weighted_magnitude / total_weight
            ensemble_confidence = weighted_confidence / total_weight
        else:
            return self._neutral_prediction()
        
        # Calculate model agreement
        directions = [pred.direction for _, pred in predictions]
        agreement = self._calculate_agreement(directions)
        
        # Determine action
        action, contributing, disagreeing = self._determine_action(
            predictions, ensemble_direction, agreement
        )
        
        # Position size scalar based on agreement and confidence
        position_scalar = self._calculate_position_scalar(
            ensemble_confidence, agreement
        )
        
        return EnsemblePrediction(
            direction=ensemble_direction,
            magnitude=ensemble_magnitude,
            confidence=ensemble_confidence,
            action=action,
            position_size_scalar=position_scalar,
            model_agreement=agreement,
            contributing_models=contributing,
            disagreeing_models=disagreeing,
            timestamp=pd.Timestamp.now()
        )
    
    def update_weights(self, prediction: EnsemblePrediction, actual_return: float):
        """
        Update model weights based on realized performance.
        
        Uses gradient descent on prediction error.
        """
        # Store for batch learning
        self.actual_returns.append(actual_return)
        
        # Apply weight decay
        for name in self.weights:
            self.weights[name] *= self.weight_decay
        
        # Update weights based on direction correctness
        correct_direction = (prediction.direction > 0 and actual_return > 0) or \
                          (prediction.direction < 0 and actual_return < 0)
        
        # Increase weights for models that agreed with correct direction
        if correct_direction:
            for name in prediction.contributing_models:
                self.weights[name] += 0.05 * abs(actual_return)
        else:
            for name in prediction.disagreeing_models:
                self.weights[name] += 0.03 * abs(actual_return)
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total
        
        # Retrain meta-learner periodically
        if len(self.actual_returns) >= 50 and len(self.actual_returns) % 10 == 0:
            self._train_meta_learner()
    
    def _calculate_agreement(self, directions: List[float]) -> float:
        """
        Calculate agreement among models.
        
        Returns 1.0 if all agree, 0.0 if evenly split.
        """
        if len(directions) == 0:
            return 0.0
        
        # Count positive vs negative
        positive = sum(1 for d in directions if d > 0)
        negative = sum(1 for d in directions if d < 0)
        total = len(directions)
        
        if total == 0:
            return 0.0
        
        majority = max(positive, negative)
        return majority / total
    
    def _determine_action(
        self,
        predictions: List[Tuple[str, ModelPrediction]],
        ensemble_direction: float,
        agreement: float
    ) -> Tuple[str, List[str], List[str]]:
        """
        Determine final action and categorize models.
        """
        if agreement < self.agreement_threshold:
            return "neutral", [], []
        
        if ensemble_direction > 0.1:
            action = "long"
        elif ensemble_direction < -0.1:
            action = "short"
        else:
            action = "neutral"
        
        contributing = []
        disagreeing = []
        
        for name, pred in predictions:
            if (action == "long" and pred.direction > 0) or \
               (action == "short" and pred.direction < 0):
                contributing.append(name)
            elif action != "neutral":
                disagreeing.append(name)
        
        return action, contributing, disagreeing
    
    def _calculate_position_scalar(
        self,
        confidence: float,
        agreement: float
    ) -> float:
        """
        Calculate position size scalar based on conviction.
        
        High confidence + high agreement = full size
        Low confidence or disagreement = reduced size
        """
        if confidence < self.min_confidence:
            return 0.0
        
        # Base scalar from confidence
        base = (confidence - self.min_confidence) / (1 - self.min_confidence)
        
        # Reduce for disagreement
        agreement_factor = (agreement - 0.5) / 0.5  # 0 at 50%, 1 at 100%
        agreement_factor = max(0, agreement_factor)
        
        return base * agreement_factor
    
    def _neutral_prediction(self) -> EnsemblePrediction:
        """Return neutral prediction when no signals."""
        return EnsemblePrediction(
            direction=0.0,
            magnitude=0.0,
            confidence=0.0,
            action="neutral",
            position_size_scalar=0.0,
            model_agreement=0.0,
            contributing_models=[],
            disagreeing_models=[],
            timestamp=pd.Timestamp.now()
        )
    
    def _train_meta_learner(self):
        """Train stacking meta-learner on historical predictions."""
        if len(self.prediction_history) < 30:
            return
        
        # Prepare training data
        X = np.array(self.prediction_history)
        y = np.array(self.actual_returns)
        
        # Standardize
        X_scaled = self.meta_scaler.fit_transform(X)
        
        # Train
        self.meta_model.fit(X_scaled, y)
        self.meta_trained = True
        
        self.logger.info("Meta-learner retrained on {} samples".format(len(y)))
```

---

## PART V: EXECUTION & RISK ENGINE

### 5.1 Institutional-Grade Execution Algorithm

```python
"""
core/execution/execution_algo.py

Smart order execution to minimize market impact and slippage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5
import time
import logging


class ExecutionStyle(Enum):
    MARKET = "market"           # Immediate execution
    TWAP = "twap"               # Time-weighted average price
    VWAP = "vwap"               # Volume-weighted average price
    ICEBERG = "iceberg"         # Hidden size
    ADAPTIVE = "adaptive"       # Adjust based on conditions


@dataclass
class ExecutionPlan:
    """Plan for executing a position."""
    symbol: str
    direction: str              # "buy" or "sell"
    total_size: float           # Total lots
    style: ExecutionStyle
    slices: List[Dict]          # List of order slices
    max_duration_seconds: int
    urgency: float              # 0-1, higher = faster execution
    stop_loss: float
    take_profit: float


@dataclass
class ExecutionResult:
    """Result of an execution."""
    success: bool
    avg_fill_price: float
    total_filled: float
    slippage_pips: float
    execution_time_seconds: float
    num_fills: int
    market_impact_estimate: float


class SmartExecutor:
    """
    Institutional-grade order execution engine.
    
    Features:
    - TWAP/VWAP execution for large orders
    - Iceberg orders to hide size
    - Slippage prediction and monitoring
    - Market impact estimation
    - Adaptive execution speed
    """
    
    def __init__(
        self,
        max_market_impact_pct: float = 0.01,  # 1 pip max impact
        min_slice_lots: float = 0.01,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        self.max_impact = max_market_impact_pct
        self.min_slice = min_slice_lots
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)
        
        # Track execution quality
        self.execution_history: List[ExecutionResult] = []
    
    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        Execute a trading plan with smart order slicing.
        """
        start_time = time.time()
        
        if plan.style == ExecutionStyle.MARKET:
            return self._execute_market(plan)
        elif plan.style == ExecutionStyle.TWAP:
            return self._execute_twap(plan)
        elif plan.style == ExecutionStyle.VWAP:
            return self._execute_vwap(plan)
        elif plan.style == ExecutionStyle.ICEBERG:
            return self._execute_iceberg(plan)
        elif plan.style == ExecutionStyle.ADAPTIVE:
            return self._execute_adaptive(plan)
        else:
            return self._execute_market(plan)
    
    def create_execution_plan(
        self,
        symbol: str,
        direction: str,
        size: float,
        stop_loss: float,
        take_profit: float,
        urgency: float = 0.5,
        market_conditions: Optional[Dict] = None
    ) -> ExecutionPlan:
        """
        Create optimal execution plan based on size and conditions.
        """
        # Get market info
        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        
        if not symbol_info or not tick:
            raise ValueError(f"Cannot get info for {symbol}")
        
        # Estimate market impact
        avg_volume = self._get_average_volume(symbol)
        impact_ratio = size / avg_volume if avg_volume > 0 else 0
        
        # Determine execution style
        if size <= self.min_slice * 5:
            # Small order - just execute
            style = ExecutionStyle.MARKET
            slices = [{"size": size, "delay": 0}]
        elif impact_ratio > 0.1:
            # Large relative to volume - use TWAP
            style = ExecutionStyle.TWAP
            num_slices = min(10, int(size / self.min_slice))
            slice_size = size / num_slices
            interval = 60  # 1 minute between slices
            slices = [
                {"size": slice_size, "delay": i * interval}
                for i in range(num_slices)
            ]
        elif urgency < 0.3:
            # Low urgency - use VWAP
            style = ExecutionStyle.VWAP
            slices = self._create_vwap_slices(symbol, size)
        else:
            # Default - iceberg
            style = ExecutionStyle.ICEBERG
            show_size = max(self.min_slice, size * 0.2)
            slices = [{"size": size, "show_size": show_size, "delay": 0}]
        
        return ExecutionPlan(
            symbol=symbol,
            direction=direction,
            total_size=size,
            style=style,
            slices=slices,
            max_duration_seconds=int(300 * (1 - urgency)),
            urgency=urgency,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def _execute_market(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute immediate market order."""
        start_time = time.time()
        
        tick = mt5.symbol_info_tick(plan.symbol)
        entry_price = tick.ask if plan.direction == "buy" else tick.bid
        
        order_type = mt5.ORDER_TYPE_BUY if plan.direction == "buy" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": plan.symbol,
            "volume": plan.total_size,
            "type": order_type,
            "price": entry_price,
            "sl": plan.stop_loss,
            "tp": plan.take_profit,
            "deviation": 20,
            "magic": 999999,
            "comment": "smart_exec",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            slippage = abs(result.price - entry_price) / mt5.symbol_info(plan.symbol).point
            
            exec_result = ExecutionResult(
                success=True,
                avg_fill_price=result.price,
                total_filled=plan.total_size,
                slippage_pips=slippage,
                execution_time_seconds=time.time() - start_time,
                num_fills=1,
                market_impact_estimate=slippage * 0.0001  # Rough estimate
            )
        else:
            exec_result = ExecutionResult(
                success=False,
                avg_fill_price=0,
                total_filled=0,
                slippage_pips=0,
                execution_time_seconds=time.time() - start_time,
                num_fills=0,
                market_impact_estimate=0
            )
        
        self.execution_history.append(exec_result)
        return exec_result
    
    def _execute_twap(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute using Time-Weighted Average Price strategy."""
        start_time = time.time()
        fills = []
        total_filled = 0
        
        for slice_info in plan.slices:
            if time.time() - start_time > plan.max_duration_seconds:
                break
            
            # Wait for scheduled time
            time.sleep(slice_info["delay"])
            
            # Execute slice
            tick = mt5.symbol_info_tick(plan.symbol)
            price = tick.ask if plan.direction == "buy" else tick.bid
            
            order_type = mt5.ORDER_TYPE_BUY if plan.direction == "buy" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": plan.symbol,
                "volume": slice_info["size"],
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 999999,
                "comment": "twap_slice",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                fills.append((result.price, slice_info["size"]))
                total_filled += slice_info["size"]
        
        # Calculate VWAP of fills
        if fills:
            avg_price = sum(p * s for p, s in fills) / total_filled
            first_price = fills[0][0]
            slippage = abs(avg_price - first_price) / mt5.symbol_info(plan.symbol).point
        else:
            avg_price = 0
            slippage = 0
        
        # Set SL/TP on the aggregate position
        if total_filled > 0:
            self._set_position_sltp(plan.symbol, plan.stop_loss, plan.take_profit)
        
        return ExecutionResult(
            success=total_filled > 0,
            avg_fill_price=avg_price,
            total_filled=total_filled,
            slippage_pips=slippage,
            execution_time_seconds=time.time() - start_time,
            num_fills=len(fills),
            market_impact_estimate=slippage * 0.0001
        )
    
    def _execute_vwap(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute following volume profile for VWAP."""
        # Similar to TWAP but slice sizes based on expected volume
        return self._execute_twap(plan)  # Simplified for now
    
    def _execute_iceberg(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute with hidden size (show only partial)."""
        start_time = time.time()
        fills = []
        total_filled = 0
        remaining = plan.total_size
        
        show_size = plan.slices[0].get("show_size", plan.total_size * 0.2)
        
        while remaining > 0 and time.time() - start_time < plan.max_duration_seconds:
            slice_size = min(show_size, remaining)
            
            tick = mt5.symbol_info_tick(plan.symbol)
            price = tick.ask if plan.direction == "buy" else tick.bid
            
            order_type = mt5.ORDER_TYPE_BUY if plan.direction == "buy" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": plan.symbol,
                "volume": slice_size,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 999999,
                "comment": "iceberg",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                fills.append((result.price, slice_size))
                total_filled += slice_size
                remaining -= slice_size
            else:
                time.sleep(1)  # Brief pause before retry
        
        if fills:
            avg_price = sum(p * s for p, s in fills) / total_filled
            slippage = abs(fills[-1][0] - fills[0][0]) / mt5.symbol_info(plan.symbol).point
        else:
            avg_price = 0
            slippage = 0
        
        if total_filled > 0:
            self._set_position_sltp(plan.symbol, plan.stop_loss, plan.take_profit)
        
        return ExecutionResult(
            success=total_filled == plan.total_size,
            avg_fill_price=avg_price,
            total_filled=total_filled,
            slippage_pips=slippage,
            execution_time_seconds=time.time() - start_time,
            num_fills=len(fills),
            market_impact_estimate=slippage * 0.0001 * (plan.total_size / show_size)
        )
    
    def _execute_adaptive(self, plan: ExecutionPlan) -> ExecutionResult:
        """Adaptive execution that adjusts based on market conditions."""
        # Monitor spread and adjust execution speed
        # Use market orders when spread tight, limit orders when wide
        return self._execute_iceberg(plan)  # Simplified for now
    
    def _get_average_volume(self, symbol: str, period: int = 20) -> float:
        """Get average volume for sizing estimation."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period)
        if rates is not None and len(rates) > 0:
            return np.mean([r['tick_volume'] for r in rates])
        return 1000  # Default
    
    def _create_vwap_slices(self, symbol: str, total_size: float) -> List[Dict]:
        """Create slices weighted by expected volume profile."""
        # Get hourly volume profile
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
        
        if rates is None or len(rates) == 0:
            # Fallback to equal slices
            num_slices = 5
            return [{"size": total_size / num_slices, "delay": i * 60} for i in range(num_slices)]
        
        volumes = [r['tick_volume'] for r in rates]
        total_volume = sum(volumes)
        
        slices = []
        delay = 0
        
        for i, vol in enumerate(volumes[:5]):  # Next 5 hours
            if total_volume > 0:
                weight = vol / total_volume
                slice_size = total_size * weight * 5  # Normalize
                slices.append({"size": slice_size, "delay": delay})
                delay += 3600  # 1 hour
        
        return slices
    
    def _set_position_sltp(self, symbol: str, sl: float, tp: float):
        """Set SL/TP on open position."""
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": sl,
                    "tp": tp,
                }
                mt5.order_send(request)
```

---

## PART VI: OUTPUT REQUIREMENTS

For each bot file, you must produce:

### 6.1 Standalone Bot Structure

Each of the four bot files must be completely self-contained and include:

1. **All imports** (no external dependencies on project modules - copy everything inline)
2. **Complete configuration section** with credentials placeholders
3. **All class implementations** (Connector, Risk, ML, Signals, Execution, Telegram)
4. **Main trading loop** with:
   - Market data collection
   - Feature engineering
   - Signal generation from ensemble
   - Risk validation
   - Position sizing
   - Smart execution
   - Position management
   - State persistence
5. **Backtesting mode** toggle
6. **Comprehensive logging**
7. **Error handling and recovery**

### 6.2 File Naming

```
bots/gft_bot_account1.py    # ~3000-4000 lines
bots/gft_bot_account2.py    # ~3000-4000 lines  
bots/gft_bot_account3.py    # ~3000-4000 lines
bots/the5ers_bot.py         # ~3000-4000 lines
```

### 6.3 Code Quality Requirements

- **Type hints** on all functions
- **Docstrings** explaining purpose, parameters, returns
- **Comments** for complex logic
- **No magic numbers** - all parameters derived from market data
- **Defensive programming** - validate all inputs
- **Idempotent operations** - safe to restart anytime
- **Atomic state updates** - no partial writes

### 6.4 Testing Requirements

Include inline test functions that can be run with:
```bash
python gft_bot_account1.py --test
```

---

## PART VII: CRITICAL REMINDERS

### ABSOLUTE RULES (NEVER VIOLATE)

1. **Prop firm rules are INVIOLABLE** - No trade that risks breaching limits
2. **Guardian limits** - Always maintain buffer from actual limits
3. **No magic numbers** - Derive everything from market data
4. **Self-healing** - Auto-recover from any failure
5. **State persistence** - Never lose track of positions or P&L
6. **Graceful degradation** - If one model fails, others continue
7. **Logging everything** - Full audit trail of all decisions

### PERFORMANCE TARGETS

| Metric | Target | Guardian Limit |
|--------|--------|----------------|
| Annual Return | >40% | N/A |
| Sharpe Ratio | >2.0 | >1.5 |
| Max Drawdown | <6% | <7% (GFT), <8% (5ers) |
| Win Rate | >55% | >50% |
| Profit Factor | >1.5 | >1.2 |
| Daily Loss | N/A | <4% (5ers only) |

### MODEL HIERARCHY

When models disagree:
1. **Risk models always win** - Never override risk limits
2. **Regime detection second** - Don't fight the regime
3. **Ensemble signals third** - Combined wisdom
4. **Individual models last** - Lowest priority

---

## BEGIN IMPLEMENTATION

Start with `gft_bot_account1.py`. Build it as the master template, then create variants for accounts 2 and 3. Finally, create `the5ers_bot.py` with forex-specific adjustments.

Remember: This system should be able to run profitably for 12+ years without manual intervention, just like Renaissance's Medallion Fund.
# ADDENDUM: ADVANCED INSTITUTIONAL COMPONENTS

## PART VIII: ALTERNATIVE DATA INTEGRATION

### 8.1 Crypto-Specific Data Sources (For GFT Accounts)

```python
"""
data/external/crypto_alternative.py

Alternative data sources for crypto alpha generation.
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging


class CryptoAlternativeData:
    """
    Aggregates alternative data sources for crypto trading.
    
    Data Sources:
    1. Funding Rates (Coinglass/Binance) - Crowded positioning indicator
    2. Open Interest - Leverage buildup
    3. Liquidation Data - Forced selling/buying
    4. Exchange Flows - Smart money movement
    5. Whale Transactions - Large holder activity
    6. Fear & Greed Index - Sentiment
    7. Social Volume - Attention metrics
    """
    
    # Free API endpoints
    ENDPOINTS = {
        "fear_greed": "https://api.alternative.me/fng/",
        "coinglass_funding": "https://open-api.coinglass.com/public/v2/funding",
        "coinglass_oi": "https://open-api.coinglass.com/public/v2/open_interest",
        "binance_funding": "https://fapi.binance.com/fapi/v1/fundingRate",
        "binance_oi": "https://fapi.binance.com/fapi/v1/openInterest",
        "glassnode_base": "https://api.glassnode.com/v1/metrics/",
    }
    
    def __init__(
        self,
        coinglass_api_key: Optional[str] = None,
        glassnode_api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.coinglass_key = coinglass_api_key
        self.glassnode_key = glassnode_api_key
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache to avoid excessive API calls
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def get_all_signals(self, symbol: str = "BTC") -> Dict[str, float]:
        """
        Fetch all alternative data and compute signals.
        
        Returns dict of signal names to values (-1 to 1 scale).
        """
        signals = {}
        
        # Funding rate signal
        funding = await self.get_funding_rate(symbol)
        if funding is not None:
            # Extreme negative funding = long signal (shorts paying longs)
            # Extreme positive funding = short signal (longs paying shorts)
            signals["funding_signal"] = -self._normalize_funding(funding)
        
        # Fear & Greed signal
        fg = await self.get_fear_greed()
        if fg is not None:
            # Extreme fear = long signal (contrarian)
            # Extreme greed = short signal (contrarian)
            signals["sentiment_signal"] = (50 - fg) / 50
        
        # Open Interest change signal
        oi_change = await self.get_oi_change(symbol)
        if oi_change is not None:
            # Rising OI with price up = trend confirmation
            # Rising OI with price down = trend confirmation
            # Falling OI = profit taking/derisking
            signals["oi_signal"] = self._interpret_oi(oi_change)
        
        # Liquidation imbalance
        liqs = await self.get_liquidations(symbol)
        if liqs is not None:
            signals["liquidation_signal"] = liqs
        
        return signals
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate from Binance perpetuals.
        
        Funding rate interpretation:
        - Positive: Longs pay shorts (market bullish, crowded long)
        - Negative: Shorts pay longs (market bearish, crowded short)
        - Extreme values (>0.1% or <-0.1%) often precede reversals
        """
        cache_key = f"funding_{symbol}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]["value"]
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ENDPOINTS['binance_funding']}?symbol={symbol}USDT&limit=1"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            rate = float(data[0]["fundingRate"])
                            self._cache(cache_key, rate)
                            return rate
        except Exception as e:
            self.logger.error(f"Funding rate fetch failed: {e}")
        
        return None
    
    async def get_fear_greed(self) -> Optional[int]:
        """
        Get Fear & Greed Index (0-100).
        
        0-25: Extreme Fear (potential buying opportunity)
        25-45: Fear
        45-55: Neutral
        55-75: Greed
        75-100: Extreme Greed (potential selling opportunity)
        """
        cache_key = "fear_greed"
        if self._is_cached(cache_key):
            return self.cache[cache_key]["value"]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.ENDPOINTS["fear_greed"]) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        value = int(data["data"][0]["value"])
                        self._cache(cache_key, value)
                        return value
        except Exception as e:
            self.logger.error(f"Fear & Greed fetch failed: {e}")
        
        return None
    
    async def get_oi_change(self, symbol: str, hours: int = 24) -> Optional[float]:
        """
        Get Open Interest change over period.
        
        Returns percentage change.
        """
        cache_key = f"oi_{symbol}_{hours}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]["value"]
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ENDPOINTS['binance_oi']}?symbol={symbol}USDT"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        current_oi = float(data["openInterest"])
                        # Would need historical OI for change calculation
                        # Simplified: just return current level scaled
                        self._cache(cache_key, current_oi)
                        return current_oi
        except Exception as e:
            self.logger.error(f"OI fetch failed: {e}")
        
        return None
    
    async def get_liquidations(self, symbol: str) -> Optional[float]:
        """
        Get liquidation imbalance signal.
        
        Returns:
        - Positive: More shorts liquidated (bullish pressure)
        - Negative: More longs liquidated (bearish pressure)
        """
        # This would require a premium data source or websocket
        # Placeholder for implementation
        return None
    
    async def get_exchange_flows(self, symbol: str) -> Optional[Dict]:
        """
        Get exchange inflow/outflow data.
        
        Interpretation:
        - High inflows: Selling pressure incoming
        - High outflows: Accumulation (bullish)
        """
        if not self.glassnode_key:
            return None
        
        # Glassnode API implementation
        # Requires API key
        return None
    
    def _normalize_funding(self, rate: float) -> float:
        """
        Normalize funding rate to -1 to 1 scale.
        
        Typical range: -0.1% to 0.1% (8-hour rate)
        Extreme: beyond ±0.3%
        """
        # Clip to reasonable range and scale
        clipped = max(-0.003, min(0.003, rate))
        return clipped / 0.003
    
    def _interpret_oi(self, oi_value: float) -> float:
        """
        Interpret Open Interest for trading signal.
        """
        # Would need price context for full interpretation
        # Simplified: high OI = mean reversion signal
        # This is a placeholder
        return 0.0
    
    def _is_cached(self, key: str) -> bool:
        """Check if cache is valid."""
        if key not in self.cache:
            return False
        age = (datetime.now() - self.cache[key]["timestamp"]).total_seconds()
        return age < self.cache_ttl
    
    def _cache(self, key: str, value):
        """Store value in cache."""
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }


class ForexAlternativeData:
    """
    Alternative data sources for forex trading.
    
    Data Sources:
    1. COT Reports (Commitment of Traders) - Institutional positioning
    2. Central Bank Rate Expectations - Interest rate differentials
    3. Economic Surprise Index - Data vs expectations
    4. Risk-Off Indicators - VIX, credit spreads
    5. Intermarket Correlations - Equities, commodities
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    async def get_all_signals(self, pair: str) -> Dict[str, float]:
        """Get all forex alternative data signals."""
        signals = {}
        
        # Interest rate differential
        ird = await self.get_rate_differential(pair)
        if ird is not None:
            signals["carry_signal"] = self._normalize_carry(ird)
        
        # Risk sentiment
        risk = await self.get_risk_sentiment()
        if risk is not None:
            signals["risk_signal"] = risk
        
        return signals
    
    async def get_rate_differential(self, pair: str) -> Optional[float]:
        """
        Get interest rate differential for carry trade signal.
        """
        # Would need central bank rate data
        # Placeholder
        return None
    
    async def get_risk_sentiment(self) -> Optional[float]:
        """
        Get overall risk sentiment.
        
        Returns:
        - Positive: Risk-on (favor AUD, NZD, CAD)
        - Negative: Risk-off (favor JPY, CHF, USD)
        """
        # Would aggregate VIX, credit spreads, etc.
        return None
    
    def _normalize_carry(self, differential: float) -> float:
        """Normalize interest rate differential to signal."""
        # 2% differential = strong signal
        return max(-1, min(1, differential / 2))
```

### 8.2 Sentiment Analysis from Social Media

```python
"""
data/external/sentiment.py

NLP-based sentiment analysis from social media.
"""

import asyncio
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging

# For production, use transformers
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from textblob import TextBlob


class SentimentAnalyzer:
    """
    Analyzes sentiment from social media for trading signals.
    
    Sources:
    - Twitter/X (via API or scraping)
    - Reddit (r/cryptocurrency, r/forex, r/wallstreetbets)
    - Telegram groups
    - Discord servers
    - News headlines
    
    Models:
    - FinBERT for financial sentiment (best)
    - TextBlob for fallback (simple)
    """
    
    def __init__(
        self,
        use_finbert: bool = True,
        cache_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.sentiment_cache = deque(maxlen=cache_size)
        
        # Initialize FinBERT if available
        self.finbert = None
        if use_finbert and TRANSFORMERS_AVAILABLE:
            try:
                self.finbert = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert"
                )
                self.logger.info("FinBERT loaded for sentiment analysis")
            except Exception as e:
                self.logger.warning(f"FinBERT load failed: {e}, using TextBlob")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            {
                "sentiment": -1 to 1 (negative to positive),
                "confidence": 0 to 1,
                "subjectivity": 0 to 1
            }
        """
        # Clean text
        text = self._clean_text(text)
        
        if not text:
            return {"sentiment": 0.0, "confidence": 0.0, "subjectivity": 0.0}
        
        if self.finbert:
            return self._analyze_finbert(text)
        else:
            return self._analyze_textblob(text)
    
    def analyze_batch(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze batch of texts and return aggregate sentiment.
        """
        if not texts:
            return {"sentiment": 0.0, "confidence": 0.0, "volume": 0}
        
        sentiments = []
        confidences = []
        
        for text in texts:
            result = self.analyze_text(text)
            if result["confidence"] > 0.3:  # Filter low confidence
                sentiments.append(result["sentiment"])
                confidences.append(result["confidence"])
        
        if not sentiments:
            return {"sentiment": 0.0, "confidence": 0.0, "volume": 0}
        
        # Confidence-weighted average
        weights = [c / sum(confidences) for c in confidences]
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights))
        
        return {
            "sentiment": weighted_sentiment,
            "confidence": sum(confidences) / len(confidences),
            "volume": len(texts),
            "bullish_ratio": sum(1 for s in sentiments if s > 0.2) / len(sentiments),
            "bearish_ratio": sum(1 for s in sentiments if s < -0.2) / len(sentiments)
        }
    
    def get_contrarian_signal(self, sentiment: float, volume: int) -> float:
        """
        Generate contrarian signal from sentiment.
        
        Extreme bullishness with high volume = bearish signal
        Extreme bearishness with high volume = bullish signal
        """
        # Volume threshold for significance
        if volume < 10:
            return 0.0
        
        # Contrarian only at extremes
        if abs(sentiment) < 0.6:
            return 0.0
        
        # Scale by extremity
        extremity = (abs(sentiment) - 0.6) / 0.4  # 0 at 0.6, 1 at 1.0
        
        # Return opposite signal
        return -sentiment * extremity
    
    def _analyze_finbert(self, text: str) -> Dict[str, float]:
        """Use FinBERT for financial sentiment."""
        try:
            result = self.finbert(text[:512])[0]  # Truncate to model limit
            
            label = result["label"].lower()
            score = result["score"]
            
            if label == "positive":
                sentiment = score
            elif label == "negative":
                sentiment = -score
            else:
                sentiment = 0.0
            
            return {
                "sentiment": sentiment,
                "confidence": score,
                "subjectivity": 0.5  # FinBERT doesn't provide this
            }
        except Exception as e:
            self.logger.error(f"FinBERT error: {e}")
            return self._analyze_textblob(text)
    
    def _analyze_textblob(self, text: str) -> Dict[str, float]:
        """Use TextBlob for simple sentiment."""
        blob = TextBlob(text)
        
        return {
            "sentiment": blob.sentiment.polarity,  # -1 to 1
            "confidence": abs(blob.sentiment.polarity),
            "subjectivity": blob.sentiment.subjectivity
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keep the word)
        text = re.sub(r'#', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
```

---

## PART IX: ADVANCED ML ARCHITECTURES

### 9.1 Temporal Fusion Transformer

```python
"""
models/temporal/tft.py

Temporal Fusion Transformer for multi-horizon forecasting.

This is state-of-the-art for time series prediction, combining:
- Variable selection networks (learn which features matter)
- Gated residual networks (learn complex patterns)
- Multi-head attention (capture long-range dependencies)
- Quantile outputs (uncertainty estimation)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network for feature processing.
    
    Allows the network to skip connections when input is sufficient.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # Main pathway
        if context_dim is not None:
            self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        
        # Gating
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Residual connection
        if input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = None
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Residual
        if self.residual_projection is not None:
            residual = self.residual_projection(x)
        else:
            residual = x
        
        # Main pathway
        hidden = self.fc1(x)
        
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_projection(context)
        
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating
        gate = self.sigmoid(self.gate(hidden))
        
        # Output
        output = self.fc3(hidden)
        output = gate * output + (1 - gate) * residual
        
        return self.layer_norm(output)


class VariableSelectionNetwork(nn.Module):
    """
    Learns which input features are most important.
    
    Uses softmax attention over features to weight them.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        
        # Feature-wise GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim, hidden_dim, hidden_dim, dropout, context_dim
            )
            for _ in range(num_inputs)
        ])
        
        # Variable selection weights
        self.selection_weights = GatedResidualNetwork(
            input_dim * num_inputs, hidden_dim, num_inputs, dropout, context_dim
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, num_inputs, input_dim)
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process each feature
        processed = []
        for i, grn in enumerate(self.feature_grns):
            processed.append(grn(x[:, i, :], context))
        
        processed = torch.stack(processed, dim=1)  # (batch, num_inputs, hidden)
        
        # Calculate selection weights
        flattened = x.reshape(x.shape[0], -1)  # (batch, num_inputs * input_dim)
        weights = self.selection_weights(flattened, context)
        weights = self.softmax(weights)  # (batch, num_inputs)
        
        # Apply weights
        weights = weights.unsqueeze(-1)  # (batch, num_inputs, 1)
        selected = (processed * weights).sum(dim=1)  # (batch, hidden)
        
        return selected, weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """
    Full TFT implementation for financial time series.
    
    Architecture:
    1. Variable Selection (static + temporal features)
    2. LSTM Encoder for local patterns
    3. Static Enrichment
    4. Multi-Head Attention for long-range
    5. Position-wise Feed Forward
    6. Quantile outputs for uncertainty
    """
    
    def __init__(
        self,
        num_static_features: int,
        num_temporal_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        num_quantiles: int = 3,  # [0.1, 0.5, 0.9]
        forecast_horizon: int = 1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.num_quantiles = num_quantiles
        
        # Static variable selection
        self.static_vsn = VariableSelectionNetwork(
            input_dim=1,
            num_inputs=num_static_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Temporal variable selection
        self.temporal_vsn = VariableSelectionNetwork(
            input_dim=1,
            num_inputs=num_temporal_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            context_dim=hidden_dim
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout, hidden_dim
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output layers
        self.output_layer = nn.Linear(hidden_dim, num_quantiles * forecast_horizon)
    
    def forward(
        self,
        static_features: torch.Tensor,   # (batch, num_static)
        temporal_features: torch.Tensor  # (batch, seq_len, num_temporal)
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, num_temporal = temporal_features.shape
        
        # Process static features
        static_input = static_features.unsqueeze(-1)  # (batch, num_static, 1)
        static_context, static_weights = self.static_vsn(static_input)
        
        # Process temporal features
        temporal_outputs = []
        temporal_weights_list = []
        
        for t in range(seq_len):
            temp_input = temporal_features[:, t, :].unsqueeze(-1)  # (batch, num_temp, 1)
            temp_out, temp_weights = self.temporal_vsn(temp_input, static_context)
            temporal_outputs.append(temp_out)
            temporal_weights_list.append(temp_weights)
        
        temporal_processed = torch.stack(temporal_outputs, dim=1)  # (batch, seq_len, hidden)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(temporal_processed)
        
        # Static enrichment
        enriched = []
        for t in range(seq_len):
            e = self.static_enrichment(lstm_out[:, t, :], static_context)
            enriched.append(e)
        enriched = torch.stack(enriched, dim=1)
        
        # Self-attention
        attended, attention_weights = self.attention(enriched, enriched, enriched)
        
        # Output (use last timestep)
        output = self.output_layer(attended[:, -1, :])
        output = output.reshape(batch_size, self.forecast_horizon, self.num_quantiles)
        
        # Return predictions and interpretability info
        return output, {
            "static_weights": static_weights,
            "temporal_weights": torch.stack(temporal_weights_list, dim=1),
            "attention_weights": attention_weights
        }


class TFTPredictor:
    """
    Wrapper for TFT model with training and prediction.
    """
    
    def __init__(
        self,
        num_static: int,
        num_temporal: int,
        hidden_dim: int = 64,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None
    ):
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = TemporalFusionTransformer(
            num_static_features=num_static,
            num_temporal_features=num_temporal,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def predict(
        self,
        static: np.ndarray,
        temporal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction with uncertainty.
        
        Returns:
            {
                "low": 10th percentile prediction,
                "mid": 50th percentile prediction (median),
                "high": 90th percentile prediction,
                "feature_importance": static feature weights
            }
        """
        self.model.eval()
        
        with torch.no_grad():
            static_t = torch.FloatTensor(static).unsqueeze(0).to(self.device)
            temporal_t = torch.FloatTensor(temporal).unsqueeze(0).to(self.device)
            
            output, interpretability = self.model(static_t, temporal_t)
            
            output = output.cpu().numpy()[0, 0]  # First horizon
            
            return {
                "low": output[0],
                "mid": output[1],
                "high": output[2],
                "confidence": (output[2] - output[0]) / abs(output[1] + 1e-6),
                "feature_importance": interpretability["static_weights"].cpu().numpy()
            }
```

---

## PART X: MARKET MICROSTRUCTURE SIGNALS

### 10.1 Order Flow Analysis

```python
"""
signals/generators/microstructure.py

Market microstructure signals from tick data and order flow.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import logging


@dataclass
class TickData:
    """Single tick data point."""
    timestamp: pd.Timestamp
    bid: float
    ask: float
    last: float
    volume: float
    bid_size: float
    ask_size: float


class OrderFlowAnalyzer:
    """
    Analyzes order flow for trading signals.
    
    Signals generated:
    1. Trade imbalance (buy vs sell volume)
    2. Spread dynamics (widening = uncertainty)
    3. Quote pressure (bid/ask size imbalance)
    4. Trade intensity (Hawkes process)
    5. Toxicity (adverse selection risk)
    """
    
    def __init__(
        self,
        tick_buffer_size: int = 10000,
        logger: Optional[logging.Logger] = None
    ):
        self.tick_buffer = deque(maxlen=tick_buffer_size)
        self.logger = logger or logging.getLogger(__name__)
        
        # State tracking
        self.last_tick: Optional[TickData] = None
        self.cumulative_buy_volume = 0.0
        self.cumulative_sell_volume = 0.0
        self.trade_timestamps = deque(maxlen=1000)
    
    def process_tick(self, tick: TickData) -> Dict[str, float]:
        """
        Process a new tick and return signals.
        """
        signals = {}
        
        # Classify trade direction (tick rule)
        if self.last_tick is not None:
            if tick.last > self.last_tick.last:
                direction = 1  # Buy
                self.cumulative_buy_volume += tick.volume
            elif tick.last < self.last_tick.last:
                direction = -1  # Sell
                self.cumulative_sell_volume += tick.volume
            else:
                direction = 0  # Unknown
        else:
            direction = 0
        
        # Update buffers
        self.tick_buffer.append(tick)
        self.trade_timestamps.append(tick.timestamp)
        self.last_tick = tick
        
        # Calculate signals
        signals["trade_imbalance"] = self._calculate_imbalance()
        signals["spread_signal"] = self._calculate_spread_signal(tick)
        signals["quote_pressure"] = self._calculate_quote_pressure(tick)
        signals["trade_intensity"] = self._calculate_intensity()
        signals["vpin"] = self._calculate_vpin()
        
        return signals
    
    def _calculate_imbalance(self) -> float:
        """
        Calculate trade imbalance signal.
        
        Positive = more buying pressure
        Negative = more selling pressure
        """
        total = self.cumulative_buy_volume + self.cumulative_sell_volume
        if total == 0:
            return 0.0
        
        imbalance = (self.cumulative_buy_volume - self.cumulative_sell_volume) / total
        
        # Decay over time
        self.cumulative_buy_volume *= 0.999
        self.cumulative_sell_volume *= 0.999
        
        return imbalance
    
    def _calculate_spread_signal(self, tick: TickData) -> float:
        """
        Calculate spread-based signal.
        
        Widening spread = uncertainty, potential reversal
        Tightening spread = confidence, trend continuation
        """
        if len(self.tick_buffer) < 100:
            return 0.0
        
        current_spread = tick.ask - tick.bid
        mid = (tick.ask + tick.bid) / 2
        spread_pct = current_spread / mid * 100
        
        # Compare to recent average
        recent_spreads = [
            (t.ask - t.bid) / ((t.ask + t.bid) / 2) * 100
            for t in list(self.tick_buffer)[-100:]
        ]
        avg_spread = np.mean(recent_spreads)
        
        if avg_spread == 0:
            return 0.0
        
        # Normalized spread deviation
        spread_z = (spread_pct - avg_spread) / (np.std(recent_spreads) + 1e-6)
        
        # High spread = negative signal (uncertainty)
        return -np.tanh(spread_z / 2)
    
    def _calculate_quote_pressure(self, tick: TickData) -> float:
        """
        Calculate quote size imbalance.
        
        More bid size = buying pressure
        More ask size = selling pressure
        """
        total_size = tick.bid_size + tick.ask_size
        if total_size == 0:
            return 0.0
        
        pressure = (tick.bid_size - tick.ask_size) / total_size
        return pressure
    
    def _calculate_intensity(self) -> float:
        """
        Calculate trade arrival intensity.
        
        High intensity often precedes moves.
        Uses simplified Hawkes process estimation.
        """
        if len(self.trade_timestamps) < 10:
            return 0.0
        
        timestamps = list(self.trade_timestamps)
        
        # Calculate inter-arrival times
        deltas = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            if delta > 0:
                deltas.append(delta)
        
        if not deltas:
            return 0.0
        
        # Current intensity vs average
        recent_intensity = 1 / np.mean(deltas[-10:])
        avg_intensity = 1 / np.mean(deltas)
        
        if avg_intensity == 0:
            return 0.0
        
        intensity_ratio = recent_intensity / avg_intensity
        
        # Normalize to -1 to 1
        return np.tanh(intensity_ratio - 1)
    
    def _calculate_vpin(self, bucket_size: int = 50) -> float:
        """
        Calculate Volume-synchronized Probability of Informed Trading.
        
        High VPIN = high probability of informed trading = toxicity
        """
        if len(self.tick_buffer) < bucket_size * 2:
            return 0.0
        
        ticks = list(self.tick_buffer)[-bucket_size*10:]
        
        # Create volume buckets
        buckets_buy = []
        buckets_sell = []
        
        current_buy = 0
        current_sell = 0
        current_volume = 0
        
        for i in range(len(ticks) - 1):
            tick = ticks[i]
            next_tick = ticks[i + 1]
            
            volume = tick.volume
            current_volume += volume
            
            # Classify direction
            if next_tick.last > tick.last:
                current_buy += volume
            elif next_tick.last < tick.last:
                current_sell += volume
            else:
                # Split evenly
                current_buy += volume / 2
                current_sell += volume / 2
            
            # Check if bucket complete
            if current_volume >= bucket_size:
                buckets_buy.append(current_buy)
                buckets_sell.append(current_sell)
                current_buy = 0
                current_sell = 0
                current_volume = 0
        
        if len(buckets_buy) < 5:
            return 0.0
        
        # VPIN = average absolute imbalance
        imbalances = [
            abs(b - s) / (b + s) if (b + s) > 0 else 0
            for b, s in zip(buckets_buy, buckets_sell)
        ]
        
        vpin = np.mean(imbalances)
        
        # High VPIN is bearish signal (informed selling usually)
        return -vpin
    
    def get_aggregate_signal(self) -> Dict[str, float]:
        """
        Get aggregate microstructure signal.
        """
        if not self.last_tick:
            return {"microstructure_signal": 0.0, "confidence": 0.0}
        
        signals = self.process_tick(self.last_tick)
        
        # Weight and combine signals
        weights = {
            "trade_imbalance": 0.3,
            "spread_signal": 0.15,
            "quote_pressure": 0.25,
            "trade_intensity": 0.15,
            "vpin": 0.15
        }
        
        weighted_sum = sum(
            signals.get(k, 0) * w for k, w in weights.items()
        )
        
        # Confidence based on data sufficiency
        confidence = min(1.0, len(self.tick_buffer) / 1000)
        
        return {
            "microstructure_signal": weighted_sum,
            "confidence": confidence,
            **signals
        }
```

---

## PART XI: FINAL CHECKLIST

Before generating the bot files, verify:

### Architecture Checklist
- [ ] All parameters derived from market data (no magic numbers)
- [ ] Multi-model ensemble with disagreement handling
- [ ] Regime-aware model selection
- [ ] Alternative data integration
- [ ] Institutional execution algorithms
- [ ] Online learning and adaptation

### Risk Checklist
- [ ] Guardian limits below actual prop firm limits
- [ ] Daily loss tracking with auto-cutoff (The5ers)
- [ ] Drawdown tracking with auto-cutoff
- [ ] Position correlation management
- [ ] Consistency rule enforcement (The5ers)
- [ ] Inactivity ping trades (GFT)

### Infrastructure Checklist
- [ ] Auto-reconnect with exponential backoff
- [ ] State persistence and recovery
- [ ] Comprehensive logging
- [ ] Telegram alerts and commands
- [ ] Graceful shutdown handling
- [ ] Health monitoring

### Testing Checklist
- [ ] Unit tests for risk calculations
- [ ] Integration tests for MT5
- [ ] Backtest with realistic slippage
- [ ] Stress test for connection failures
- [ ] Monte Carlo for worst-case scenarios

---

## EXECUTE

Now generate the four complete, standalone bot files that implement everything described above. Each file should be 3000-5000 lines of production-grade Python code.

Start with `gft_bot_account1.py`.
