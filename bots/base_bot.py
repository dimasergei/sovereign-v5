"""
Base Trading Bot - Core bot infrastructure.

All specific bots (GFT, The5ers) inherit from this base class.
"""

import logging
import signal
import sys
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import MetaTrader5 as mt5
import schedule

from core import (
    MT5Connector, MT5Credentials,
    RiskManager, FirmRules, AccountRiskState,
    FirmType
)
from core.lossless import MarketCalibrator
from core.execution import SmartExecutor
from data import MT5DataFetcher, FeatureEngineer
from signals import SignalGenerator, TradingSignal
from models import RegimeDetector, MeanReversionModel, EnsembleMetaLearner
from monitoring import TelegramNotifier, TelegramConfig, AlertLevel


logger = logging.getLogger(__name__)


class BaseTradingBot(ABC):
    """
    Base class for all trading bots.
    
    Provides:
    - Connection management
    - Risk management
    - Signal generation
    - Order execution
    - Monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        account_name: str,
        credentials: MT5Credentials,
        firm_rules: FirmRules,
        symbols: List[str],
        state_file: str,
        telegram_config: Optional[TelegramConfig] = None,
        scan_interval_seconds: int = 60,
        timeframe: str = "M5"
    ):
        """
        Initialize trading bot.
        
        Args:
            account_name: Identifier for this account
            credentials: MT5 login credentials
            firm_rules: Prop firm rules
            symbols: List of symbols to trade
            state_file: Path to state persistence file
            telegram_config: Telegram notification config
            scan_interval_seconds: Seconds between market scans
            timeframe: Primary trading timeframe
        """
        self.account_name = account_name
        self.credentials = credentials
        self.firm_rules = firm_rules
        self.symbols = symbols
        self.state_file = state_file
        self.telegram_config = telegram_config
        self.scan_interval = scan_interval_seconds
        self.timeframe = timeframe
        
        # State
        self.is_running = False
        self.is_paused = False
        
        # Components (initialized in start())
        self.connector: Optional[MT5Connector] = None
        self.risk_manager: Optional[RiskManager] = None
        self.calibrator: Optional[MarketCalibrator] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.data_fetcher: Optional[MT5DataFetcher] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.executor: Optional[SmartExecutor] = None
        self.telegram: Optional[TelegramNotifier] = None
        
        # Models
        self.regime_detector: Optional[RegimeDetector] = None
        self.ensemble: Optional[EnsembleMetaLearner] = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Bot initialized: {account_name}")
    
    def start(self):
        """Start the trading bot."""
        logger.info(f"{'='*60}")
        logger.info(f"Starting {self.account_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Initialize components
            self._init_connection()
            self._init_risk_manager()
            self._init_ml_components()
            self._init_signal_generator()
            self._init_telegram()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Setup scheduled tasks
            self._setup_schedule()
            
            # Send startup notification
            if self.telegram:
                self.telegram.send_alert(
                    f"🚀 {self.account_name} started\n"
                    f"Symbols: {', '.join(self.symbols)}",
                    AlertLevel.SUCCESS
                )
            
            # Run main loop
            self._run_main_loop()
            
        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
            if self.telegram:
                self.telegram.send_alert(
                    f"💀 {self.account_name} crashed: {e}",
                    AlertLevel.CRITICAL
                )
            raise
        finally:
            self._cleanup()
    
    def _init_connection(self):
        """Initialize MT5 connection."""
        logger.info("Initializing MT5 connection...")
        
        self.connector = MT5Connector(
            credentials=self.credentials,
            on_disconnect_callback=self._on_disconnect,
            on_reconnect_callback=self._on_reconnect
        )
        
        if not self.connector.connect():
            raise ConnectionError("Failed to connect to MT5")
        
        # Verify account
        account_info = self.connector.get_account_info()
        logger.info(
            f"Connected: Account #{account_info['login']}, "
            f"Balance: ${account_info['balance']:.2f}"
        )
        
        # Initialize data fetcher
        self.data_fetcher = MT5DataFetcher(self.connector)
        
        # Initialize executor
        self.executor = SmartExecutor(self.connector)
    
    def _init_risk_manager(self):
        """Initialize risk management."""
        logger.info("Initializing risk manager...")
        
        # Load existing state or create new
        state = RiskManager.load_state(self.state_file)
        
        if state is None:
            account_info = self.connector.get_account_info()
            state = AccountRiskState(
                initial_balance=self.firm_rules.initial_balance,
                highest_balance=account_info['balance'],
                current_balance=account_info['balance'],
                current_equity=account_info['equity'],
                daily_starting_balance=account_info['balance'],
                daily_pnl=0.0,
                daily_date=date.today().isoformat(),
            )
        
        self.risk_manager = RiskManager(
            rules=self.firm_rules,
            state=state,
            state_file=self.state_file,
            on_violation_callback=self._on_risk_violation
        )
        
        logger.info(
            f"Risk manager initialized: DD={self.risk_manager.get_current_drawdown_pct():.2f}%"
        )
    
    def _init_ml_components(self):
        """Initialize ML models and calibrator."""
        logger.info("Initializing ML components...")
        
        # Calibrator
        self.calibrator = MarketCalibrator()
        
        # Feature engineer (will be configured after first calibration)
        self.feature_engineer = FeatureEngineer()
        
        # Regime detector
        self.regime_detector = RegimeDetector()
        
        # Mean reversion model
        mr_model = MeanReversionModel()
        
        # Ensemble
        self.ensemble = EnsembleMetaLearner(
            models={'mean_reversion': mr_model, 'regime': self.regime_detector},
            min_confidence=0.5,
            agreement_threshold=0.5
        )
    
    def _init_signal_generator(self):
        """Initialize signal generator."""
        logger.info("Initializing signal generator...")
        
        self.signal_generator = SignalGenerator(
            calibrator=self.calibrator,
            feature_engineer=self.feature_engineer,
            ensemble=self.ensemble,
            regime_detector=self.regime_detector
        )
    
    def _init_telegram(self):
        """Initialize Telegram notifications."""
        if self.telegram_config:
            self.telegram = TelegramNotifier(self.telegram_config)
            logger.info("Telegram notifications enabled")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    def _setup_schedule(self):
        """Setup scheduled tasks."""
        # Check for inactivity (GFT specific)
        schedule.every().day.at("08:00").do(self._check_inactivity)
        
        # Daily risk reset (The5ers specific)
        schedule.every().day.at("00:00").do(self._daily_reset)
    
    def _run_main_loop(self):
        """Main trading loop."""
        self.is_running = True

        logger.info("Entering main loop...")

        last_floating_check = time.time()
        FLOATING_CHECK_INTERVAL = 1.0  # Check every second for GFT

        while self.is_running:
            try:
                # Run scheduled tasks
                schedule.run_pending()

                # CRITICAL: Check floating PnL frequently for GFT accounts
                if time.time() - last_floating_check >= FLOATING_CHECK_INTERVAL:
                    if self.risk_manager and self.risk_manager.rules.max_trade_floating_loss_pct:
                        positions_at_risk = self.risk_manager.monitor_floating_pnl()
                        for pos in positions_at_risk:
                            self.risk_manager.emergency_close_position(
                                pos["ticket"],
                                pos["reason"]
                            )
                            if self.telegram:
                                self.telegram.send_alert(
                                    f"🚨 EMERGENCY CLOSE: {pos['symbol']} "
                                    f"({pos['floating_pnl_pct']:.2f}% loss)",
                                    AlertLevel.CRITICAL
                                )
                    last_floating_check = time.time()

                # Skip if paused
                if self.is_paused:
                    time.sleep(1)
                    continue

                # Update account state
                if not self._update_account_state():
                    time.sleep(30)
                    continue

                # Check if locked
                if self.risk_manager.state.is_locked:
                    logger.warning("Account locked, skipping trading")
                    time.sleep(60)
                    continue

                # Scan symbols for signals
                for symbol in self.symbols:
                    if not self.is_running:
                        break
                    self._analyze_and_trade(symbol)

                # Manage existing positions
                self._manage_positions()

                # Sleep between iterations
                time.sleep(self.scan_interval)

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                if self.telegram:
                    self.telegram.send_alert(
                        f"⚠️ Error: {e}",
                        AlertLevel.WARNING
                    )
                time.sleep(30)
    
    def _update_account_state(self) -> bool:
        """Update account state from MT5."""
        if not self.connector.ensure_connected():
            return False
        
        account_info = self.connector.get_account_info()
        if not account_info:
            return False
        
        risk_status = self.risk_manager.update_account_state(
            balance=account_info['balance'],
            equity=account_info['equity']
        )
        
        # Check for violations
        if risk_status.get('violations'):
            logger.warning(f"Risk violations: {risk_status['violations']}")
        
        return True
    
    def _analyze_and_trade(self, symbol: str):
        """Analyze symbol and execute trades if signal."""
        try:
            # Get historical data
            df = self.data_fetcher.get_historical_bars(
                symbol, self.timeframe, 500
            )
            
            if df.empty or len(df) < 100:
                return
            
            # Generate signal
            signal = self.signal_generator.generate_signal(symbol, df)
            
            # Skip if neutral or low confidence
            if signal.action == "neutral" or signal.confidence < 0.5:
                return
            
            # Check if already have position in this symbol
            positions = self.connector.get_positions(symbol)
            if positions:
                logger.debug(f"Already have position in {symbol}")
                return
            
            # Calculate entry, SL, TP
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return
            
            if signal.action == "long":
                entry = tick.ask
                sl = signal.get_stop_loss(entry, is_long=True)
                tp = signal.get_take_profit(entry, is_long=True)
            else:
                entry = tick.bid
                sl = signal.get_stop_loss(entry, is_long=False)
                tp = signal.get_take_profit(entry, is_long=False)
            
            # Calculate position size
            symbol_info = self.connector.get_symbol_info(symbol)
            if not symbol_info:
                return
            
            lots, sizing_details = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=entry,
                stop_loss_price=sl,
                account_equity=self.connector.get_account_info()['equity'],
                symbol_info=symbol_info
            )
            
            # Apply signal scalar
            lots = lots * signal.position_scalar
            
            # Round to volume step
            vol_step = symbol_info.get('volume_step', 0.01)
            lots = round(lots / vol_step) * vol_step
            
            # Check minimum
            vol_min = symbol_info.get('volume_min', 0.01)
            if lots < vol_min:
                logger.debug(f"Position size too small: {lots} < {vol_min}")
                return
            
            # Validate trade
            valid, violation, msg = self.risk_manager.validate_trade(
                symbol=symbol,
                lot_size=lots,
                direction=signal.action,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp
            )
            
            if not valid:
                logger.warning(f"Trade rejected: {msg}")
                return
            
            # Execute
            plan = self.executor.create_plan(
                symbol=symbol,
                direction="buy" if signal.action == "long" else "sell",
                size=lots,
                stop_loss=sl,
                take_profit=tp,
                comment=f"{self.account_name}|{signal.regime}"
            )
            
            result = self.executor.execute(plan)
            
            if result.success:
                logger.info(
                    f"EXECUTED: {signal.action} {lots:.2f} {symbol} "
                    f"@ {result.avg_fill_price:.5f}"
                )
                
                if self.telegram:
                    self.telegram.send_trade(
                        symbol=symbol,
                        action="OPEN",
                        direction=signal.action.upper(),
                        price=result.avg_fill_price,
                        size=lots
                    )
            else:
                logger.error(f"Execution failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
    
    def _manage_positions(self):
        """Manage existing positions (trailing stops, etc)."""
        positions = self.connector.get_positions()
        
        for pos in positions:
            try:
                self._manage_single_position(pos)
            except Exception as e:
                logger.error(f"Error managing position {pos['ticket']}: {e}")
    
    def _manage_single_position(self, position: Dict):
        """Manage a single position."""
        # Implement trailing stop logic
        symbol = position['symbol']
        ticket = position['ticket']
        is_long = position['type'] == 'buy'
        entry_price = position['price_open']
        current_sl = position['sl']
        current_profit = position['profit']
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        
        current_price = tick.bid if is_long else tick.ask
        
        # Get ATR for trailing
        df = self.data_fetcher.get_historical_bars(symbol, self.timeframe, 50)
        if df.empty:
            return
        
        atr = df['high'].values[-14:].max() - df['low'].values[-14:].min()
        atr = atr / 14  # Rough ATR estimate
        
        # Trail stop to breakeven at 1 ATR profit
        if is_long:
            profit_distance = current_price - entry_price
            if profit_distance > atr and current_sl < entry_price:
                # Move SL to breakeven
                new_sl = entry_price + (atr * 0.1)  # Slight buffer
                self.executor.modify_position_sl(ticket, new_sl)
                logger.info(f"Trailing SL to breakeven for {symbol}")
        else:
            profit_distance = entry_price - current_price
            if profit_distance > atr and current_sl > entry_price:
                new_sl = entry_price - (atr * 0.1)
                self.executor.modify_position_sl(ticket, new_sl)
                logger.info(f"Trailing SL to breakeven for {symbol}")
    
    def _check_inactivity(self):
        """Check for inactivity and ping if needed."""
        if self.risk_manager.check_inactivity():
            logger.warning("Inactivity warning - consider placing a small trade")
            if self.telegram:
                self.telegram.send_alert(
                    f"⚠️ Inactivity warning: {self.risk_manager.state.days_since_last_trade} days",
                    AlertLevel.WARNING
                )
    
    def _daily_reset(self):
        """Daily reset tasks."""
        logger.info("Daily reset...")
        # Risk manager handles daily tracking reset automatically
    
    def _on_disconnect(self, attempt: int, total: int):
        """Handle disconnect callback."""
        logger.warning(f"Disconnected (attempt {attempt}, total {total})")
        if self.telegram:
            self.telegram.send_alert(
                f"⚠️ MT5 disconnected (attempt {attempt})",
                AlertLevel.WARNING
            )
    
    def _on_reconnect(self, attempt: int):
        """Handle reconnect callback."""
        logger.info(f"Reconnected on attempt {attempt}")
        if self.telegram:
            self.telegram.send_alert(
                f"✅ MT5 reconnected",
                AlertLevel.SUCCESS
            )
    
    def _on_risk_violation(self, violation_type, message: str):
        """Handle risk violation callback."""
        logger.critical(f"RISK VIOLATION: {violation_type} - {message}")
        
        if self.telegram:
            self.telegram.send_alert(
                f"🚨 RISK VIOLATION\n{violation_type}\n{message}",
                AlertLevel.CRITICAL
            )
        
        # Close all positions
        self.executor.close_all_positions()
    
    def _cleanup(self):
        """Cleanup on shutdown."""
        logger.info("Cleaning up...")
        
        if self.connector:
            self.connector.disconnect()
        
        if self.telegram:
            self.telegram.send_alert(
                f"🛑 {self.account_name} stopped",
                AlertLevel.WARNING
            )
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("storage/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.account_name}_{date.today().isoformat()}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        logging.getLogger().addHandler(file_handler)
    
    # Control methods for Telegram
    def pause(self):
        """Pause trading."""
        self.is_paused = True
        logger.info("Trading paused")
    
    def resume(self):
        """Resume trading."""
        self.is_paused = False
        logger.info("Trading resumed")
    
    def emergency_stop(self):
        """Emergency stop - close all and lock."""
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        # Close all positions
        if self.executor:
            self.executor.close_all_positions()
        
        # Lock account
        if self.risk_manager:
            self.risk_manager.emergency_close_all("Manual emergency stop")
        
        self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status for Telegram."""
        account = self.connector.get_account_info() if self.connector else {}
        positions = self.connector.get_positions() if self.connector else []
        
        return {
            'balance': account.get('balance', 0),
            'equity': account.get('equity', 0),
            'profit': account.get('profit', 0),
            'drawdown_pct': self.risk_manager.get_current_drawdown_pct() if self.risk_manager else 0,
            'open_positions': len(positions),
            'is_trading': self.is_running and not self.is_paused,
            'is_locked': self.risk_manager.state.is_locked if self.risk_manager else False,
        }
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get risk status for Telegram."""
        if self.risk_manager:
            return self.risk_manager.get_risk_status()
        return {}
    
    def get_performance(self) -> Dict[str, Any]:
        """Get performance stats."""
        if not self.risk_manager:
            return {}
        
        return {
            'total_trades': self.risk_manager.state.total_trades,
            'win_rate': self.risk_manager.get_win_rate(),
            'profit_factor': self.risk_manager.get_profit_factor(),
            'total_profit': self.risk_manager.state.total_realized_profit,
            'best_trade': 0,  # Would need trade history
            'worst_trade': 0,
        }