"""
Prometheus Metrics Exporter - Real-time monitoring and alerting.

Exposes trading system metrics in Prometheus format for monitoring,
alerting, and dashboarding with Grafana.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, using built-in metrics server")


class MetricsRegistry:
    """
    Central registry for all trading metrics.
    
    Tracks:
    - Account metrics (balance, equity, drawdown)
    - Trading metrics (trades, win rate, PnL)
    - System metrics (latency, errors, uptime)
    - Model metrics (predictions, accuracy)
    """
    
    def __init__(self, prefix: str = "trading"):
        """
        Initialize metrics registry.
        
        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix
        self._metrics: Dict[str, Any] = {}
        self._values: Dict[str, float] = {}
        self._labels: Dict[str, Dict[str, str]] = {}
        
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self._init_simple_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Account metrics
        self._metrics['balance'] = Gauge(
            f'{self.prefix}_account_balance',
            'Current account balance in USD',
            ['account', 'broker'],
            registry=self.registry
        )
        
        self._metrics['equity'] = Gauge(
            f'{self.prefix}_account_equity',
            'Current account equity in USD',
            ['account', 'broker'],
            registry=self.registry
        )
        
        self._metrics['drawdown'] = Gauge(
            f'{self.prefix}_drawdown_percent',
            'Current drawdown percentage',
            ['account'],
            registry=self.registry
        )
        
        self._metrics['daily_pnl'] = Gauge(
            f'{self.prefix}_daily_pnl',
            'Daily profit/loss in USD',
            ['account'],
            registry=self.registry
        )
        
        # Trading metrics
        self._metrics['trades_total'] = Counter(
            f'{self.prefix}_trades_total',
            'Total number of trades executed',
            ['account', 'symbol', 'direction'],
            registry=self.registry
        )
        
        self._metrics['trades_won'] = Counter(
            f'{self.prefix}_trades_won_total',
            'Total number of winning trades',
            ['account', 'symbol'],
            registry=self.registry
        )
        
        self._metrics['trade_pnl'] = Histogram(
            f'{self.prefix}_trade_pnl',
            'Trade PnL distribution',
            ['account', 'symbol'],
            buckets=(-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000),
            registry=self.registry
        )
        
        self._metrics['position_size'] = Gauge(
            f'{self.prefix}_position_size',
            'Current position size',
            ['account', 'symbol'],
            registry=self.registry
        )
        
        # Risk metrics
        self._metrics['risk_utilization'] = Gauge(
            f'{self.prefix}_risk_utilization_percent',
            'Current risk utilization percentage',
            ['account'],
            registry=self.registry
        )
        
        self._metrics['margin_level'] = Gauge(
            f'{self.prefix}_margin_level_percent',
            'Current margin level percentage',
            ['account'],
            registry=self.registry
        )
        
        # System metrics
        self._metrics['heartbeat'] = Gauge(
            f'{self.prefix}_heartbeat_timestamp',
            'Last heartbeat timestamp',
            ['account'],
            registry=self.registry
        )
        
        self._metrics['errors'] = Counter(
            f'{self.prefix}_errors_total',
            'Total number of errors',
            ['account', 'error_type'],
            registry=self.registry
        )
        
        self._metrics['latency'] = Histogram(
            f'{self.prefix}_operation_latency_seconds',
            'Operation latency in seconds',
            ['account', 'operation'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        self._metrics['connection_status'] = Gauge(
            f'{self.prefix}_connection_status',
            'Connection status (1=connected, 0=disconnected)',
            ['account', 'broker'],
            registry=self.registry
        )
        
        # Model metrics
        self._metrics['model_prediction'] = Gauge(
            f'{self.prefix}_model_prediction',
            'Latest model prediction',
            ['account', 'model', 'symbol'],
            registry=self.registry
        )
        
        self._metrics['model_confidence'] = Gauge(
            f'{self.prefix}_model_confidence',
            'Model prediction confidence',
            ['account', 'model', 'symbol'],
            registry=self.registry
        )
        
        self._metrics['signal_strength'] = Gauge(
            f'{self.prefix}_signal_strength',
            'Combined signal strength',
            ['account', 'symbol'],
            registry=self.registry
        )
    
    def _init_simple_metrics(self):
        """Initialize simple in-memory metrics when Prometheus not available."""
        self._values = {
            'balance': 0,
            'equity': 0,
            'drawdown': 0,
            'daily_pnl': 0,
            'trades_total': 0,
            'trades_won': 0,
            'risk_utilization': 0,
            'heartbeat': 0,
            'errors': 0,
            'connection_status': 0
        }
    
    # Account metrics
    def set_balance(self, value: float, account: str = "default", broker: str = "mt5"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['balance'].labels(account=account, broker=broker).set(value)
        else:
            self._values['balance'] = value
    
    def set_equity(self, value: float, account: str = "default", broker: str = "mt5"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['equity'].labels(account=account, broker=broker).set(value)
        else:
            self._values['equity'] = value
    
    def set_drawdown(self, value: float, account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['drawdown'].labels(account=account).set(value)
        else:
            self._values['drawdown'] = value
    
    def set_daily_pnl(self, value: float, account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['daily_pnl'].labels(account=account).set(value)
        else:
            self._values['daily_pnl'] = value
    
    # Trading metrics
    def inc_trades(self, account: str = "default", symbol: str = "unknown", direction: str = "long"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['trades_total'].labels(
                account=account, symbol=symbol, direction=direction
            ).inc()
        else:
            self._values['trades_total'] = self._values.get('trades_total', 0) + 1
    
    def inc_wins(self, account: str = "default", symbol: str = "unknown"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['trades_won'].labels(account=account, symbol=symbol).inc()
        else:
            self._values['trades_won'] = self._values.get('trades_won', 0) + 1
    
    def observe_trade_pnl(self, value: float, account: str = "default", symbol: str = "unknown"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['trade_pnl'].labels(account=account, symbol=symbol).observe(value)
    
    def set_position(self, value: float, account: str = "default", symbol: str = "unknown"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['position_size'].labels(account=account, symbol=symbol).set(value)
    
    # Risk metrics
    def set_risk_utilization(self, value: float, account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['risk_utilization'].labels(account=account).set(value)
        else:
            self._values['risk_utilization'] = value
    
    def set_margin_level(self, value: float, account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['margin_level'].labels(account=account).set(value)
    
    # System metrics
    def heartbeat(self, account: str = "default"):
        timestamp = time.time()
        if PROMETHEUS_AVAILABLE:
            self._metrics['heartbeat'].labels(account=account).set(timestamp)
        else:
            self._values['heartbeat'] = timestamp
    
    def inc_errors(self, error_type: str = "unknown", account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['errors'].labels(account=account, error_type=error_type).inc()
        else:
            self._values['errors'] = self._values.get('errors', 0) + 1
    
    def observe_latency(self, value: float, operation: str = "unknown", account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['latency'].labels(account=account, operation=operation).observe(value)
    
    def set_connection_status(self, connected: bool, account: str = "default", broker: str = "mt5"):
        value = 1 if connected else 0
        if PROMETHEUS_AVAILABLE:
            self._metrics['connection_status'].labels(account=account, broker=broker).set(value)
        else:
            self._values['connection_status'] = value
    
    # Model metrics
    def set_prediction(self, value: float, model: str, symbol: str, account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['model_prediction'].labels(
                account=account, model=model, symbol=symbol
            ).set(value)
    
    def set_confidence(self, value: float, model: str, symbol: str, account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['model_confidence'].labels(
                account=account, model=model, symbol=symbol
            ).set(value)
    
    def set_signal_strength(self, value: float, symbol: str, account: str = "default"):
        if PROMETHEUS_AVAILABLE:
            self._metrics['signal_strength'].labels(account=account, symbol=symbol).set(value)
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            # Generate simple text format
            lines = []
            for name, value in self._values.items():
                lines.append(f"{self.prefix}_{name} {value}")
            return '\n'.join(lines)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        return self._values.copy()


class MetricsServer:
    """
    HTTP server for exposing Prometheus metrics.
    
    Usage:
        metrics = MetricsRegistry()
        server = MetricsServer(metrics, port=9090)
        server.start()
        
        # Update metrics
        metrics.set_balance(10000)
        metrics.heartbeat()
        
        # Stop server
        server.stop()
    """
    
    def __init__(self, registry: MetricsRegistry, port: int = 9090, host: str = "0.0.0.0"):
        """
        Initialize metrics server.
        
        Args:
            registry: MetricsRegistry instance
            port: Port to listen on
            host: Host to bind to
        """
        self.registry = registry
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start the metrics server."""
        if PROMETHEUS_AVAILABLE:
            # Use prometheus_client's built-in server
            start_http_server(self.port, addr=self.host, registry=self.registry.registry)
            logger.info(f"Prometheus metrics server started on {self.host}:{self.port}")
        else:
            # Use simple HTTP server
            self._start_simple_server()
    
    def _start_simple_server(self):
        """Start simple HTTP server for metrics."""
        registry = self.registry
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(registry.get_metrics_text().encode())
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'healthy'}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        self._server = HTTPServer((self.host, self.port), MetricsHandler)
        self._running = True
        
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        
        logger.info(f"Simple metrics server started on {self.host}:{self.port}")
    
    def _serve(self):
        """Serve requests."""
        while self._running:
            self._server.handle_request()
    
    def stop(self):
        """Stop the metrics server."""
        self._running = False
        if self._server:
            self._server.shutdown()


class TradingMetricsCollector:
    """
    Collects and aggregates trading metrics from bot instances.
    
    Usage:
        collector = TradingMetricsCollector()
        
        # In your bot's main loop
        collector.update_from_bot(bot)
        
        # Get aggregated metrics
        summary = collector.get_summary()
    """
    
    def __init__(self, registry: MetricsRegistry = None):
        """Initialize collector."""
        self.registry = registry or MetricsRegistry()
        self._last_update: Dict[str, datetime] = {}
    
    def update_from_bot(self, bot, account: str = "default"):
        """
        Update metrics from a trading bot instance.
        
        Args:
            bot: Bot instance with get_status() and get_risk_status() methods
            account: Account identifier
        """
        try:
            # Get bot status
            status = bot.get_status() if hasattr(bot, 'get_status') else {}
            risk_status = bot.get_risk_status() if hasattr(bot, 'get_risk_status') else {}
            
            # Update account metrics
            if 'balance' in status:
                self.registry.set_balance(status['balance'], account)
            
            if 'equity' in status:
                self.registry.set_equity(status['equity'], account)
            
            # Update risk metrics
            if 'current_drawdown' in risk_status:
                self.registry.set_drawdown(risk_status['current_drawdown'], account)
            
            if 'daily_loss' in risk_status:
                self.registry.set_daily_pnl(-risk_status['daily_loss'], account)
            
            # Update connection status
            connected = status.get('connected', False)
            self.registry.set_connection_status(connected, account)
            
            # Heartbeat
            self.registry.heartbeat(account)
            
            self._last_update[account] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            self.registry.inc_errors("metrics_update", account)
    
    def record_trade(
        self,
        pnl: float,
        symbol: str,
        direction: str,
        account: str = "default"
    ):
        """Record a completed trade."""
        self.registry.inc_trades(account, symbol, direction)
        self.registry.observe_trade_pnl(pnl, account, symbol)
        
        if pnl > 0:
            self.registry.inc_wins(account, symbol)
    
    def record_signal(
        self,
        symbol: str,
        direction: float,
        confidence: float,
        model: str = "ensemble",
        account: str = "default"
    ):
        """Record a trading signal."""
        self.registry.set_prediction(direction, model, symbol, account)
        self.registry.set_confidence(confidence, model, symbol, account)
        self.registry.set_signal_strength(abs(direction) * confidence, symbol, account)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'metrics': self.registry.get_metrics_dict(),
            'last_updates': {k: v.isoformat() for k, v in self._last_update.items()}
        }


# Global metrics instance for easy access
_global_metrics: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    """Get global metrics registry."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsRegistry()
    return _global_metrics


def init_metrics_server(port: int = 9090) -> MetricsServer:
    """Initialize and start global metrics server."""
    metrics = get_metrics()
    server = MetricsServer(metrics, port=port)
    server.start()
    return server
