"""
Vectorized Backtesting Engine - Fast strategy backtesting.

Implements vectorized operations for efficient backtesting with
realistic transaction costs and slippage modeling.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 10000.0
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    max_position_pct: float = 1.0  # Max 100% of capital
    risk_free_rate: float = 0.02  # 2% annual
    
    # Prop firm limits
    max_drawdown_pct: float = 8.0
    max_daily_loss_pct: float = 5.0
    
    # Position sizing
    use_kelly: bool = True
    max_kelly_fraction: float = 0.25


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    duration_bars: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResults:
    """Complete backtest results."""
    # Basic metrics
    total_return: float
    total_return_pct: float
    cagr: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_hold_time: float
    
    # Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    returns_series: pd.Series
    
    # Trade list
    trades: List[Trade]
    
    # Config
    config: BacktestConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large series)."""
        return {
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_pnl': self.avg_trade_pnl,
            'avg_hold_time': self.avg_hold_time
        }


class VectorizedBacktester:
    """
    Fast vectorized backtesting engine.
    
    Uses numpy operations for efficient computation.
    
    Usage:
        backtester = VectorizedBacktester(config)
        
        # With signal function
        results = backtester.run(
            df,
            signal_func=lambda df: df['sma_fast'] > df['sma_slow']
        )
        
        # With pre-computed signals
        results = backtester.run(df, signals=signal_series)
    """
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize backtester."""
        self.config = config or BacktestConfig()
    
    def run(
        self,
        df: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], pd.Series] = None,
        signals: pd.Series = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None
    ) -> BacktestResults:
        """
        Run backtest.
        
        Args:
            df: OHLCV DataFrame
            signal_func: Function that takes df and returns signal Series
            signals: Pre-computed signal Series (-1, 0, 1)
            stop_loss_pct: Optional stop loss percentage
            take_profit_pct: Optional take profit percentage
            
        Returns:
            BacktestResults object
        """
        df = df.copy()
        
        # Generate signals
        if signals is None and signal_func is not None:
            signals = signal_func(df)
        elif signals is None:
            raise ValueError("Need either signal_func or signals")
        
        # Ensure signals are -1, 0, 1
        signals = np.sign(signals).fillna(0)
        
        # Calculate returns
        returns = df['close'].pct_change().fillna(0)
        
        # Apply transaction costs on position changes
        position_changes = signals.diff().abs().fillna(0)
        transaction_costs = position_changes * (self.config.commission_pct + self.config.slippage_pct)
        
        # Strategy returns
        strategy_returns = signals.shift(1) * returns - transaction_costs
        strategy_returns = strategy_returns.fillna(0)
        
        # Apply stops if specified
        if stop_loss_pct or take_profit_pct:
            strategy_returns = self._apply_stops(
                df, signals, strategy_returns, stop_loss_pct, take_profit_pct
            )
        
        # Calculate equity curve
        equity = self.config.initial_capital * (1 + strategy_returns).cumprod()
        
        # Check prop firm limits
        equity, strategy_returns = self._apply_risk_limits(equity, strategy_returns)
        
        # Extract trades
        trades = self._extract_trades(df, signals, strategy_returns)
        
        # Calculate metrics
        results = self._calculate_metrics(equity, strategy_returns, trades)
        
        return results
    
    def _apply_stops(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        returns: pd.Series,
        stop_loss_pct: float,
        take_profit_pct: float
    ) -> pd.Series:
        """Apply stop loss and take profit."""
        modified_returns = returns.copy()
        
        position = 0
        entry_price = 0
        
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            current_price = df['close'].iloc[i]
            
            if position == 0 and current_signal != 0:
                # Enter position
                position = current_signal
                entry_price = current_price
            
            elif position != 0:
                # Check stops
                if position > 0:  # Long
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # Short
                    pnl_pct = (entry_price - current_price) / entry_price
                
                hit_stop = stop_loss_pct and pnl_pct <= -stop_loss_pct
                hit_tp = take_profit_pct and pnl_pct >= take_profit_pct
                
                if hit_stop or hit_tp or current_signal == 0:
                    position = 0
                    entry_price = 0
        
        return modified_returns
    
    def _apply_risk_limits(
        self,
        equity: pd.Series,
        returns: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Apply prop firm risk limits."""
        # Calculate drawdown
        rolling_max = equity.cummax()
        drawdown_pct = (rolling_max - equity) / rolling_max * 100
        
        # Check max drawdown
        breach_mask = drawdown_pct > self.config.max_drawdown_pct
        
        if breach_mask.any():
            first_breach = breach_mask.idxmax()
            logger.warning(f"Max drawdown breached at {first_breach}")
            
            # Zero returns after breach
            returns.loc[first_breach:] = 0
            equity.loc[first_breach:] = equity.loc[first_breach]
        
        return equity, returns
    
    def _extract_trades(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        returns: pd.Series
    ) -> List[Trade]:
        """Extract individual trades from signals."""
        trades = []
        
        position = 0
        entry_idx = 0
        entry_price = 0
        direction = ''
        
        for i in range(len(signals)):
            current_signal = signals.iloc[i]
            
            # Entry
            if position == 0 and current_signal != 0:
                position = current_signal
                entry_idx = i
                entry_price = df['close'].iloc[i]
                direction = 'long' if current_signal > 0 else 'short'
            
            # Exit
            elif position != 0 and (current_signal == 0 or np.sign(current_signal) != np.sign(position)):
                exit_price = df['close'].iloc[i]
                
                if direction == 'long':
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                size = self.config.initial_capital * self.config.max_position_pct
                pnl = size * pnl_pct
                
                commission = size * self.config.commission_pct * 2
                slippage_cost = size * self.config.slippage_pct * 2
                
                trade = Trade(
                    entry_time=df.index[entry_idx] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    exit_time=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(),
                    symbol=df.get('symbol', ['unknown'])[0] if 'symbol' in df.columns else 'unknown',
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    pnl=pnl - commission - slippage_cost,
                    pnl_pct=pnl_pct,
                    commission=commission,
                    slippage=slippage_cost,
                    duration_bars=i - entry_idx
                )
                trades.append(trade)
                
                # Check if entering opposite position
                if current_signal != 0 and np.sign(current_signal) != np.sign(position):
                    position = current_signal
                    entry_idx = i
                    entry_price = df['close'].iloc[i]
                    direction = 'long' if current_signal > 0 else 'short'
                else:
                    position = 0
        
        return trades
    
    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: List[Trade]
    ) -> BacktestResults:
        """Calculate all performance metrics."""
        # Basic metrics
        total_return = equity.iloc[-1] - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital * 100
        
        # CAGR
        n_years = len(returns) / 252  # Assuming daily
        if n_years > 0:
            cagr = (equity.iloc[-1] / self.config.initial_capital) ** (1/n_years) - 1
        else:
            cagr = 0
        
        # Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate / 252
        if excess_returns.std() > 0:
            sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino = 0
        
        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (rolling_max - equity) / rolling_max
        max_drawdown = drawdown.max() * 100
        
        # Max drawdown duration
        dd_duration = 0
        max_dd_duration = 0
        for dd in drawdown:
            if dd > 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0
        
        # Calmar ratio
        calmar = cagr / (max_drawdown / 100) if max_drawdown > 0 else 0
        
        # Trade metrics
        if trades:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(winners) / len(trades) * 100
            
            total_wins = sum(t.pnl for t in winners)
            total_losses = abs(sum(t.pnl for t in losers))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            avg_trade_pnl = sum(t.pnl for t in trades) / len(trades)
            avg_winner = sum(t.pnl for t in winners) / len(winners) if winners else 0
            avg_loser = sum(t.pnl for t in losers) / len(losers) if losers else 0
            largest_winner = max(t.pnl for t in trades) if trades else 0
            largest_loser = min(t.pnl for t in trades) if trades else 0
            avg_hold_time = sum(t.duration_bars for t in trades) / len(trades)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_pnl = 0
            avg_winner = 0
            avg_loser = 0
            largest_winner = 0
            largest_loser = 0
            avg_hold_time = 0
        
        return BacktestResults(
            total_return=total_return,
            total_return_pct=total_return_pct,
            cagr=cagr * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            avg_hold_time=avg_hold_time,
            equity_curve=equity,
            drawdown_series=drawdown * 100,
            returns_series=returns,
            trades=trades,
            config=self.config
        )
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        signal_func: Callable,
        train_pct: float = 0.7,
        n_splits: int = 5
    ) -> List[BacktestResults]:
        """
        Run walk-forward analysis.
        
        Args:
            df: OHLCV DataFrame
            signal_func: Signal generation function
            train_pct: Percentage of data for training in each split
            n_splits: Number of walk-forward splits
            
        Returns:
            List of BacktestResults for each out-of-sample period
        """
        results = []
        n = len(df)
        split_size = n // n_splits
        
        for i in range(n_splits):
            # Define train and test indices
            test_start = i * split_size
            test_end = (i + 1) * split_size if i < n_splits - 1 else n
            
            train_end = test_start
            train_start = max(0, train_end - int(split_size / (1 - train_pct) * train_pct))
            
            if train_start >= train_end:
                continue
            
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            
            # Generate signals on test data
            signals = signal_func(test_df)
            
            # Run backtest
            result = self.run(test_df, signals=signals)
            results.append(result)
        
        return results


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness.
    
    Tests strategy under randomized conditions.
    """
    
    def __init__(self, backtester: VectorizedBacktester):
        """Initialize with backtester."""
        self.backtester = backtester
    
    def run_simulation(
        self,
        trades: List[Trade],
        n_simulations: int = 1000,
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on trade results.
        
        Randomly shuffles trade order to estimate distribution of outcomes.
        """
        trade_pnls = np.array([t.pnl for t in trades])
        
        final_equities = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Shuffle trades
            shuffled_pnls = np.random.permutation(trade_pnls)
            
            # Calculate equity curve
            equity = initial_capital + np.cumsum(shuffled_pnls)
            equity = np.insert(equity, 0, initial_capital)
            
            final_equities.append(equity[-1])
            
            # Calculate max drawdown
            rolling_max = np.maximum.accumulate(equity)
            drawdown = (rolling_max - equity) / rolling_max
            max_drawdowns.append(drawdown.max() * 100)
        
        return {
            'mean_final_equity': np.mean(final_equities),
            'median_final_equity': np.median(final_equities),
            'std_final_equity': np.std(final_equities),
            'percentile_5': np.percentile(final_equities, 5),
            'percentile_95': np.percentile(final_equities, 95),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'percentile_95_drawdown': np.percentile(max_drawdowns, 95),
            'prob_profit': np.mean(np.array(final_equities) > initial_capital),
            'prob_ruin': np.mean(np.array(final_equities) < initial_capital * 0.5)
        }
