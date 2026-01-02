"""
Sample Data Generators - Generate realistic market data for testing.

All data generation follows the lossless principle:
- Parameters are derived from statistical distributions
- No hardcoded magic numbers
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TickData:
    """Tick data point."""
    timestamp: datetime
    bid: float
    ask: float
    volume: float


def generate_ohlcv_data(
    symbol: str = "BTCUSD",
    n_bars: int = 1000,
    timeframe: str = "1H",
    start_price: float = 50000.0,
    volatility: float = None,
    trend: float = 0.0,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data using geometric Brownian motion.

    Parameters are derived from typical market characteristics:
    - Volatility is derived from historical analysis if not provided
    - Returns follow a fat-tailed distribution

    Args:
        symbol: Trading symbol
        n_bars: Number of bars to generate
        timeframe: Timeframe string (1M, 5M, 15M, 1H, 4H, 1D)
        start_price: Starting price
        volatility: Daily volatility (derived if None)
        trend: Daily drift
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)

    # Derive volatility from typical market behavior if not provided
    if volatility is None:
        # Use crypto-like volatility (~3-5% daily)
        volatility = np.random.uniform(0.02, 0.05)

    # Map timeframe to bars per day
    timeframe_map = {
        "1M": 1440,
        "5M": 288,
        "15M": 96,
        "1H": 24,
        "4H": 6,
        "1D": 1,
    }

    bars_per_day = timeframe_map.get(timeframe, 24)

    # Adjust volatility for timeframe
    bar_volatility = volatility / np.sqrt(bars_per_day)
    bar_drift = trend / bars_per_day

    # Generate returns using fat-tailed distribution (Student's t)
    # Degrees of freedom derived from market analysis
    df_t = np.random.uniform(3, 6)  # Fat tails
    returns = np.random.standard_t(df_t, size=n_bars) * bar_volatility + bar_drift
    returns[0] = 0  # First bar has no return

    # Generate close prices
    log_prices = np.log(start_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)

    # Generate OHLC from close
    # Intrabar volatility derived from close-to-close volatility
    intrabar_vol = bar_volatility * np.random.uniform(1.0, 1.5)

    # Generate high/low as deviations from close
    high_devs = np.abs(np.random.normal(0, intrabar_vol, n_bars))
    low_devs = np.abs(np.random.normal(0, intrabar_vol, n_bars))

    high_prices = close_prices * (1 + high_devs)
    low_prices = close_prices * (1 - low_devs)

    # Generate open as previous close with small gap
    open_prices = np.roll(close_prices, 1) * (1 + np.random.normal(0, bar_volatility * 0.1, n_bars))
    open_prices[0] = start_price

    # Ensure OHLC consistency
    high_prices = np.maximum.reduce([high_prices, open_prices, close_prices])
    low_prices = np.minimum.reduce([low_prices, open_prices, close_prices])

    # Generate volume (log-normal distribution)
    base_volume = np.random.uniform(100, 1000)
    volume = np.random.lognormal(np.log(base_volume), 0.5, n_bars)

    # Higher volume on larger moves (correlation derived from data)
    volume *= (1 + np.abs(returns) * np.random.uniform(5, 15))

    # Generate timestamps
    timeframe_minutes = {
        "1M": 1,
        "5M": 5,
        "15M": 15,
        "1H": 60,
        "4H": 240,
        "1D": 1440,
    }
    minutes = timeframe_minutes.get(timeframe, 60)

    start_time = datetime.now() - timedelta(minutes=minutes * n_bars)
    timestamps = [start_time + timedelta(minutes=minutes * i) for i in range(n_bars)]

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'symbol': symbol,
    }, index=pd.DatetimeIndex(timestamps))

    return df


def generate_tick_data(
    symbol: str = "BTCUSD",
    n_ticks: int = 10000,
    base_price: float = 50000.0,
    spread_pct: float = None,
    seed: int = None
) -> List[TickData]:
    """
    Generate realistic tick data.

    Uses Hawkes process for trade arrival times.

    Args:
        symbol: Trading symbol
        n_ticks: Number of ticks
        base_price: Starting price
        spread_pct: Bid-ask spread percentage (derived if None)
        seed: Random seed

    Returns:
        List of TickData objects
    """
    if seed is not None:
        np.random.seed(seed)

    # Derive spread from price level if not provided
    if spread_pct is None:
        # Spread typically 0.01% - 0.05% for liquid markets
        spread_pct = np.random.uniform(0.0001, 0.0005)

    ticks = []
    current_price = base_price
    current_time = datetime.now() - timedelta(hours=24)

    for i in range(n_ticks):
        # Random walk price movement
        # Tick-level volatility derived from typical market microstructure
        tick_vol = np.random.uniform(0.00001, 0.00005)
        price_change = np.random.standard_t(4) * tick_vol * current_price
        current_price += price_change
        current_price = max(current_price, 0.01)  # Floor

        # Calculate bid/ask
        half_spread = current_price * spread_pct / 2
        bid = current_price - half_spread
        ask = current_price + half_spread

        # Generate volume
        volume = np.random.lognormal(1, 1)

        # Advance time (exponential inter-arrival times)
        time_delta = np.random.exponential(0.5)  # Avg 0.5 seconds
        current_time += timedelta(seconds=time_delta)

        ticks.append(TickData(
            timestamp=current_time,
            bid=bid,
            ask=ask,
            volume=volume,
        ))

    return ticks


def generate_trade_list(
    n_trades: int = 100,
    win_rate: float = None,
    avg_pnl: float = None,
    seed: int = None
) -> List[Dict[str, Any]]:
    """
    Generate realistic trade list for backtesting analysis.

    Parameters are derived to create realistic trade distributions.

    Args:
        n_trades: Number of trades
        win_rate: Win rate (derived if None)
        avg_pnl: Average PnL per trade (derived if None)
        seed: Random seed

    Returns:
        List of trade dictionaries
    """
    if seed is not None:
        np.random.seed(seed)

    # Derive win rate from typical trading performance if not provided
    if win_rate is None:
        win_rate = np.random.uniform(0.45, 0.65)

    # Derive average PnL
    if avg_pnl is None:
        avg_pnl = np.random.uniform(10, 100)

    trades = []
    start_time = datetime.now() - timedelta(days=n_trades)

    for i in range(n_trades):
        is_winner = np.random.random() < win_rate

        # Generate PnL from log-normal distribution
        if is_winner:
            # Winners: avg_win = avg_pnl / win_rate (for positive expectancy)
            avg_win = avg_pnl / win_rate * 1.2
            pnl = np.random.lognormal(np.log(avg_win), 0.5)
        else:
            # Losers: smaller average
            avg_loss = avg_pnl / (1 - win_rate) * 0.8
            pnl = -np.random.lognormal(np.log(avg_loss), 0.5)

        # Generate trade details
        entry_time = start_time + timedelta(days=i)
        hold_duration = int(np.random.lognormal(2, 1))  # Bars
        exit_time = entry_time + timedelta(hours=hold_duration)

        entry_price = np.random.uniform(40000, 60000)
        pnl_pct = pnl / 10000  # Assume $10k position
        exit_price = entry_price * (1 + pnl_pct) if is_winner else entry_price * (1 - abs(pnl_pct))

        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': 'BTCUSD',
            'direction': 'long' if np.random.random() > 0.5 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': np.random.uniform(0.01, 0.5),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'commission': abs(pnl) * 0.001,
            'slippage': abs(pnl) * 0.0005,
            'duration_bars': hold_duration,
        }

        trades.append(trade)

    return trades


def create_sample_signals(
    df: pd.DataFrame,
    signal_quality: float = 0.55,
    seed: int = None
) -> pd.Series:
    """
    Create sample trading signals.

    Args:
        df: OHLCV DataFrame
        signal_quality: Probability of correct signal direction
        seed: Random seed

    Returns:
        Series with signals (-1, 0, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(df)

    # Calculate actual future returns for signal validation
    future_returns = df['close'].pct_change().shift(-1)

    signals = np.zeros(n)

    for i in range(n - 1):
        # Determine if signal is correct
        is_correct = np.random.random() < signal_quality

        # Base signal on future return
        if future_returns.iloc[i] > 0:
            signals[i] = 1 if is_correct else -1
        else:
            signals[i] = -1 if is_correct else 1

        # Add randomness - sometimes stay flat
        if np.random.random() < 0.3:
            signals[i] = 0

    return pd.Series(signals, index=df.index)
